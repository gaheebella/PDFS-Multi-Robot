"""Evaluate a localization-free Junction suspicion detector in shadow mode.

The detector consumes only continuous local SPH/topology and cheap-LiDAR
summaries produced by ``pre_exploration_general_pipeline_simulator``. Ground
truth is joined only after every detector update for evaluation and plotting.
No detector state changes robot motion or simulator state.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygame_simulator.pre_exploration_general_pipeline_simulator import (
    SAMPLE_PERIOD,
    PygameRenderer,
    SimulationRunner,
    run_headless,
)

DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/junction_shadow_detection"
CASES = ("M0_STRAIGHT", "M1_CROSS_BASELINE")
VARIANTS = ("sph_only", "lidar_only", "fusion")
BOOTSTRAP_SAMPLES = 3
CALIBRATION_FRACTION = 0.50
EMA_ALPHA = 0.50
PERSISTENCE_SAMPLES = 3
EPSILON = 1e-9

# This whitelist is the detector's complete runtime interface. In particular,
# it excludes map_case, gt_*, global positions, geometry and branch metadata.
RUNTIME_FEATURES = (
    "local_front_lateral_span",
    "local_front_lateral_variance",
    "motion_bearing_spread",
    "boundary_count",
    "boundary_fraction",
    "boundary_component_count",
    "boundary_largest_component_fraction",
    "boundary_second_component_fraction",
    "boundary_membership_retention",
    "mean_neighbor_degree",
    "lidar_left_wall_support",
    "lidar_right_wall_support",
    "lidar_forward_range",
    "lidar_free_space_angular_span",
    "lidar_range_profile_change",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write homogeneous dictionaries to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runtime_features(row: dict) -> dict[str, float]:
    """Copy the explicit localization-free feature contract from a row."""
    return {name: float(row[name]) for name in RUNTIME_FEATURES}


@dataclass(frozen=True)
class EvidenceThresholds:
    """Frozen M0-derived thresholds and their statistical margins."""

    sph: float
    boundary: float
    lidar: float
    sph_margin: float
    boundary_margin: float
    lidar_margin: float


# Frozen result of the existing M0 calibration. The clean GUI imports this
# immutable value; it does not recalibrate or alter detector logic at runtime.
FROZEN_GUI_THRESHOLDS = EvidenceThresholds(
    sph=0.05190502225289483,
    boundary=0.08701475362117486,
    lidar=0.012338797689551723,
    sph_margin=0.0,
    boundary_margin=0.0,
    lidar_margin=0.0,
)


class LocalEvidenceNormalizer:
    """Convert local signals to dimensionless changes from an initial baseline.

    The first three diagnostic samples are buffered without using GT. Their
    component-wise median is frozen as the corridor baseline. Three samples is
    the shortest median that rejects a one-frame spike and does not consume the
    short approach interval of the current M1 initial condition.
    """

    BASELINE_KEYS = (
        "local_front_lateral_span",
        "local_front_lateral_variance",
        "motion_bearing_spread",
        "boundary_fraction",
        "boundary_largest_component_fraction",
        "boundary_second_component_fraction",
        "mean_neighbor_degree",
        "lidar_free_space_angular_span",
    )

    def __init__(self) -> None:
        self.bootstrap: list[dict[str, float]] = []
        self.baseline: dict[str, float] | None = None
        self.smoothed = {"sph": 0.0, "boundary": 0.0, "lidar": 0.0}

    def update(self, values: dict[str, float]) -> dict[str, float]:
        """Return raw and EMA-smoothed evidence scores for one sample."""
        if self.baseline is None:
            self.bootstrap.append(values.copy())
            if len(self.bootstrap) >= BOOTSTRAP_SAMPLES:
                self.baseline = {
                    key: float(np.median([row[key] for row in self.bootstrap]))
                    for key in self.BASELINE_KEYS
                }

        base = self.baseline or {
            key: float(np.median([row[key] for row in self.bootstrap]))
            for key in self.BASELINE_KEYS
        }
        span_ratio = values["local_front_lateral_span"] / max(base["local_front_lateral_span"], EPSILON)
        variance_ratio = values["local_front_lateral_variance"] / max(base["local_front_lateral_variance"], EPSILON)
        # The median combines the two shape changes with motion dispersion, so
        # one transient velocity spike cannot become SPH expansion by itself.
        shape_terms = (
            max(0.0, span_ratio - 1.0),
            max(0.0, math.sqrt(max(variance_ratio, 0.0)) - 1.0),
            max(0.0, (values["motion_bearing_spread"] - base["motion_bearing_spread"]) / 45.0),
        )
        sph_score = float(np.median(shape_terms))

        largest_drop = max(
            0.0,
            base["boundary_largest_component_fraction"]
            - values["boundary_largest_component_fraction"],
        )
        second_growth = max(
            0.0,
            values["boundary_second_component_fraction"]
            - base["boundary_second_component_fraction"],
        )
        fraction_change = abs(values["boundary_fraction"] - base["boundary_fraction"])
        # A tiny fragment contributes little: score strength follows both the
        # second component and loss of the formerly dominant boundary body.
        boundary_score = largest_drop + second_growth + fraction_change

        side_wall_loss = 1.0 - min(
            values["lidar_left_wall_support"], values["lidar_right_wall_support"]
        )
        free_span_growth = max(
            0.0,
            values["lidar_free_space_angular_span"]
            - base["lidar_free_space_angular_span"],
        ) / 360.0
        scan_change = values["lidar_range_profile_change"] / 150.0
        lidar_score = side_wall_loss + free_span_growth + scan_change

        raw = {"sph": sph_score, "boundary": boundary_score, "lidar": lidar_score}
        for name, score in raw.items():
            self.smoothed[name] = EMA_ALPHA * score + (1.0 - EMA_ALPHA) * self.smoothed[name]

        return {
            "baseline_ready": self.baseline is not None,
            "lateral_span_ratio": span_ratio,
            "lateral_variance_ratio": variance_ratio,
            "motion_spread_delta": values["motion_bearing_spread"] - base["motion_bearing_spread"],
            "boundary_fraction_delta": values["boundary_fraction"] - base["boundary_fraction"],
            "largest_component_drop": largest_drop,
            "second_component_growth": second_growth,
            "neighbor_mean_ratio": values["mean_neighbor_degree"] / max(base["mean_neighbor_degree"], EPSILON),
            "lidar_side_wall_loss": side_wall_loss,
            "lidar_free_span_growth": free_span_growth,
            "sph_evidence_raw": sph_score,
            "boundary_evidence_raw": boundary_score,
            "lidar_evidence_raw": lidar_score,
            "sph_evidence_smoothed": self.smoothed["sph"],
            "boundary_evidence_smoothed": self.smoothed["boundary"],
            "lidar_evidence_smoothed": self.smoothed["lidar"],
            **{f"baseline_{key}": value for key, value in base.items()},
        }


class ShadowJunctionDetector:
    """Tri-evidence shadow detector with no actuation side effects."""

    def __init__(self, thresholds: EvidenceThresholds) -> None:
        self.thresholds = thresholds
        self.normalizer = LocalEvidenceNormalizer()
        self.sample_period = SAMPLE_PERIOD
        self.flag_runs = {"sph": 0, "boundary": 0, "lidar": 0}
        self.candidate_runs = {name: 0 for name in VARIANTS}
        self.trigger_runs = {name: 0 for name in VARIANTS}
        self.first_evidence_time = {name: math.nan for name in ("sph", "boundary", "lidar")}
        self.first_trigger_time = {name: math.nan for name in VARIANTS}

    def update(self, timestamp: float, values: dict[str, float]) -> dict:
        """Update detector state from a whitelisted local feature dictionary."""
        if set(values) != set(RUNTIME_FEATURES):
            raise ValueError("runtime detector input violates the local feature contract")
        evidence = self.normalizer.update(values)
        ready = bool(evidence["baseline_ready"])
        flags = {
            "sph": ready and evidence["sph_evidence_smoothed"] > self.thresholds.sph,
            "boundary": ready and evidence["boundary_evidence_smoothed"] > self.thresholds.boundary,
            "lidar": ready and evidence["lidar_evidence_smoothed"] > self.thresholds.lidar,
        }
        for name, flag in flags.items():
            self.flag_runs[name] = self.flag_runs[name] + 1 if flag else 0
            if flag and math.isnan(self.first_evidence_time[name]):
                self.first_evidence_time[name] = timestamp

        candidates = {
            "sph_only": flags["sph"] and flags["boundary"],
            "lidar_only": flags["lidar"],
            "fusion": flags["sph"] and flags["boundary"] and flags["lidar"],
        }
        triggers = {}
        for name, candidate in candidates.items():
            self.candidate_runs[name] = self.candidate_runs[name] + 1 if candidate else 0
            triggered = self.candidate_runs[name] >= PERSISTENCE_SAMPLES
            self.trigger_runs[name] = self.trigger_runs[name] + 1 if triggered else 0
            triggers[name] = triggered
            if triggered and math.isnan(self.first_trigger_time[name]):
                self.first_trigger_time[name] = timestamp

        boundary_count = int(round(values["boundary_count"]))
        second_fraction = values["boundary_second_component_fraction"]
        return {
            **evidence,
            "sph_threshold": self.thresholds.sph,
            "boundary_threshold": self.thresholds.boundary,
            "lidar_threshold": self.thresholds.lidar,
            "sph_threshold_margin": evidence["sph_evidence_smoothed"] - self.thresholds.sph,
            "boundary_threshold_margin": evidence["boundary_evidence_smoothed"] - self.thresholds.boundary,
            "lidar_threshold_margin": evidence["lidar_evidence_smoothed"] - self.thresholds.lidar,
            "boundary_second_component_size": int(round(values["boundary_count"] * second_fraction)),
            "boundary_largest_component_size": int(round(values["boundary_count"] * values["boundary_largest_component_fraction"])),
            "boundary_component_persistence": self.flag_runs["boundary"] * self.sample_period,
            "shadow_sph_expansion": flags["sph"],
            "shadow_boundary_change": flags["boundary"],
            "shadow_lidar_geometry_change": flags["lidar"],
            "shadow_sph_only_trigger": triggers["sph_only"],
            "shadow_lidar_only_trigger": triggers["lidar_only"],
            "shadow_fusion_trigger": triggers["fusion"],
            "shadow_junction_suspected": triggers["fusion"],
            "sph_evidence_dwell": self.flag_runs["sph"] * self.sample_period,
            "boundary_evidence_dwell": self.flag_runs["boundary"] * self.sample_period,
            "lidar_evidence_dwell": self.flag_runs["lidar"] * self.sample_period,
            "trigger_dwell": self.trigger_runs["fusion"] * self.sample_period,
            "boundary_component_count_runtime": boundary_count,
        }


def _threshold(values: list[float]) -> tuple[float, float]:
    """Return a conservative M0-only 99th-percentile plus robust 3-sigma limit."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    robust_sigma = 1.4826 * float(np.median(np.abs(array - median)))
    margin = max(3.0 * robust_sigma, 0.01)
    return float(np.quantile(array, 0.99) + margin), margin


def calibrate_thresholds(rows: list[dict]) -> tuple[EvidenceThresholds, list[dict]]:
    """Calibrate normalized evidence limits on the first half of M0 only."""
    if len(rows) < BOOTSTRAP_SAMPLES + 1:
        raise ValueError(
            f"at least {BOOTSTRAP_SAMPLES + 1} sampled rows are required for calibration"
        )
    normalizer = LocalEvidenceNormalizer()
    features = [normalizer.update(runtime_features(row)) for row in rows]
    usable = [row for row in features[: max(BOOTSTRAP_SAMPLES + 1, int(len(features) * CALIBRATION_FRACTION))] if row["baseline_ready"]]
    sph, sph_margin = _threshold([row["sph_evidence_smoothed"] for row in usable])
    boundary, boundary_margin = _threshold([row["boundary_evidence_smoothed"] for row in usable])
    lidar, lidar_margin = _threshold([row["lidar_evidence_smoothed"] for row in usable])
    return EvidenceThresholds(sph, boundary, lidar, sph_margin, boundary_margin, lidar_margin), features


def replay_detector(rows: list[dict], thresholds: EvidenceThresholds) -> list[dict]:
    """Replay one detector online, joining GT only after its update."""
    detector = ShadowJunctionDetector(thresholds)
    output = []
    for source in rows:
        detected = detector.update(float(source["timestamp"]), runtime_features(source))
        output.append({**source, **detected})
    return output


class LiveShadowRunner(SimulationRunner):
    """Simulation runner that appends shadow outputs without changing physics."""

    def __init__(self, case_id: str, thresholds: EvidenceThresholds) -> None:
        super().__init__(case_id, "local_forward")
        self.shadow_detector = ShadowJunctionDetector(thresholds)

    def step(self, frame: int):
        row = super().step(frame)
        if row is not None:
            row.update(self.shadow_detector.update(float(row["timestamp"]), runtime_features(row)))
        return row


def _episodes(rows: list[dict], field: str) -> list[tuple[float, float]]:
    """Return inclusive sampled trigger episodes."""
    episodes: list[tuple[float, float]] = []
    start = None
    previous = None
    for row in rows:
        time = float(row["timestamp"])
        if bool(row[field]) and start is None:
            start = time
        if not bool(row[field]) and start is not None:
            episodes.append((start, float(previous) + SAMPLE_PERIOD))
            start = None
        previous = time
    if start is not None and previous is not None:
        episodes.append((start, previous + SAMPLE_PERIOD))
    return episodes


def summarize(case: str, rows: list[dict]) -> list[dict]:
    """Calculate false-positive and detection-timing metrics per variant."""
    result = []
    positive_phases = {"OPENING_APPROACH", "BOUNDARY_CROSSING", "JUNCTION_REGION"}
    gt_onset = next((float(row["timestamp"]) for row in rows if row["gt_phase"] in positive_phases), math.nan)
    first_evidence = {
        "sph": next((float(row["timestamp"]) for row in rows if row["shadow_sph_expansion"]), math.nan),
        "boundary": next((float(row["timestamp"]) for row in rows if row["shadow_boundary_change"]), math.nan),
        "lidar": next((float(row["timestamp"]) for row in rows if row["shadow_lidar_geometry_change"]), math.nan),
    }
    for variant in VARIANTS:
        field = f"shadow_{variant}_trigger"
        episodes = _episodes(rows, field)
        first = episodes[0][0] if episodes else math.nan
        first_row = next((row for row in rows if bool(row[field])), None)
        durations = [end - start for start, end in episodes]
        false_rows = [row for row in rows if bool(row[field]) and row["gt_phase"] not in positive_phases]
        result.append({
            "map_case": case,
            "variant": variant,
            "sample_count": len(rows),
            "detection_success": bool(episodes) if case == "M1_CROSS_BASELINE" else not bool(episodes),
            "first_trigger_time": first,
            "gt_phase_at_first_trigger": first_row["gt_phase"] if first_row else "NONE",
            "gt_opening_approach_time": gt_onset,
            "trigger_lag_from_gt_onset": first - gt_onset if math.isfinite(first) and math.isfinite(gt_onset) else math.nan,
            "trigger_episode_count": len(episodes),
            "trigger_fraction": float(np.mean([bool(row[field]) for row in rows])),
            "longest_trigger_duration": max(durations, default=0.0),
            "total_trigger_duration": sum(durations),
            "false_early_trigger_sample_count": len(false_rows),
            "first_sph_evidence_time": first_evidence["sph"],
            "first_boundary_evidence_time": first_evidence["boundary"],
            "first_lidar_evidence_time": first_evidence["lidar"],
        })
    return result


def _save_plot(m0: list[dict], m1: list[dict], output: Path) -> None:
    """Save the requested eight-panel M0/M1 shadow audit."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex="col")
    cases = (("M0 Straight", m0), ("M1 Cross", m1))
    phase_colors = {"OPENING_APPROACH": "#ffe8a3", "BOUNDARY_CROSSING": "#ffc8a3", "JUNCTION_REGION": "#ffaaa3"}
    for column, (title, rows) in enumerate(cases):
        time = np.asarray([row["timestamp"] for row in rows])
        for phase, color in phase_colors.items():
            mask = np.asarray([row["gt_phase"] == phase for row in rows])
            if np.any(mask):
                for axis in axes[:, column]:
                    axis.fill_between(time, 0, 1, where=mask, color=color, alpha=0.18, transform=axis.get_xaxis_transform())
        axes[0, column].plot(time, [row["local_front_lateral_span"] for row in rows], label="lateral span")
        axes[0, column].plot(time, [math.sqrt(max(row["local_front_lateral_variance"], 0.0)) for row in rows], label="sqrt variance")
        axes[0, column].legend(fontsize=8); axes[0, column].set_title(title)
        axes[1, column].plot(time, [row["boundary_fraction"] for row in rows], label="boundary fraction")
        axes[1, column].plot(time, [row["boundary_largest_component_fraction"] for row in rows], label="largest B-comp frac")
        axes[1, column].plot(time, [row["boundary_second_component_fraction"] for row in rows], label="second B-comp frac")
        axes[1, column].step(time, [row["boundary_component_count"] for row in rows], where="post", label="B-comp count", alpha=.7)
        axes[1, column].legend(fontsize=7)
        axes[2, column].plot(time, [row["mean_neighbor_degree"] for row in rows], label="neighbor mean")
        axes[2, column].plot(time, [row["lidar_left_wall_support"] for row in rows], label="LiDAR left support")
        axes[2, column].plot(time, [row["lidar_right_wall_support"] for row in rows], label="LiDAR right support")
        axes[2, column].plot(time, np.asarray([row["lidar_forward_range"] for row in rows]) / 150.0, label="forward / max")
        axes[2, column].plot(time, np.asarray([row["lidar_free_space_angular_span"] for row in rows]) / 180.0, label="free span / 180")
        axes[2, column].plot(time, np.asarray([row["lidar_range_profile_change"] for row in rows]) / 15.0, label="scan change / 15", alpha=.7)
        axes[2, column].legend(fontsize=7)
        axes[3, column].step(time, [row["shadow_sph_only_trigger"] for row in rows], where="post", label="SPH-only")
        axes[3, column].step(time, np.asarray([row["shadow_lidar_only_trigger"] for row in rows]) * .7, where="post", label="LiDAR-only")
        axes[3, column].step(time, np.asarray([row["shadow_fusion_trigger"] for row in rows]) * .4, where="post", label="Fusion")
        axes[3, column].set_ylim(-.05, 1.1); axes[3, column].set_xlabel("time [s]"); axes[3, column].legend(fontsize=8)
    axes[0, 0].set_ylabel("local spread")
    axes[1, 0].set_ylabel("boundary topology")
    axes[2, 0].set_ylabel("neighbor / LiDAR")
    axes[3, 0].set_ylabel("shadow trigger")
    fig.suptitle("Localization-free Junction shadow detection (GT shading: evaluation only)")
    fig.tight_layout()
    fig.savefig(output / "junction_shadow_detection_audit.png", dpi=150)
    plt.close(fig)


def _run_gui_m1(frames: int, thresholds: EvidenceThresholds, gui_scale: float) -> LiveShadowRunner:
    """Run bounded M1 with the normal renderer and a shadow-only runner."""
    import pygame

    runner = LiveShadowRunner("M1_CROSS_BASELINE", thresholds)
    renderer = PygameRenderer(runner.geometry, gui_scale=gui_scale, show_gt=True)
    frame = 0
    running = True
    while running and frame < frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        runner.step(frame)
        frame += 1
        renderer.draw(runner, frame)
    pygame.quit()
    return runner


def run_experiment(frames: int, output: Path, gui_m1: bool, gui_scale: float) -> dict:
    """Run M0/M1, write minimal artifacts, and return the verdict row."""
    output.mkdir(parents=True, exist_ok=True)
    m0_raw = run_headless("M0_STRAIGHT", frames, "local_forward").rows
    thresholds, _ = calibrate_thresholds(m0_raw)
    m0 = replay_detector(m0_raw, thresholds)
    if gui_m1:
        m1 = _run_gui_m1(frames, thresholds, gui_scale).rows
    else:
        m1_raw = run_headless("M1_CROSS_BASELINE", frames, "local_forward").rows
        m1 = replay_detector(m1_raw, thresholds)

    _write_csv(output / "junction_shadow_timeline_m0.csv", m0)
    _write_csv(output / "junction_shadow_timeline_m1.csv", m1)
    summary = summarize("M0_STRAIGHT", m0) + summarize("M1_CROSS_BASELINE", m1)
    _write_csv(output / "junction_shadow_summary.csv", summary)
    _write_csv(output / "junction_shadow_detector_comparison.csv", summary)

    m0_fusion = next(row for row in summary if row["map_case"] == "M0_STRAIGHT" and row["variant"] == "fusion")
    m1_fusion = next(row for row in summary if row["map_case"] == "M1_CROSS_BASELINE" and row["variant"] == "fusion")
    if m0_fusion["trigger_episode_count"] == 0 and m1_fusion["trigger_episode_count"] > 0:
        verdict_name = "A. SHADOW_JUNCTION_DETECTION_VALID"
    elif m1_fusion["trigger_episode_count"] > 0:
        verdict_name = "B. PARTIALLY_VALID"
    else:
        individual_detects = any(
            row["trigger_episode_count"] > 0
            for row in summary
            if row["map_case"] == "M1_CROSS_BASELINE" and row["variant"] != "fusion"
        )
        verdict_name = "C. SIGNALS_INSUFFICIENT" if individual_detects else "D. INVALID"
    verdict = {
        "verdict": verdict_name,
        "frames_per_case": frames,
        "sample_period": SAMPLE_PERIOD,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "calibration_source": "first_half_of_M0_local_observables_only",
        "sph_threshold": thresholds.sph,
        "boundary_threshold": thresholds.boundary,
        "lidar_threshold": thresholds.lidar,
        "persistence_samples": PERSISTENCE_SAMPLES,
        "m0_fusion_false_trigger_count": m0_fusion["trigger_episode_count"],
        "m0_fusion_false_trigger_fraction": m0_fusion["trigger_fraction"],
        "m0_fusion_longest_false_trigger": m0_fusion["longest_trigger_duration"],
        "m1_fusion_first_trigger_time": m1_fusion["first_trigger_time"],
        "m1_fusion_gt_phase_at_trigger_eval_only": m1_fusion["gt_phase_at_first_trigger"],
        "m1_fusion_trigger_lag": m1_fusion["trigger_lag_from_gt_onset"],
        "m1_fusion_longest_trigger_duration": m1_fusion["longest_trigger_duration"],
        "runtime_feature_contract": "|".join(RUNTIME_FEATURES),
        "front_quantile_used_by_detector": False,
        "gt_or_global_geometry_used_by_detector": False,
        "physics_actuation_changed_by_detector": False,
    }
    _write_csv(output / "junction_shadow_verdict.csv", [verdict])
    _save_plot(m0, m1, output)
    return verdict


def parse_args(argv=None):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gui-m1", action="store_true")
    parser.add_argument("--gui-scale", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the bounded shadow experiment."""
    args = parse_args(argv)
    verdict = run_experiment(args.frames, args.output_dir, args.gui_m1, args.gui_scale)
    print(f"verdict={verdict['verdict']} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
