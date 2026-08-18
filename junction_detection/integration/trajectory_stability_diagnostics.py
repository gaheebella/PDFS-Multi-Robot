"""Map-independent trajectory stability analysis for Branch tangents.

Ground-truth angles are consumed only after candidate selection, when errors
are scored.  The candidate itself uses realized relative motion, update order,
and local-neighbour consensus.  This module is diagnostic code; it does not
modify the production simulator or handoff controller.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1.0e-12


def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle to the half-open interval [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def signed_angle_delta_deg(first: float, second: float) -> float:
    """Return ``first - second`` along the shortest circular arc."""
    return normalize_angle_deg(float(first) - float(second))


def angular_error_deg(estimate: float, ground_truth: float) -> float:
    """Return unsigned axial-direction error in degrees."""
    return abs(signed_angle_delta_deg(estimate, ground_truth))


def circular_statistics_deg(angles: Sequence[float]) -> tuple[float, float, float]:
    """Return circular mean, dispersion in degrees, and resultant length."""
    if not angles:
        return float("nan"), float("nan"), 0.0
    radians = np.radians(np.asarray(angles, dtype=float))
    mean_cos = float(np.mean(np.cos(radians)))
    mean_sin = float(np.mean(np.sin(radians)))
    resultant = min(1.0, math.hypot(mean_cos, mean_sin))
    mean = normalize_angle_deg(math.degrees(math.atan2(mean_sin, mean_cos)))
    dispersion = (
        180.0
        if resultant <= EPSILON
        else min(180.0, math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))))
    )
    return mean, dispersion, resultant


def _percentile_ranks(values: Sequence[float], *, ascending: bool) -> np.ndarray:
    """Return deterministic empirical ranks in [0, 1]."""
    data = np.asarray(values, dtype=float)
    if data.size <= 1:
        return np.ones(data.size, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(data.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, data.size)
    return ranks if ascending else 1.0 - ranks


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous dictionaries as a stable UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def annotate_trajectory_stability(
    segment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add local-only temporal and cohort stability features to segments.

    The score is distribution-free: each component is converted to an
    empirical percentile within one case/Branch before the five percentiles
    are averaged.  No GT angle, map rectangle, world axis, or Branch label is
    read while constructing the score.
    """
    rows = [dict(row) for row in segment_rows]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)

    for group in groups.values():
        group.sort(key=lambda row: (int(row["frame"]), int(row["robot_id"])))
        previous_angle: dict[int, float] = {}
        persistence: dict[int, int] = defaultdict(int)
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            robot_id = int(row["robot_id"])
            angle = float(row["motion_angle_deg"])
            length = max(float(row["displacement_length"]), EPSILON)
            delta = (
                abs(signed_angle_delta_deg(angle, previous_angle[robot_id]))
                if robot_id in previous_angle
                else float("nan")
            )
            previous_angle[robot_id] = angle
            if math.isfinite(delta) and delta <= 10.0:
                persistence[robot_id] += 1
            else:
                persistence[robot_id] = 1
            row["temporal_delta_deg"] = delta
            row["curvature_deg_per_unit"] = delta / length if math.isfinite(delta) else float("nan")
            row["persistence_updates"] = persistence[robot_id]
            by_frame[int(row["frame"])].append(row)

        for frame_rows in by_frame.values():
            _, dispersion, resultant = circular_statistics_deg(
                [float(row["motion_angle_deg"]) for row in frame_rows]
            )
            for row in frame_rows:
                row["cohort_dispersion_deg"] = dispersion
                row["cohort_resultant_length"] = resultant

        finite_delta = [
            float(row["temporal_delta_deg"])
            if math.isfinite(float(row["temporal_delta_deg"])) else 180.0
            for row in group
        ]
        curvature = [
            float(row["curvature_deg_per_unit"])
            if math.isfinite(float(row["curvature_deg_per_unit"])) else 180.0
            for row in group
        ]
        resultant = [float(row["cohort_resultant_length"]) for row in group]
        persistence_values = [float(row["persistence_updates"]) for row in group]
        lengths = [float(row["displacement_length"]) for row in group]
        components = np.vstack([
            _percentile_ranks(finite_delta, ascending=False),
            _percentile_ranks(curvature, ascending=False),
            _percentile_ranks(resultant, ascending=True),
            _percentile_ranks(persistence_values, ascending=True),
            _percentile_ranks(lengths, ascending=True),
        ])
        scores = np.mean(components, axis=0)
        # Median is an empirical rank boundary, not a pixel/time/map constant.
        gate = float(np.median(scores))
        for row, score in zip(group, scores):
            row["stability_score"] = float(score)
            row["stable_candidate"] = bool(score >= gate)
            row["stability_gate_quantile"] = 0.50
    return rows


def equal_robot_robust_tangent(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Estimate tangent using component median after equal robot weighting."""
    by_robot: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if bool(row.get("stable_candidate", True)):
            by_robot[int(row["robot_id"])].append(float(row["motion_angle_deg"]))
    vectors = []
    for angles in by_robot.values():
        mean, _, resultant = circular_statistics_deg(angles)
        if resultant > EPSILON:
            vectors.append((math.cos(math.radians(mean)), math.sin(math.radians(mean))))
    if not vectors:
        return None
    x = float(np.median([vector[0] for vector in vectors]))
    y = float(np.median([vector[1] for vector in vectors]))
    if math.hypot(x, y) <= EPSILON:
        return None
    return normalize_angle_deg(math.degrees(math.atan2(y, x)))


def mouth_pca_tangent(
    mouth_points: Sequence[Sequence[float]],
    local_outward_angles: Sequence[float],
) -> float | None:
    """Reproduce transverse-mouth PCA and choose sign from local motion only."""
    points = np.asarray(mouth_points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return None
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, int(np.argmax(values))]
    tangent = np.asarray([normal[1], -normal[0]], dtype=float)
    flow_angle, _, flow_resultant = circular_statistics_deg(local_outward_angles)
    if flow_resultant > EPSILON:
        flow = np.asarray([math.cos(math.radians(flow_angle)), math.sin(math.radians(flow_angle))])
        if float(np.dot(tangent, flow)) < 0.0:
            tangent *= -1.0
    return normalize_angle_deg(math.degrees(math.atan2(tangent[1], tangent[0])))


def _summary_rows(annotated: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    summaries = []
    for (case_id, branch_id), group in sorted(groups.items()):
        progress = np.asarray([float(row["progress_fraction"]) for row in group])
        for label, mask in (
            ("TURNING", progress <= 1.0 / 3.0),
            ("MIDDLE", (progress > 1.0 / 3.0) & (progress < 2.0 / 3.0)),
            ("STABLE", progress >= 2.0 / 3.0),
        ):
            selected = [row for row, keep in zip(group, mask) if keep]
            if not selected:
                continue
            deltas = [float(row["temporal_delta_deg"]) for row in selected if math.isfinite(float(row["temporal_delta_deg"]))]
            curvatures = [float(row["curvature_deg_per_unit"]) for row in selected if math.isfinite(float(row["curvature_deg_per_unit"]))]
            _, dispersion, resultant = circular_statistics_deg([float(row["motion_angle_deg"]) for row in selected])
            summaries.append({
                "case_id": case_id,
                "branch_id": branch_id,
                "evaluation_epoch": label,
                "segment_count": len(selected),
                "robot_count": len({int(row["robot_id"]) for row in selected}),
                "median_angular_change_deg": float(np.median(deltas)) if deltas else float("nan"),
                "median_curvature_deg_per_unit": float(np.median(curvatures)) if curvatures else float("nan"),
                "circular_dispersion_deg": dispersion,
                "resultant_length": resultant,
                "median_persistence_updates": float(np.median([float(row["persistence_updates"]) for row in selected])),
                "median_displacement_length": float(np.median([float(row["displacement_length"]) for row in selected])),
                "stable_candidate_ratio": float(np.mean([bool(row["stable_candidate"]) for row in selected])),
            })
    return summaries


def _estimator_rows(
    cases: Sequence[Mapping[str, Any]],
    annotated: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    case_by_id = {str(row["case_id"]): row for row in cases}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in annotated:
        groups[(str(row["case_id"]), str(row["branch_id"]))].append(row)
    results = []
    for (case_id, branch_id), group in sorted(groups.items()):
        case = case_by_id[case_id]
        ground_truth = float(group[0]["gt_branch_angle_deg"])
        mouth_points_by_id: dict[int, tuple[float, float]] = {}
        for row in group:
            if "mouth_x" in row and "mouth_y" in row:
                mouth_points_by_id[int(row["robot_id"])] = (float(row["mouth_x"]), float(row["mouth_y"]))
        pca = mouth_pca_tangent(
            list(mouth_points_by_id.values()),
            [float(row["motion_angle_deg"]) for row in group],
        )
        # A frozen production measurement is an evaluation label and should
        # take precedence over reconstructed synthetic mouth points.  It never
        # participates in stability scoring or candidate sample selection.
        if group[0].get("pca_tangent_deg", "") != "":
            pca = float(group[0]["pca_tangent_deg"])
        candidate = equal_robot_robust_tangent(group)
        results.append({
            "case_id": case_id,
            "source": case.get("source", "unknown"),
            "seed": case.get("seed", ""),
            "way_count": case.get("way_count", ""),
            "branch_id": branch_id,
            "gt_tangent_deg": ground_truth,
            "pca_tangent_deg": "" if pca is None else pca,
            "pca_error_deg": "" if pca is None else angular_error_deg(pca, ground_truth),
            "candidate_tangent_deg": "" if candidate is None else candidate,
            "candidate_error_deg": "" if candidate is None else angular_error_deg(candidate, ground_truth),
            "length_group": case.get("length_group", ""),
            "corridor_width_group": case.get("corridor_width_group", ""),
            "rotation_deg": case.get("rotation_deg", ""),
            "stable_segment_count": sum(bool(row["stable_candidate"]) for row in group),
            "robot_count": len({int(row["robot_id"]) for row in group}),
        })
    return results


def _generalization_rows(estimator_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    dimensions = ("gt_tangent_deg", "length_group", "corridor_width_group", "seed", "rotation_deg")
    for dimension in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in estimator_rows:
            groups[str(row[dimension])].append(row)
        for value, group in sorted(groups.items()):
            for estimator in ("pca", "candidate"):
                errors = [float(row[f"{estimator}_error_deg"]) for row in group if row[f"{estimator}_error_deg"] != ""]
                if not errors:
                    continue
                output.append({
                    "group_dimension": dimension,
                    "group_value": value,
                    "estimator": estimator,
                    "count": len(errors),
                    "mean_error_deg": float(np.mean(errors)),
                    "median_error_deg": float(np.median(errors)),
                    "p90_error_deg": float(np.percentile(errors, 90)),
                    "max_error_deg": float(np.max(errors)),
                })
    return output


def _plot_results(
    directory: Path,
    annotated: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    estimator_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Create the benchmark plots required by the research request."""
    if not annotated or not estimator_rows:
        return
    example_key = (str(annotated[0]["case_id"]), str(annotated[0]["branch_id"]))
    example = [row for row in annotated if (str(row["case_id"]), str(row["branch_id"])) == example_key]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for robot_id in sorted({int(row["robot_id"]) for row in example})[:8]:
        robot_rows = [row for row in example if int(row["robot_id"]) == robot_id]
        axes[0].plot([float(row["progress_fraction"]) for row in robot_rows], [float(row["motion_angle_deg"]) for row in robot_rows], alpha=0.65)
    axes[0].set(title=f"Turning to stable motion: {example_key[0]} / {example_key[1]}", xlabel="normalized branch progress", ylabel="motion angle [deg]")
    axes[1].scatter([float(row["progress_fraction"]) for row in example], [float(row["stability_score"]) for row in example], c=[bool(row["stable_candidate"]) for row in example], s=9, cmap="coolwarm")
    axes[1].set(xlabel="normalized branch progress", ylabel="local-only stability score")
    fig.savefig(directory / "turning_vs_stable_motion_examples.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sampled = annotated[:: max(1, len(annotated) // 5000)]
    axis.scatter([float(row["progress_fraction"]) for row in sampled], [angular_error_deg(float(row["motion_angle_deg"]), float(row["gt_branch_angle_deg"])) for row in sampled], s=5, alpha=0.25)
    axis.set(title="Direction error over Branch progress", xlabel="normalized branch progress", ylabel="motion-to-GT error [deg]")
    fig.savefig(directory / "direction_error_over_branch_progress.png", dpi=160)
    plt.close(fig)

    pca = [float(row["pca_error_deg"]) for row in estimator_rows if row["pca_error_deg"] != ""]
    candidate = [float(row["candidate_error_deg"]) for row in estimator_rows if row["candidate_error_deg"] != ""]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot([pca, candidate], tick_labels=["mouth PCA", "stable motion"], showfliers=True)
    axis.set(title="Estimator angular error distribution", ylabel="absolute error [deg]")
    fig.savefig(directory / "pca_vs_candidate_error_distribution.png", dpi=160)
    plt.close(fig)

    def grouped_plot(filename: str, key: str, xlabel: str) -> None:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in estimator_rows:
            groups[str(row[key])].append(row)
        labels = list(sorted(groups))
        pca_means = [np.mean([float(row["pca_error_deg"]) for row in groups[label] if row["pca_error_deg"] != ""]) for label in labels]
        candidate_means = [np.mean([float(row["candidate_error_deg"]) for row in groups[label] if row["candidate_error_deg"] != ""]) for label in labels]
        x = np.arange(len(labels))
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        axis.plot(x, pca_means, "o-", label="mouth PCA")
        axis.plot(x, candidate_means, "o-", label="stable motion")
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set(title=f"Error vs {xlabel}", xlabel=xlabel, ylabel="mean absolute error [deg]")
        axis.legend()
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)

    grouped_plot("error_vs_branch_angle.png", "gt_tangent_deg", "Branch angle")
    grouped_plot("error_vs_branch_length.png", "length_group", "Branch length group")
    grouped_plot("error_vs_corridor_width.png", "corridor_width_group", "corridor width group")
    grouped_plot("rotation_invariance.png", "rotation_deg", "rigid rotation [deg]")


def analyze_benchmark(
    cases: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze rows, save all CSV/PNG artifacts, and classify Case A/B/C."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    annotated = annotate_trajectory_stability(segment_rows)
    summaries = _summary_rows(annotated)
    estimators = _estimator_rows(cases, annotated)
    generalization = _generalization_rows(estimators)
    _write_rows(directory / "branch_direction_cases.csv", cases)
    _write_rows(directory / "trajectory_segments.csv", annotated)
    _write_rows(directory / "trajectory_stability_summary.csv", summaries)
    _write_rows(directory / "branch_tangent_estimator_comparison.csv", estimators)
    _write_rows(directory / "branch_tangent_generalization_summary.csv", generalization)
    _plot_results(directory, annotated, summaries, estimators)

    turning = [row for row in summaries if row["evaluation_epoch"] == "TURNING"]
    stable = [row for row in summaries if row["evaluation_epoch"] == "STABLE"]
    turning_delta = float(np.median([float(row["median_angular_change_deg"]) for row in turning]))
    stable_delta = float(np.median([float(row["median_angular_change_deg"]) for row in stable]))
    turning_resultant = float(np.median([float(row["resultant_length"]) for row in turning]))
    stable_resultant = float(np.median([float(row["resultant_length"]) for row in stable]))
    pca_errors = [float(row["pca_error_deg"]) for row in estimators if row["pca_error_deg"] != ""]
    candidate_errors = [float(row["candidate_error_deg"]) for row in estimators if row["candidate_error_deg"] != ""]
    signal_exists = stable_delta < turning_delta and stable_resultant > turning_resultant
    improves = bool(candidate_errors and pca_errors and np.mean(candidate_errors) < np.mean(pca_errors))

    rotation_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in estimators:
        if row["candidate_error_deg"] != "":
            base_case = str(row["case_id"]).rsplit("_rot", 1)[0]
            rotation_groups[(base_case, str(row["branch_id"]))].append(float(row["candidate_error_deg"]))
    max_rotation_span = max((max(values) - min(values) for values in rotation_groups.values() if len(values) > 1), default=0.0)
    rotation_invariant = max_rotation_span <= 1.0e-7
    classification = "A" if signal_exists and improves and rotation_invariant else ("B" if signal_exists else "C")
    result = {
        "case": classification,
        "turning_median_delta_deg": turning_delta,
        "stable_median_delta_deg": stable_delta,
        "turning_median_resultant": turning_resultant,
        "stable_median_resultant": stable_resultant,
        "pca_mean_error_deg": float(np.mean(pca_errors)) if pca_errors else float("nan"),
        "pca_median_error_deg": float(np.median(pca_errors)) if pca_errors else float("nan"),
        "pca_p90_error_deg": float(np.percentile(pca_errors, 90)) if pca_errors else float("nan"),
        "pca_max_error_deg": float(np.max(pca_errors)) if pca_errors else float("nan"),
        "candidate_mean_error_deg": float(np.mean(candidate_errors)) if candidate_errors else float("nan"),
        "candidate_median_error_deg": float(np.median(candidate_errors)) if candidate_errors else float("nan"),
        "candidate_p90_error_deg": float(np.percentile(candidate_errors, 90)) if candidate_errors else float("nan"),
        "candidate_max_error_deg": float(np.max(candidate_errors)) if candidate_errors else float("nan"),
        "max_rotation_error_span_deg": max_rotation_span,
        "rotation_invariant": rotation_invariant,
        "case_count": len(cases),
        "branch_count": len(estimators),
        "segment_count": len(annotated),
    }
    (directory / "benchmark_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_synthetic_test() -> None:
    """Verify wrap-around, turning/stable scoring, and rotation invariance."""
    assert normalize_angle_deg(180.0) == -180.0
    assert angular_error_deg(-179.0, 179.0) == 2.0
    rows = []
    for rotation in (0.0, 47.0):
        case_id = f"test_rot{int(rotation)}"
        for robot_id in range(4):
            x = y = 0.0
            for update in range(8):
                angle = rotation + 24.0 * math.exp(-update / 1.5) + (robot_id - 1.5) * 0.15
                dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                x += dx
                y += dy
                rows.append({
                    "case_id": case_id, "branch_id": "B0", "robot_id": robot_id,
                    "frame": update, "motion_angle_deg": normalize_angle_deg(angle),
                    "displacement_length": 1.0, "progress_fraction": (update + 1) / 8,
                    "gt_branch_angle_deg": normalize_angle_deg(rotation), "mouth_x": 0.0,
                    "mouth_y": float(robot_id),
                })
    annotated = annotate_trajectory_stability(rows)
    estimates = []
    for case_id in ("test_rot0", "test_rot47"):
        estimates.append(equal_robot_robust_tangent([row for row in annotated if row["case_id"] == case_id]))
    assert estimates[0] is not None and estimates[1] is not None
    assert abs(angular_error_deg(estimates[0], 0.0) - angular_error_deg(estimates[1], 47.0)) < 1.0e-9


if __name__ == "__main__":
    run_synthetic_test()
    print("trajectory_stability_diagnostics synthetic test: PASS")
