"""Read-only threshold sensitivity diagnostic for adaptive Anchor stopping.

Every alpha receives a fresh AdaptiveSession/world.  The imported detector,
physics, grouping, confirmation, and Anchor implementation are never mutated.
The known M1 entrance is used only after detection for evaluation columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import lidar_junction_detection_adaptive_w_tau_anchor_stop as adaptive


DEFAULT_ALPHAS = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95,
)
REPRESENTATIVE_ALPHAS = (0.10, 0.50, 0.90)
JUNCTION_ENTRANCE_Y_EVAL_ONLY = -42.0
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "adaptive_threshold_sensitivity_output"
)


def _empty(value: Any) -> Any:
    return "" if value is None else value


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (tuple(rows[0]) if rows else ("status",)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _circular_distance_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _snapshot_at_frame(
    session: adaptive.AdaptiveSession,
    frame: int | None,
) -> adaptive.AdaptiveSnapshot | None:
    if frame is None:
        return None
    return next(
        (
            snapshot for snapshot in session.snapshots
            if snapshot.physics_frame == frame
        ),
        None,
    )


def _opening_rows(
    alpha: float,
    before: adaptive.AdaptiveSnapshot | None,
    detected: adaptive.AdaptiveSnapshot | None,
) -> list[dict[str, Any]]:
    phases = (("BEFORE_DETECTION", before), ("FIRST_DETECTION", detected))
    new_index: int | None = None
    if before is not None and detected is not None and detected.opening_groups:
        before_centers = [
            float(group["center_angle"]) for group in before.opening_groups
        ]
        if before_centers:
            novelty = [
                min(
                    _circular_distance_deg(
                        float(group["center_angle"]), center
                    )
                    for center in before_centers
                )
                for group in detected.opening_groups
            ]
            new_index = int(max(range(len(novelty)), key=novelty.__getitem__))
        else:
            new_index = 0

    rows: list[dict[str, Any]] = []
    for phase, snapshot in phases:
        if snapshot is None:
            continue
        for index, opening in enumerate(snapshot.opening_groups):
            selected = snapshot.adaptive_selected_threshold
            mean_range = float(opening["mean_range_m"])
            peak_range = float(opening["peak_range_m"])
            rows.append(
                {
                    "alpha": alpha,
                    "phase": phase,
                    "frame": snapshot.physics_frame,
                    "timestamp": snapshot.timestamp,
                    "opening_count": len(snapshot.opening_groups),
                    "opening_index": index,
                    "is_new_third_opening": bool(
                        phase == "FIRST_DETECTION" and index == new_index
                    ),
                    "start_angle": opening["start_angle"],
                    "end_angle": opening["end_angle"],
                    "center_angle": opening["center_angle"],
                    "width_deg": opening["width_deg"],
                    "mean_range": mean_range,
                    "peak_range": peak_range,
                    "confidence": opening["confidence"],
                    "runtime_selected_threshold": _empty(selected),
                    "mean_minus_threshold": (
                        "" if selected is None else mean_range - selected
                    ),
                    "peak_minus_threshold": (
                        "" if selected is None else peak_range - selected
                    ),
                    "runtime_gt_map_used": False,
                }
            )
    return rows


def run_alpha(
    alpha: float,
    *,
    primary_horizon: int,
    extended_horizon: int,
) -> tuple[adaptive.AdaptiveSession, dict[str, Any], list[dict[str, Any]]]:
    config = adaptive.DetectorExperimentConfig(
        threshold_mode="w-tau",
        worst_wall_range=100.0,
        noise_model="none",
        noise_fraction=adaptive.DEFAULT_NOISE_FRACTION,
        noise_seed=adaptive.DEFAULT_NOISE_SEED,
        threshold_alpha=float(alpha),
        smoothing_window=5,
        anchor_stop_on_detect=True,
        evaluate_interval=False,
    )
    # A fresh world is constructed here for every alpha.  No trajectory or
    # fixed-Anchor state is shared with another run.
    session = adaptive.AdaptiveSession(
        adaptive.M1_PRE_CORRIDOR_CASE,
        config,
    )
    while session.next_physics_frame < primary_horizon:
        session.advance_physics_frame()
    if session.first_detection_frame is None:
        while (
            session.next_physics_frame < extended_horizon
            and session.first_detection_frame is None
        ):
            session.advance_physics_frame()

    detected = session.first_detection_snapshot()
    detection_index = next(
        (
            index for index, snapshot in enumerate(session.snapshots)
            if snapshot.physics_frame == session.first_detection_frame
        ),
        None,
    )
    before = (
        None if detection_index is None or detection_index == 0
        else session.snapshots[detection_index - 1]
    )
    fix_y = (
        None if session.anchor_fix_position is None
        else float(session.anchor_fix_position[1])
    )
    row = {
        "alpha": alpha,
        "run_world_identity": id(session.runner.world),
        "primary_horizon": primary_horizon,
        "extended_horizon": extended_horizon,
        "no_detection_within_480_frames": bool(
            session.first_detection_frame is None
            and extended_horizon >= 480
        ),
        "first_open_support_frame": _empty(session.first_open_support_frame),
        "first_open_support_time": _empty(session.first_open_support_time),
        "first_opening_frame": _empty(session.first_opening_frame),
        "first_opening_time": _empty(session.first_opening_time),
        "first_detection_frame": _empty(session.first_detection_frame),
        "first_detection_time": _empty(session.first_detection_time),
        "anchor_fix_frame": _empty(session.anchor_fix_frame),
        "anchor_fix_time": _empty(session.anchor_fix_time),
        "anchor_fix_x_eval_only": (
            "" if session.anchor_fix_position is None
            else float(session.anchor_fix_position[0])
        ),
        "anchor_fix_y_eval_only": _empty(fix_y),
        "junction_entrance_y_eval_only": JUNCTION_ENTRANCE_Y_EVAL_ONLY,
        "distance_to_entrance_eval_only": (
            "" if fix_y is None
            else abs(fix_y - JUNCTION_ENTRANCE_Y_EVAL_ONLY)
        ),
        "left_lateral_range_at_detection": (
            "" if detected is None else _empty(detected.left_lateral_range)
        ),
        "right_lateral_range_at_detection": (
            "" if detected is None else _empty(detected.right_lateral_range)
        ),
        "estimated_corridor_width_at_detection": (
            "" if detected is None
            else _empty(detected.estimated_corridor_width)
        ),
        "estimated_lateral_offset_at_detection": (
            "" if detected is None
            else _empty(detected.estimated_lateral_offset)
        ),
        "adaptive_w_at_detection": (
            "" if detected is None else detected.adaptive_worst_wall_range
        ),
        "adaptive_margin_ratio": adaptive.ADAPTIVE_W_MARGIN_RATIO,
        "adaptive_tmin_at_detection": (
            "" if detected is None else detected.adaptive_lower_bound
        ),
        "adaptive_tmax_at_detection": (
            "" if detected is None else detected.adaptive_upper_bound
        ),
        "runtime_selected_threshold_at_detection": (
            "" if detected is None
            else _empty(detected.adaptive_selected_threshold)
        ),
        "opening_count_at_detection": (
            "" if detected is None else len(detected.opening_groups)
        ),
        "open_support_count_at_detection": (
            "" if detected is None
            else int(detected.open_support_mask.sum())
        ),
        "opening_count_before_detection": (
            "" if before is None else len(before.opening_groups)
        ),
        "before_detection_frame": (
            "" if before is None else before.physics_frame
        ),
        "adaptive_estimate_source_at_detection": (
            "" if detected is None else detected.adaptive_estimate_source
        ),
        "post_fix_max_position_drift": session.post_fix_max_position_drift,
        "runtime_gt_map_used": False,
        "evaluation_gt_used": True,
    }
    details = (
        _opening_rows(alpha, before, detected)
        if any(math.isclose(alpha, value) for value in REPRESENTATIVE_ALPHAS)
        else []
    )
    return session, row, details


def render_representative_frames(
    output: Path,
    alpha: float,
    session: adaptive.AdaptiveSession,
) -> None:
    import pygame

    detected = session.first_detection_snapshot()
    if detected is None:
        return
    detection_index = next(
        index for index, snapshot in enumerate(session.snapshots)
        if snapshot is detected
    )
    indices = (
        ("before_detection", detection_index - 1),
        ("first_detection", detection_index),
    )
    pygame.init()
    renderer = adaptive.AdaptiveRenderer(
        pygame, session.runner.geometry, show_profile=True
    )
    original_index = session.view_index
    for label, index in indices:
        if index < 0:
            continue
        session.view_index = index
        renderer.draw(session, paused=True)
        name = f"alpha_{round(alpha * 100):03d}_{label}.png"
        pygame.image.save(renderer.screen, output / name)
    session.view_index = original_index
    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--primary-horizon", type=int, default=300)
    parser.add_argument("--extended-horizon", type=int, default=480)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-png", action="store_true")
    args = parser.parse_args()
    if args.primary_horizon < 300:
        parser.error("--primary-horizon must be at least 300")
    if args.extended_horizon < args.primary_horizon:
        parser.error("--extended-horizon must not be smaller than primary")
    if any(not 0.0 < alpha < 1.0 for alpha in args.alphas):
        parser.error("all alphas must be strictly between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    adaptive._audit_frozen_detector_defaults()
    adaptive._audit_adaptive_math_and_localization()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    opening_rows: list[dict[str, Any]] = []
    world_ids: list[int] = []
    retained_sessions: list[adaptive.AdaptiveSession] = []
    for alpha in args.alphas:
        session, summary, details = run_alpha(
            alpha,
            primary_horizon=args.primary_horizon,
            extended_horizon=args.extended_horizon,
        )
        summary_rows.append(summary)
        opening_rows.extend(details)
        world_ids.append(int(summary["run_world_identity"]))
        retained_sessions.append(session)
        if args.render_png and any(
            math.isclose(alpha, value) for value in REPRESENTATIVE_ALPHAS
        ):
            render_representative_frames(args.output_dir, alpha, session)
        print(json.dumps(summary, sort_keys=True))
    if len(set(world_ids)) != len(world_ids):
        raise AssertionError("alpha runs unexpectedly shared a world identity")
    _write_csv(args.output_dir / "alpha_sensitivity_summary.csv", summary_rows)
    _write_csv(
        args.output_dir / "representative_opening_transition.csv",
        opening_rows,
        tuple(opening_rows[0]) if opening_rows else ("status",),
    )
    _write_csv(
        args.output_dir / "diagnostic_verdict_inputs.csv",
        [
            {
                "map_case": adaptive.M1_PRE_CORRIDOR_CASE,
                "noise_model": "none",
                "fallback_w": 100.0,
                "adaptive_margin_ratio": adaptive.ADAPTIVE_W_MARGIN_RATIO,
                "tau": 150.0 * adaptive.DEFAULT_NOISE_FRACTION,
                "adaptive_tmax": (
                    150.0 - 150.0 * adaptive.DEFAULT_NOISE_FRACTION
                ),
                "independent_world_per_alpha": True,
                "runtime_gt_map_used": False,
                "evaluation_gt_used": True,
            }
        ],
    )


if __name__ == "__main__":
    main()
