"""Audit the frozen LiDAR corridor-model lifecycle before M1 detection.

The audit intentionally does not implement a detector variant. The current
detector already holds its stable model across invalid current wall fits, but
M1 never produces a valid fit before its detection scan. Consequently there
is no causal last-valid model to persist at frame 120.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    LIDAR_MAX_RANGE,
    SimulationRunner,
)


MAP_CASE = "M1_CROSS_BASELINE"
END_FRAME = 126
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/corridor_model_lifecycle"
LIFECYCLE_FIELDS = (
    "frame",
    "time",
    "side_walls_valid",
    "width_observation",
    "offset_observation",
    "current_corridor_orientation_deg",
    "left_wall_orientation_deg",
    "right_wall_orientation_deg",
    "parallel_error_deg",
    "estimated_corridor_width",
    "estimated_offset",
    "stable_corridor_orientation_deg",
    "current_model_valid",
    "last_valid_model_exists_before_scan",
    "corridor_model_initialized",
    "corridor_model_just_initialized",
    "corridor_model_update_count",
    "corridor_model_frozen",
    "model_lifecycle_state",
    "expected_profile_source",
    "max_range_fallback_used",
    "expected_profile_min",
    "expected_profile_max",
    "open_candidate_count",
    "opening_group_count",
    "junction_detected",
    "lidar_x_eval_only",
    "lidar_y_eval_only",
)


def _source_and_state(row: dict[str, Any]) -> tuple[str, str, bool]:
    """Label existing detector branches without changing their computation."""
    initialized_before_scan = bool(row["corridor_model_initialized"]) and not bool(
        row["corridor_model_just_initialized"]
    )
    if initialized_before_scan:
        source = "STABLE_LAST_VALID_MODEL"
        state = (
            "CURRENT_MODEL_VALID"
            if row["side_walls_valid"]
            else "LAST_VALID_MODEL_HELD"
        )
        return source, state, False
    if row["side_walls_valid"]:
        return "CURRENT_WALL_FIT", "CURRENT_MODEL_VALID", False
    return "MAX_RANGE_FALLBACK", "MODEL_UNINITIALIZED", True


def run_lifecycle() -> list[dict[str, Any]]:
    detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner = SimulationRunner(
        MAP_CASE,
        "local_forward",
        profile_detector=detector,
        hold_on_profile_detection=False,
    )
    rows: list[dict[str, Any]] = []
    prior_valid_exists = False
    for frame in range(END_FRAME + 1):
        sampled = runner.step(frame)
        if sampled is None:
            continue
        source, state, fallback = _source_and_state(sampled)
        leader = next(
            robot for robot in runner.world.robots
            if robot.robot_id == runner.world.lidar_robot_id
        )
        rows.append(
            {
                "frame": frame,
                "time": float(sampled["timestamp"]),
                "side_walls_valid": bool(sampled["side_walls_valid"]),
                "width_observation": sampled["width_observation"],
                "offset_observation": sampled["offset_observation"],
                "current_corridor_orientation_deg": sampled[
                    "current_corridor_orientation_deg"
                ],
                "left_wall_orientation_deg": sampled["left_wall_orientation_deg"],
                "right_wall_orientation_deg": sampled["right_wall_orientation_deg"],
                "parallel_error_deg": sampled["parallel_error_deg"],
                "estimated_corridor_width": sampled["estimated_corridor_width"],
                "estimated_offset": sampled["estimated_offset"],
                "stable_corridor_orientation_deg": sampled[
                    "stable_corridor_orientation_deg"
                ],
                "current_model_valid": bool(sampled["side_walls_valid"]),
                "last_valid_model_exists_before_scan": prior_valid_exists,
                "corridor_model_initialized": bool(
                    sampled["corridor_model_initialized"]
                ),
                "corridor_model_just_initialized": bool(
                    sampled["corridor_model_just_initialized"]
                ),
                "corridor_model_update_count": int(
                    sampled["corridor_model_update_count"]
                ),
                "corridor_model_frozen": bool(sampled["corridor_model_frozen"]),
                "model_lifecycle_state": state,
                "expected_profile_source": source,
                "max_range_fallback_used": fallback,
                "expected_profile_min": sampled["expected_profile_min"],
                "expected_profile_max": sampled["expected_profile_max"],
                "open_candidate_count": int(sampled["opening_candidate_count"]),
                "opening_group_count": int(sampled["opening_group_count"]),
                "junction_detected": bool(sampled["profile_junction_detected"]),
                "lidar_x_eval_only": float(leader.position[0]),
                "lidar_y_eval_only": float(leader.position[1]),
            }
        )
        prior_valid_exists |= bool(sampled["side_walls_valid"])
    return rows


def _first_frame(rows: list[dict[str, Any]], key: str) -> int | None:
    return next((int(row["frame"]) for row in rows if row[key]), None)


def _audit_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detection_frame = _first_frame(rows, "junction_detected")
    pre_detection = [
        row for row in rows if detection_frame is None or row["frame"] < detection_frame
    ]
    first_valid = _first_frame(rows, "side_walls_valid")
    first_initialized = _first_frame(rows, "corridor_model_initialized")
    frame120 = next(row for row in rows if row["frame"] == 120)
    valid_before_detection = any(row["side_walls_valid"] for row in pre_detection)
    stable_before_detection = any(
        row["corridor_model_initialized"] for row in pre_detection
    )
    last_valid_available_at_120 = bool(
        frame120["last_valid_model_exists_before_scan"]
    )
    return {
        "question_a_valid_model_ever_before_detection": valid_before_detection,
        "question_b_first_valid_frame": "" if first_valid is None else first_valid,
        "question_b_first_initialized_stable_frame": (
            "" if first_initialized is None else first_initialized
        ),
        "question_c_frame120_only_current_fit_invalid": bool(
            not frame120["current_model_valid"] and last_valid_available_at_120
        ),
        "question_d_last_valid_could_supply_frame120_expected": last_valid_available_at_120,
        "stable_model_exists_before_detection": stable_before_detection,
        "frame120_expected_profile_source": frame120["expected_profile_source"],
        "frame120_expected_profile_min": frame120["expected_profile_min"],
        "frame120_expected_profile_max": frame120["expected_profile_max"],
        "first_open_candidate_frame": _first_frame(rows, "open_candidate_count"),
        "first_junction_detection_frame": detection_frame,
    }


def _old_vs_new(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answers = _audit_answers(rows)
    common = {
        "frame120_expected_source": answers["frame120_expected_profile_source"],
        "frame120_expected_min": answers["frame120_expected_profile_min"],
        "frame120_expected_max": answers["frame120_expected_profile_max"],
        "first_open_frame": answers["first_open_candidate_frame"],
        "first_detection_frame": answers["first_junction_detection_frame"],
    }
    return [
        {
            "variant": "OLD_CURRENT_DETECTOR",
            "implementation_status": "EXECUTED",
            "timing_change_frames": 0,
            **common,
        },
        {
            "variant": "PROPOSED_LAST_VALID_PERSISTENCE",
            "implementation_status": "NOT_IMPLEMENTED_NO_PRIOR_VALID_MODEL",
            "timing_change_frames": 0,
            **common,
        },
    ]


def _verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = _audit_answers(rows)
    return {
        "verdict": "DIAGNOSTIC_ONLY_NO_PERSISTENCE_PRECONDITION",
        "root_cause": (
            "M1 has no valid side-wall or initialized stable corridor model before "
            "frame 126; frame 120 therefore has no causal last-valid model to hold"
        ),
        **answers,
        "existing_stable_model_persistence_present": True,
        "detector_modified": False,
        "threshold_modified": False,
        "grouping_modified": False,
        "simulator_modified": False,
        "gt_or_map_used_for_runtime": False,
    }


def _write(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    lifecycle = run_lifecycle()
    comparisons = _old_vs_new(lifecycle)
    verdict = _verdict(lifecycle)
    _write(args.output / "corridor_model_lifecycle.csv", lifecycle, LIFECYCLE_FIELDS)
    _write(args.output / "old_vs_new_detection_timing.csv", comparisons)
    _write(args.output / "verdict.csv", [verdict])
    print(
        f"verdict={verdict['verdict']} output={args.output.resolve()} "
        f"first_valid={verdict['question_b_first_valid_frame']} "
        f"first_detection={verdict['first_junction_detection_frame']}"
    )


if __name__ == "__main__":
    main()
