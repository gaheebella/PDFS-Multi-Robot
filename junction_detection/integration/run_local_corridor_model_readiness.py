"""EXP-043 local-only corridor-model readiness feasibility audit.

The shadow readiness rule reuses one existing detector semantic only:
``corridor_model_initialized``. A second evaluation-only adapter suppresses
publication of raw Junction detections before readiness, without changing the
frozen detector or its inputs. GT/map values are appended post-hoc only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import (  # noqa: E402
    REAR_START_SHIFT,
)
from junction_detection.integration.run_pre_corridor_bootstrap_detection_timing import (  # noqa: E402
    BASELINE_CASE,
    BOOTSTRAP_CASE,
    M0_CASE,
    run_case,
)
from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    BASELINE_CORRIDOR_WIDTH,
    LIDAR_MAX_RANGE,
)


EXPERIMENT_ID = "EXP-043"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/local_corridor_model_readiness"
# This is the detector's existing initialization evidence count, not a new
# readiness threshold.
READINESS_WINDOW = GeometryProfileConfig(LIDAR_MAX_RANGE).initialization_scan_count
FEATURE_FIELDS = (
    "case_id",
    "frame",
    "time",
    "side_walls_valid",
    "parallel_error_deg",
    "current_width",
    "current_offset",
    "current_orientation",
    "stable_width",
    "stable_offset",
    "stable_orientation",
    "stable_update_count",
    "current_model_valid",
    "stable_model_initialized",
    "width_change_from_previous_valid",
    "orientation_change_from_previous_valid",
    "offset_change_from_previous_valid",
    "consecutive_valid_count",
    "valid_ratio_existing_init_window",
    "expected_profile_source",
    "readiness_state",
    "readiness_reason",
    "corridor_model_ready",
    "semantic_detector_state",
    "open_candidate_count",
    "opening_group_count",
    "junction_detected_raw",
    "junction_detected_gated_eval",
    "readiness_rising",
    "false_readiness_eval_only",
    "opening_visible_eval_only",
)


def _finite(value: Any) -> bool:
    return math.isfinite(float(value))


def _current_model_valid(row: dict[str, Any]) -> bool:
    return bool(
        row["side_walls_valid"]
        and _finite(row["current_width"])
        and _finite(row["current_offset"])
        and _finite(row["current_orientation"])
    )


def _opening_visible_eval_only(row: dict[str, Any]) -> bool:
    if row["map_case"] == "M0_STRAIGHT":
        return False
    entry_distance = float(row["distance_to_junction_entry_eval_only"])
    x = float(row["leader_x_eval_only"])
    half_width = 0.5 * BASELINE_CORRIDOR_WIDTH
    corner_distance = min(
        math.hypot(x - side, entry_distance) for side in (-half_width, half_width)
    )
    return corner_distance <= LIDAR_MAX_RANGE + 1.0e-9


def add_shadow_readiness(
    case_id: str,
    lifecycle: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append local-only readiness and post-hoc evaluation fields."""
    output: list[dict[str, Any]] = []
    previous_valid: dict[str, float] | None = None
    consecutive = 0
    recent_valid: deque[bool] = deque(maxlen=READINESS_WINDOW)
    ready_previous = False
    for row in lifecycle:
        current_valid = _current_model_valid(row)
        consecutive = consecutive + 1 if current_valid else 0
        recent_valid.append(current_valid)
        if current_valid and previous_valid is not None:
            width_change = float(row["current_width"]) - previous_valid["width"]
            offset_change = float(row["current_offset"]) - previous_valid["offset"]
            orientation_change = (
                float(row["current_orientation"]) - previous_valid["orientation"]
            )
        else:
            width_change = offset_change = orientation_change = math.nan
        if current_valid:
            previous_valid = {
                "width": float(row["current_width"]),
                "offset": float(row["current_offset"]),
                "orientation": float(row["current_orientation"]),
            }

        # READY_SIMPLE: exact reuse of existing stable-model initialization.
        ready = bool(row["stable_model_initialized"])
        if ready:
            readiness_state = "MODEL_READY"
            readiness_reason = "EXISTING_STABLE_MODEL_INITIALIZED"
        elif current_valid:
            readiness_state = "MODEL_BOOTSTRAPPING"
            readiness_reason = (
                f"VALID_LOCAL_OBSERVATION_{consecutive}_OF_EXISTING_{READINESS_WINDOW}"
            )
        else:
            readiness_state = "MODEL_UNINITIALIZED"
            readiness_reason = "NO_VALID_CURRENT_OR_STABLE_MODEL"

        raw_detected = bool(row["junction_detected"])
        gated_detected = bool(ready and raw_detected)
        if not ready:
            semantic_state = readiness_state
        elif gated_detected:
            semantic_state = "JUNCTION_DETECTED"
        else:
            semantic_state = "JUNCTION_CLEAR"
        readiness_rising = bool(ready and not ready_previous)
        false_readiness = bool(
            readiness_rising
            and row["gt_side_opening_in_fit_sector_eval_only"] > 0
        )
        output.append(
            {
                "case_id": case_id,
                "frame": int(row["physics_frame"]),
                "time": float(row["time"]),
                "side_walls_valid": bool(row["side_walls_valid"]),
                "parallel_error_deg": row["parallel_error_deg"],
                "current_width": row["current_width"],
                "current_offset": row["current_offset"],
                "current_orientation": row["current_orientation"],
                "stable_width": row["stable_width"],
                "stable_offset": row["stable_offset"],
                "stable_orientation": row["stable_orientation"],
                "stable_update_count": int(row["stable_model_update_count"]),
                "current_model_valid": current_valid,
                "stable_model_initialized": bool(
                    row["stable_model_initialized"]
                ),
                "width_change_from_previous_valid": width_change,
                "orientation_change_from_previous_valid": orientation_change,
                "offset_change_from_previous_valid": offset_change,
                "consecutive_valid_count": consecutive,
                "valid_ratio_existing_init_window": sum(recent_valid)
                / len(recent_valid),
                "expected_profile_source": row["expected_profile_source"],
                "readiness_state": readiness_state,
                "readiness_reason": readiness_reason,
                "corridor_model_ready": ready,
                "semantic_detector_state": semantic_state,
                "open_candidate_count": int(row["open_candidate_count"]),
                "opening_group_count": int(row["opening_group_count"]),
                "junction_detected_raw": raw_detected,
                "junction_detected_gated_eval": gated_detected,
                "readiness_rising": readiness_rising,
                "false_readiness_eval_only": false_readiness,
                "opening_visible_eval_only": _opening_visible_eval_only(row),
            }
        )
        ready_previous = ready
    return output


def _first(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((row for row in rows if row[key]), None)


def _value(row: dict[str, Any] | None, key: str, default: Any = "") -> Any:
    return default if row is None else row[key]


def _post_ready_stability(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    ready = [row for row in rows if row["corridor_model_ready"]]
    if not ready:
        return math.nan, math.nan, math.nan
    reference = ready[0]

    def maximum_delta(key: str) -> float:
        values = [
            abs(float(row[key]) - float(reference[key]))
            for row in ready
            if _finite(row[key]) and _finite(reference[key])
        ]
        return max(values, default=math.nan)

    return (
        maximum_delta("stable_width"),
        maximum_delta("stable_orientation"),
        maximum_delta("stable_offset"),
    )


def case_summary(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _first(rows, "current_model_valid")
    stable = _first(rows, "stable_model_initialized")
    readiness = _first(rows, "corridor_model_ready")
    opening = _first(rows, "open_candidate_count")
    raw_detection = _first(rows, "junction_detected_raw")
    gated_detection = _first(rows, "junction_detected_gated_eval")
    width_delta, orientation_delta, offset_delta = _post_ready_stability(rows)
    return {
        "case_id": case_id,
        "first_valid_model_frame": _value(valid, "frame"),
        "stable_init_frame": _value(stable, "frame"),
        "readiness_frame": _value(readiness, "frame"),
        "readiness_time": _value(readiness, "time"),
        "first_open_frame": _value(opening, "frame"),
        "first_detection_frame": _value(raw_detection, "frame"),
        "gated_detection_frame": _value(gated_detection, "frame"),
        "false_readiness_eval": any(row["false_readiness_eval_only"] for row in rows),
        "false_junction_count": sum(
            row["junction_detected_gated_eval"]
            and not row["opening_visible_eval_only"]
            for row in rows
        ),
        "raw_detection_count": sum(row["junction_detected_raw"] for row in rows),
        "gated_detection_count": sum(
            row["junction_detected_gated_eval"] for row in rows
        ),
        "final_readiness_state": rows[-1]["readiness_state"],
        "max_stable_width_change_after_ready": width_delta,
        "max_stable_orientation_change_after_ready": orientation_delta,
        "max_stable_offset_change_after_ready": offset_delta,
    }


def event_timing(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _first(rows, "current_model_valid")
    stable = _first(rows, "stable_model_initialized")
    readiness = _first(rows, "corridor_model_ready")
    opening = _first(rows, "open_candidate_count")
    raw_detection = _first(rows, "junction_detected_raw")
    return {
        "case_id": case_id,
        "frame_first_current_model_valid": _value(valid, "frame"),
        "time_first_current_model_valid": _value(valid, "time"),
        "frame_stable_model_initialized": _value(stable, "frame"),
        "time_stable_model_initialized": _value(stable, "time"),
        "frame_readiness_true": _value(readiness, "frame"),
        "time_readiness_true": _value(readiness, "time"),
        "frame_first_open": _value(opening, "frame"),
        "time_first_open": _value(opening, "time"),
        "frame_first_junction_detection": _value(raw_detection, "frame"),
        "time_first_junction_detection": _value(raw_detection, "time"),
        "readiness_to_open_latency_frames": _frame_delta(readiness, opening),
        "readiness_to_detection_latency_frames": _frame_delta(
            readiness, raw_detection
        ),
    }


def _frame_delta(
    start: dict[str, Any] | None, end: dict[str, Any] | None
) -> int | str:
    if start is None or end is None:
        return ""
    return int(end["frame"]) - int(start["frame"])


def shadow_vs_gated(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _first(rows, "junction_detected_raw")
    gated = _first(rows, "junction_detected_gated_eval")
    return {
        "case_id": case_id,
        "raw_detector_detection_frame": _value(raw, "frame"),
        "readiness_gated_detection_frame": _value(gated, "frame"),
        "readiness_blocked_samples": sum(
            row["junction_detected_raw"] and not row["corridor_model_ready"]
            for row in rows
        ),
        "readiness_false_positive_count": sum(
            row["junction_detected_gated_eval"]
            and not row["opening_visible_eval_only"]
            for row in rows
        ),
        "readiness_false_negative_count_eval": sum(
            row["junction_detected_raw"]
            and not row["junction_detected_gated_eval"]
            and row["opening_visible_eval_only"]
            for row in rows
        ),
        "shadow_detector_output_changed": False,
    }


def _signature(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            row["frame"],
            row["current_model_valid"],
            row["stable_update_count"],
            row["readiness_state"],
            row["open_candidate_count"],
            row["junction_detected_raw"],
            row["junction_detected_gated_eval"],
        )
        for row in rows
    )


def verdict(
    summaries: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    replay_match: bool,
) -> dict[str, Any]:
    by_case = {row["case_id"]: row for row in summaries}
    timing = {row["case_id"]: row for row in timings}
    shadow = {row["case_id"]: row for row in shadow_rows}
    m0 = by_case[M0_CASE]
    baseline = by_case[BASELINE_CASE]
    bootstrap = by_case[BOOTSTRAP_CASE]
    exp042_equivalent = bool(
        timing[BOOTSTRAP_CASE]["frame_stable_model_initialized"] == 6
        and timing[BOOTSTRAP_CASE]["frame_first_open"] == 30
        and timing[BOOTSTRAP_CASE]["frame_first_junction_detection"] == 36
    )
    success = bool(
        m0["readiness_frame"] != ""
        and m0["false_junction_count"] == 0
        and baseline["readiness_frame"] == ""
        and bootstrap["readiness_frame"] == 6
        and bootstrap["false_readiness_eval"] is False
        and shadow[BOOTSTRAP_CASE]["readiness_gated_detection_frame"] == 36
        and exp042_equivalent
    )
    return {
        "verdict": (
            "D_STABLE_MODEL_SEMANTICS_SUFFICIENT"
            if success
            else "READINESS_VALIDATION_FAILED"
        ),
        "ready_rule": "corridor_model_initialized == True",
        "readiness_numeric_threshold_added": False,
        "existing_initialization_scan_count": READINESS_WINDOW,
        "m0_ready_and_clear": bool(
            m0["readiness_frame"] != "" and m0["false_junction_count"] == 0
        ),
        "baseline_uninitialized_before_raw_detection": bool(
            baseline["readiness_frame"] == ""
        ),
        "bootstrap_ready_before_open": bool(
            bootstrap["readiness_frame"] != ""
            and bootstrap["readiness_frame"] < bootstrap["first_open_frame"]
        ),
        "exp042_timing_equivalent": exp042_equivalent,
        "bootstrap_gated_equals_raw": bool(
            shadow[BOOTSTRAP_CASE]["raw_detector_detection_frame"]
            == shadow[BOOTSTRAP_CASE]["readiness_gated_detection_frame"]
        ),
        "baseline_gated_false_negative_samples_eval": shadow[BASELINE_CASE][
            "readiness_false_negative_count_eval"
        ],
        "deterministic_replay_match": replay_match,
        "detector_changed": False,
        "threshold_changed": False,
        "margin_changed": False,
        "grouping_changed": False,
        "simulator_changed": False,
        "gt_or_map_used_for_runtime_readiness": False,
    }


def _write(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _plot_readiness(
    path: Path,
    baseline: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    level = {
        "MODEL_UNINITIALIZED": 0.0,
        "MODEL_BOOTSTRAPPING": 1.0,
        "MODEL_READY": 2.0,
    }
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for axis, rows, title in zip(
        axes, (baseline, bootstrap), (BASELINE_CASE, BOOTSTRAP_CASE)
    ):
        frames = [row["frame"] for row in rows]
        axis.step(
            frames,
            [level[row["readiness_state"]] for row in rows],
            where="post",
            label="readiness 0/1/2",
        )
        axis.step(
            frames,
            [2.5 if row["open_candidate_count"] else 0.0 for row in rows],
            where="post",
            label="OPEN",
        )
        axis.step(
            frames,
            [3.0 if row["junction_detected_raw"] else 0.0 for row in rows],
            where="post",
            label="raw Junction",
        )
        axis.set(title=title, ylabel="semantic state")
        axis.grid(alpha=0.22)
        axis.legend(loc="best")
    axes[-1].set_xlabel("physics frame")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_quality(path: Path, cases: list[list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    for axis, rows in zip(axes, cases):
        frames = [row["frame"] for row in rows]
        parallel = [
            min(60.0, float(row["parallel_error_deg"]))
            if _finite(row["parallel_error_deg"])
            else math.nan
            for row in rows
        ]
        axis.plot(frames, parallel, label="parallel error deg (cap 60)")
        axis.plot(
            frames,
            [float(row["stable_width"]) if _finite(row["stable_width"]) else math.nan for row in rows],
            label="stable width",
        )
        axis.step(
            frames,
            [84.0 if row["corridor_model_ready"] else 0.0 for row in rows],
            where="post",
            label="READY x84",
        )
        axis.set(title=rows[0]["case_id"], ylabel="quality / width")
        axis.grid(alpha=0.22)
        axis.legend(loc="best", ncol=3)
    axes[-1].set_xlabel("physics frame")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-frames", type=int, default=127)
    parser.add_argument("--bootstrap-frames", type=int, default=127)
    parser.add_argument("--m0-frames", type=int, default=600)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args(argv)

    _, baseline_raw = run_case(
        BASELINE_CASE, "M1_CROSS_BASELINE", args.baseline_frames
    )
    _, bootstrap_raw = run_case(
        BOOTSTRAP_CASE,
        "M1_CROSS_BASELINE",
        args.bootstrap_frames,
        REAR_START_SHIFT,
    )
    _, m0_raw = run_case(M0_CASE, "M0_STRAIGHT", args.m0_frames)
    baseline = add_shadow_readiness(BASELINE_CASE, baseline_raw)
    bootstrap = add_shadow_readiness(BOOTSTRAP_CASE, bootstrap_raw)
    m0 = add_shadow_readiness(M0_CASE, m0_raw)
    if args.skip_replay:
        replay_match = False
    else:
        _, replay_raw = run_case(
            BOOTSTRAP_CASE,
            "M1_CROSS_BASELINE",
            args.bootstrap_frames,
            REAR_START_SHIFT,
        )
        replay = add_shadow_readiness(BOOTSTRAP_CASE, replay_raw)
        replay_match = _signature(bootstrap) == _signature(replay)

    cases = [baseline, bootstrap, m0]
    summaries = [case_summary(rows[0]["case_id"], rows) for rows in cases]
    timings = [event_timing(rows[0]["case_id"], rows) for rows in cases]
    shadow_rows = [shadow_vs_gated(rows[0]["case_id"], rows) for rows in cases]
    final_verdict = verdict(summaries, timings, shadow_rows, replay_match)
    _write(
        args.output / "readiness_feature_timeline.csv",
        [row for rows in cases for row in rows],
        FEATURE_FIELDS,
    )
    _write(args.output / "readiness_case_summary.csv", summaries)
    _write(args.output / "readiness_event_timing.csv", timings)
    _write(args.output / "shadow_vs_gated.csv", shadow_rows)
    _write(args.output / "verdict.csv", [final_verdict])
    _plot_readiness(args.output / "readiness_timeline.png", baseline, bootstrap)
    _plot_quality(args.output / "model_quality_vs_readiness.png", cases)
    print(
        f"verdict={final_verdict['verdict']} output={args.output.resolve()} "
        f"m0_ready={summaries[2]['readiness_frame']} "
        f"baseline_ready={summaries[0]['readiness_frame']} "
        f"bootstrap_ready={summaries[1]['readiness_frame']} replay={replay_match}"
    )


if __name__ == "__main__":
    main()
