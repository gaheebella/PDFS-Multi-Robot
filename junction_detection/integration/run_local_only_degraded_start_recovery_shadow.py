"""EXP-044 local-only degraded-start recovery shadow validation.

This diagnostic consumes frozen detector lifecycle outputs.  It never gates or
changes a raw Junction decision.  The shadow degraded-start advisory reuses the
detector's existing initialization evidence count and only local model history;
GT/map fields are appended strictly for post-hoc evaluation.
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

from junction_detection.integration.run_lidar_local_corridor_estimation import (  # noqa: E402
    REAR_START_SHIFT,
)
from junction_detection.integration.run_local_corridor_model_readiness import (  # noqa: E402
    add_shadow_readiness,
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
    LIDAR_MAX_RANGE,
)


EXPERIMENT_ID = "EXP-044"
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/local_only_degraded_start_recovery_shadow"
)
# Existing detector evidence budget; this is not a new recovery threshold.
INITIALIZATION_EVIDENCE_COUNT = GeometryProfileConfig(
    LIDAR_MAX_RANGE
).initialization_scan_count

TIMELINE_FIELDS = (
    "case_id",
    "frame",
    "time",
    "initialization_opportunity_count",
    "valid_current_model_history_count",
    "current_model_valid",
    "stable_model_initialized",
    "readiness_state",
    "expected_profile_source",
    "side_walls_valid",
    "parallel_error_deg",
    "current_width",
    "current_offset",
    "current_orientation",
    "stable_width",
    "stable_offset",
    "stable_orientation",
    "stable_update_count",
    "initialization_budget_exhausted",
    "zero_valid_model_history",
    "local_invalid_geometry_evidence",
    "open_like_evidence",
    "degraded_local_support",
    "degraded_start_trigger",
    "degraded_start_suspected",
    "degraded_start_reason",
    "recovery_shadow_state",
    "open_candidate_count",
    "opening_group_count",
    "junction_detected_raw",
    "junction_detected_published_shadow",
    "raw_detection_preserved",
    "opening_visible_eval_only",
    "degraded_false_positive_eval_only",
)


def add_degraded_start_shadow(
    case_id: str,
    readiness_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add a latched advisory while preserving every raw detector decision."""
    output: list[dict[str, Any]] = []
    valid_history_count = 0
    degraded_latched = False
    for opportunity, row in enumerate(readiness_rows, start=1):
        if row["current_model_valid"]:
            valid_history_count += 1

        ready = bool(row["stable_model_initialized"])
        budget_exhausted = opportunity >= INITIALIZATION_EVIDENCE_COUNT
        zero_valid_history = valid_history_count == 0
        local_invalid_geometry = not bool(row["side_walls_valid"])
        open_like_evidence = int(row["open_candidate_count"]) > 0
        degraded_local_support = bool(
            local_invalid_geometry or open_like_evidence
        )
        trigger = bool(
            not ready
            and budget_exhausted
            and zero_valid_history
            and degraded_local_support
        )
        degraded_latched = bool(degraded_latched or trigger)

        if degraded_latched and not ready:
            recovery_state = "DEGRADED_START_SUSPECTED"
            reason = (
                "EXISTING_INITIALIZATION_BUDGET_EXHAUSTED_"
                "WITH_ZERO_VALID_LOCAL_MODELS_AND_LOCAL_GEOMETRY_SUPPORT"
            )
        elif ready:
            recovery_state = "MODEL_READY"
            reason = "EXISTING_STABLE_MODEL_INITIALIZED"
        else:
            recovery_state = row["readiness_state"]
            reason = "COLLECTING_EXISTING_INITIALIZATION_EVIDENCE"

        raw_detected = bool(row["junction_detected_raw"])
        # Transparent shadow adapter: degraded readiness is advisory only.
        published_detected = raw_detected
        opening_visible = bool(row["opening_visible_eval_only"])
        output.append(
            {
                "case_id": case_id,
                "frame": int(row["frame"]),
                "time": float(row["time"]),
                "initialization_opportunity_count": opportunity,
                "valid_current_model_history_count": valid_history_count,
                "current_model_valid": bool(row["current_model_valid"]),
                "stable_model_initialized": ready,
                "readiness_state": row["readiness_state"],
                "expected_profile_source": row["expected_profile_source"],
                "side_walls_valid": bool(row["side_walls_valid"]),
                "parallel_error_deg": row["parallel_error_deg"],
                "current_width": row["current_width"],
                "current_offset": row["current_offset"],
                "current_orientation": row["current_orientation"],
                "stable_width": row["stable_width"],
                "stable_offset": row["stable_offset"],
                "stable_orientation": row["stable_orientation"],
                "stable_update_count": int(row["stable_update_count"]),
                "initialization_budget_exhausted": budget_exhausted,
                "zero_valid_model_history": zero_valid_history,
                "local_invalid_geometry_evidence": local_invalid_geometry,
                "open_like_evidence": open_like_evidence,
                "degraded_local_support": degraded_local_support,
                "degraded_start_trigger": trigger,
                "degraded_start_suspected": degraded_latched,
                "degraded_start_reason": reason,
                "recovery_shadow_state": recovery_state,
                "open_candidate_count": int(row["open_candidate_count"]),
                "opening_group_count": int(row["opening_group_count"]),
                "junction_detected_raw": raw_detected,
                "junction_detected_published_shadow": published_detected,
                "raw_detection_preserved": published_detected == raw_detected,
                "opening_visible_eval_only": opening_visible,
                "degraded_false_positive_eval_only": bool(
                    trigger and not opening_visible
                ),
            }
        )
    return output


def _first(
    rows: list[dict[str, Any]], key: str
) -> dict[str, Any] | None:
    return next((row for row in rows if row[key]), None)


def _value(row: dict[str, Any] | None, key: str, default: Any = "") -> Any:
    return default if row is None else row[key]


def case_summary(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = _first(rows, "stable_model_initialized")
    degraded = _first(rows, "degraded_start_trigger")
    valid = _first(rows, "current_model_valid")
    opening = _first(rows, "open_candidate_count")
    raw = _first(rows, "junction_detected_raw")
    published = _first(rows, "junction_detected_published_shadow")
    return {
        "case_id": case_id,
        "first_valid_current_model_frame": _value(valid, "frame"),
        "model_ready_frame": _value(ready, "frame"),
        "degraded_start_suspected_frame": _value(degraded, "frame"),
        "degraded_start_suspected_time": _value(degraded, "time"),
        "first_open_frame": _value(opening, "frame"),
        "raw_detection_frame": _value(raw, "frame"),
        "published_detection_frame": _value(published, "frame"),
        "raw_detection_preserved_all_samples": all(
            row["raw_detection_preserved"] for row in rows
        ),
        "degraded_false_positive_count_eval": sum(
            row["degraded_false_positive_eval_only"] for row in rows
        ),
        "final_recovery_shadow_state": rows[-1]["recovery_shadow_state"],
        "final_readiness_state": rows[-1]["readiness_state"],
    }


def publication_comparison(
    case_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = _first(rows, "junction_detected_raw")
    published = _first(rows, "junction_detected_published_shadow")
    return {
        "case_id": case_id,
        "raw_detection_frame": _value(raw, "frame"),
        "shadow_published_detection_frame": _value(published, "frame"),
        "raw_positive_samples": sum(row["junction_detected_raw"] for row in rows),
        "published_positive_samples": sum(
            row["junction_detected_published_shadow"] for row in rows
        ),
        "blocked_raw_detection_samples": sum(
            row["junction_detected_raw"]
            and not row["junction_detected_published_shadow"]
            for row in rows
        ),
        "altered_detector_output_samples": sum(
            not row["raw_detection_preserved"] for row in rows
        ),
    }


def _signature(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            row["frame"],
            row["current_model_valid"],
            row["stable_model_initialized"],
            row["degraded_start_suspected"],
            row["open_candidate_count"],
            row["opening_group_count"],
            row["junction_detected_raw"],
            row["junction_detected_published_shadow"],
        )
        for row in rows
    )


def build_verdict(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    baseline_replay_match: bool,
) -> dict[str, Any]:
    summary = {row["case_id"]: row for row in summaries}
    comparison = {row["case_id"]: row for row in comparisons}
    baseline = summary[BASELINE_CASE]
    bootstrap = summary[BOOTSTRAP_CASE]
    m0 = summary[M0_CASE]
    feasible = bool(
        baseline["degraded_start_suspected_frame"] == 6
        and baseline["raw_detection_frame"] == 126
        and baseline["published_detection_frame"] == 126
        and bootstrap["degraded_start_suspected_frame"] == ""
        and bootstrap["model_ready_frame"] == 6
        and bootstrap["raw_detection_frame"] == 36
        and m0["degraded_start_suspected_frame"] == ""
        and m0["model_ready_frame"] == 6
        and comparison[BASELINE_CASE]["blocked_raw_detection_samples"] == 0
        and all(
            row["altered_detector_output_samples"] == 0
            for row in comparisons
        )
        and baseline_replay_match
    )
    return {
        "verdict": (
            "A_LOCAL_ONLY_DEGRADED_START_SHADOW_FEASIBLE"
            if feasible
            else "DEGRADED_START_SHADOW_VALIDATION_FAILED"
        ),
        "shadow_rule": (
            "not stable_model_initialized AND initialization opportunities >= "
            "existing initialization_scan_count AND zero valid current models "
            "AND (invalid local wall geometry OR existing OPEN candidate)"
        ),
        "existing_initialization_scan_count": INITIALIZATION_EVIDENCE_COUNT,
        "new_numeric_threshold_added": False,
        "baseline_degraded_at_frame_6": (
            baseline["degraded_start_suspected_frame"] == 6
        ),
        "baseline_raw_detection_preserved_frame_126": bool(
            baseline["raw_detection_frame"] == 126
            and baseline["published_detection_frame"] == 126
        ),
        "bootstrap_not_degraded": (
            bootstrap["degraded_start_suspected_frame"] == ""
        ),
        "m0_not_degraded": m0["degraded_start_suspected_frame"] == "",
        "all_raw_outputs_preserved": all(
            row["altered_detector_output_samples"] == 0
            for row in comparisons
        ),
        "baseline_deterministic_replay_match": baseline_replay_match,
        "runtime_gt_or_map_used": False,
        "production_gating_applied": False,
        "detector_changed": False,
        "threshold_changed": False,
        "margin_changed": False,
        "grouping_changed": False,
        "simulator_changed": False,
        "robot_control_changed": False,
    }


def _write(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, cases: list[list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    readiness_level = {
        "MODEL_UNINITIALIZED": 0.0,
        "MODEL_BOOTSTRAPPING": 1.0,
        "MODEL_READY": 2.0,
    }
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    for axis, rows in zip(axes, cases):
        frames = [row["frame"] for row in rows]
        axis.step(
            frames,
            [readiness_level[row["readiness_state"]] for row in rows],
            where="post",
            label="readiness 0/1/2",
        )
        axis.step(
            frames,
            [2.5 if row["degraded_start_suspected"] else 0.0 for row in rows],
            where="post",
            label="DEGRADED suspected",
        )
        axis.step(
            frames,
            [3.0 if row["junction_detected_raw"] else 0.0 for row in rows],
            where="post",
            label="raw = published Junction",
        )
        axis.set(title=rows[0]["case_id"], ylabel="shadow state")
        axis.grid(alpha=0.22)
        axis.legend(loc="best")
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

    baseline = add_degraded_start_shadow(
        BASELINE_CASE, add_shadow_readiness(BASELINE_CASE, baseline_raw)
    )
    bootstrap = add_degraded_start_shadow(
        BOOTSTRAP_CASE, add_shadow_readiness(BOOTSTRAP_CASE, bootstrap_raw)
    )
    m0 = add_degraded_start_shadow(
        M0_CASE, add_shadow_readiness(M0_CASE, m0_raw)
    )

    if args.skip_replay:
        replay_match = False
    else:
        _, replay_raw = run_case(
            BASELINE_CASE, "M1_CROSS_BASELINE", args.baseline_frames
        )
        replay = add_degraded_start_shadow(
            BASELINE_CASE,
            add_shadow_readiness(BASELINE_CASE, replay_raw),
        )
        replay_match = _signature(baseline) == _signature(replay)

    cases = [baseline, bootstrap, m0]
    summaries = [case_summary(rows[0]["case_id"], rows) for rows in cases]
    comparisons = [
        publication_comparison(rows[0]["case_id"], rows) for rows in cases
    ]
    final_verdict = build_verdict(summaries, comparisons, replay_match)

    _write(
        args.output / "degraded_start_feature_timeline.csv",
        [row for rows in cases for row in rows],
        TIMELINE_FIELDS,
    )
    _write(args.output / "degraded_start_case_summary.csv", summaries)
    _write(args.output / "raw_vs_shadow_publication.csv", comparisons)
    _write(args.output / "verdict.csv", [final_verdict])
    _plot(args.output / "degraded_start_timeline.png", cases)
    print(
        f"verdict={final_verdict['verdict']} output={args.output.resolve()} "
        f"baseline_degraded={summaries[0]['degraded_start_suspected_frame']} "
        f"baseline_raw={summaries[0]['raw_detection_frame']} "
        f"baseline_published={summaries[0]['published_detection_frame']} "
        f"replay={replay_match}"
    )


if __name__ == "__main__":
    main()
