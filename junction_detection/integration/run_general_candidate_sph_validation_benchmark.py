"""EXP-039 general branch candidate plus frozen SPH validation benchmark.

This integration adapter replaces only the accepted-gap candidate constructor:
the frozen EXP-038 orientation-relative representation feeds the unchanged
EXP-034/035 topology and motion state machinery.  Runtime association uses
only candidate-local directions and the already existing 0.5 directional
association gate.  Map identity and branch labels are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_general_orientation_relative_branch_candidate import (
    DEFAULT_SOURCE,
    SAFE_SOURCE,
    _evaluate_case as exp038_evaluate_case,
    _exp035_representative_input,
)
from junction_detection.pointcloud.general_branch_candidate import (
    OrientationRelativeGapDescriptor,
    build_general_branch_candidate,
    self_test as general_candidate_self_test,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import CASES, GeometryBuilder
from pygame_simulator.pre_exploration_persistent_partial_sph_validation import (
    PersistentPartialAuditRunner,
    _candidate_state_self_test,
    _candidate_summaries,
)
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (
    _axis_frame,
    _endpoint_free_continuation,
    _rotation,
)

EXPERIMENT_ID = "EXP-039"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/general_candidate_sph_validation_benchmark"
BENCHMARK_CASES = tuple(CASES)
EXP036_GEOMETRY = {
    "M0_STRAIGHT": (0, 0),
    "M1_CROSS_BASELINE": (3, 2),
    "M2_T_JUNCTION": (2, 2),
    "M3_ANGLED_Y": (2, 2),
    "M4_ASYMMETRIC_CROSS": (3, 2),
    "M5_UNEQUAL_WIDTH": (3, 2),
}
BRANCH_FIELDS = [
    "case", "candidate_id", "matched_branch_eval_only", "is_true_candidate_eval_only",
    "relative_opening_angle_deg", "gap_width_over_W_hat", "free_continuation",
    "incident_wall_alignment_deg", "t_candidate", "t_partial", "t_complete",
    "t_first_motion_evidence", "t_motion_supported", "validation_pathway", "final_state",
    "candidate_created", "complete", "motion_supported", "false_motion_supported_eval_only",
    "topology_type", "termination_count", "observed_robot_count", "directional_sample_count",
    "supporting_robot_count", "motion_reliability", "motion_tangent_deg",
]


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write a deterministic CSV, including headers for an empty audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"fields required for empty CSV: {path}")
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _branch_label(index: int, angle: float) -> str:
    """Name a GT branch strictly in post-hoc evaluation output."""
    return {0.0: "FORWARD", -90.0: "RIGHT", 90.0: "LEFT"}.get(float(angle), f"BRANCH_{index}")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _canonical(value: Any) -> Any:
    """Normalize non-finite floats for exact deterministic replay."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else str(value)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _descriptor_record(descriptor: OrientationRelativeGapDescriptor) -> dict[str, Any]:
    """Retain invariant EXP-038 observables without directional classification."""
    return {
        "relative_opening_angle_deg": descriptor.relative_opening_angle_deg,
        "gap_width_over_W_hat": descriptor.gap_width_over_W_hat,
        "free_continuation": descriptor.free_continuation,
        "incident_wall_alignment_deg": descriptor.gap_boundary_wall_alignment_error_deg,
        "candidate_reliability": descriptor.normal_free_alignment,
    }


class GeneralCandidateAuditRunner(PersistentPartialAuditRunner):
    """Feed EXP-038 candidates into the frozen topology/motion state tracker."""

    def _normal_in_candidate_frame(
        self,
        descriptor: OrientationRelativeGapDescriptor,
        candidate: Any,
    ) -> np.ndarray:
        """Express a current local opening direction in a persistent candidate frame."""
        world_normal = _rotation(self.leader().body_yaw_rad) @ descriptor.opening_normal_local
        return _rotation(candidate.anchor_yaw_rad).T @ world_normal

    def _existing_general_candidate(
        self,
        descriptor: OrientationRelativeGapDescriptor,
    ) -> Any | None:
        """Reuse the frozen 0.5 local-direction association gate without labels."""
        return next(
            (
                candidate
                for candidate in self.candidates
                if float(np.dot(
                    candidate.free_axis_local,
                    self._normal_in_candidate_frame(descriptor, candidate),
                )) > 0.5
            ),
            None,
        )

    def _candidate_frame_point(self, point_local_now: np.ndarray, candidate: Any) -> np.ndarray:
        """Transform a current scan-local point into a candidate's creation frame."""
        world = self.leader().position + _rotation(self.leader().body_yaw_rad) @ point_local_now
        return candidate.world_to_local(world)

    def _attach_descriptor(
        self,
        candidate: Any,
        descriptor: OrientationRelativeGapDescriptor,
    ) -> None:
        """Attach immutable general geometry as logging/evaluation metadata."""
        candidate.general_descriptor_record = _descriptor_record(descriptor)
        candidate.general_gap_center_local = self._candidate_frame_point(descriptor.gap_center_local, candidate)
        candidate.general_endpoint_a_local = self._candidate_frame_point(descriptor.endpoint_a_local, candidate)
        candidate.general_endpoint_b_local = self._candidate_frame_point(descriptor.endpoint_b_local, candidate)

    def _make_candidates(
        self,
        result: dict[str, Any],
        snapshot: dict[str, Any],
        orientation: float,
        width: float,
    ) -> None:
        """Replace only accepted-gap construction; preserve frozen PARTIAL handling."""
        corridor_axis, left_axis = _axis_frame(orientation)
        endpoints = {int(row["endpoint_id"]): row for row in result["endpoints"]}
        segments = {int(row["segment_id"]): row for row in result["segments"]}
        used: set[int] = set()

        for gap in result["gaps"]:
            if not gap["candidate_valid"]:
                continue
            descriptor, general = build_general_branch_candidate(
                candidate_id=f"G{len(self.candidates)}",
                timestamp=self.world.time,
                topology_type="COMPLETE",
                gap=gap,
                endpoints=endpoints,
                segments=segments,
                corridor_axis_local=corridor_axis,
            )
            if general is None:
                continue
            first = endpoints[int(gap["endpoint_a"])]
            second = endpoints[int(gap["endpoint_b"])]
            existing = self._existing_general_candidate(descriptor)
            if existing is None:
                existing = self._new_candidate(
                    "COMPLETE",
                    first,
                    result,
                    snapshot,
                    width,
                    descriptor.opening_normal_local,
                    (int(first["endpoint_id"]), int(second["endpoint_id"])),
                    gap,
                )
                # A GENERAL candidate is centered on the physical gap.  The
                # frozen SPH association/state logic is otherwise untouched.
                existing.endpoint_local = descriptor.gap_center_local.copy()
                existing.wall_tangent_local = descriptor.gap_tangent_local.copy()
            elif existing.topology_type == "PARTIAL":
                before = existing.state
                existing.topology_type = "COMPLETE"
                existing.endpoint_ids = (int(first["endpoint_id"]), int(second["endpoint_id"]))
                existing.gap_center_local = self._candidate_frame_point(descriptor.gap_center_local, existing)
                existing.gap_width = descriptor.gap_width
                if before == "PARTIAL_UNVALIDATED":
                    existing.state = "COMPLETE_UNVALIDATED"
                self._event("COMPLETE_BRANCH_CANDIDATE", existing, before, existing.state)
            self._attach_descriptor(existing, descriptor)
            used.update((int(first["endpoint_id"]), int(second["endpoint_id"])))

        # The EXP-034 one-ended PARTIAL definition and constructor stay exact.
        for endpoint in result["endpoints"]:
            endpoint_id = int(endpoint["endpoint_id"])
            if endpoint_id in used or endpoint["endpoint_type"] != "WALL_TERMINATION":
                continue
            lateral = float(np.dot(endpoint["point"], left_axis))
            if abs(lateral) <= np.finfo(float).eps:
                continue
            free_axis = left_axis * math.copysign(1.0, lateral)
            if self._existing_side_candidate(free_axis) is None:
                self._new_candidate(
                    "PARTIAL", endpoint, result, snapshot, width, free_axis,
                    (endpoint_id,), None,
                )


def _evaluate_candidate(runner: GeneralCandidateAuditRunner, candidate: Any) -> str:
    """Apply the corrected EXP-036 rectangle-edge matcher after runtime ends."""
    # Preserve EXP-036 exactly for an existing PARTIAL promoted to COMPLETE:
    # its observed endpoint is the frozen evaluation reference.  A direct
    # GENERAL candidate has no earlier endpoint state, so its gap center is its
    # representation reference.  Neither value is consumed by runtime state.
    partial_observed = any(
        row["candidate_id"] == candidate.candidate_id
        and row["event"] == "PARTIAL_BRANCH_CANDIDATE"
        for row in runner.events
    )
    reference_local = (
        candidate.endpoint_local
        if partial_observed
        else getattr(candidate, "general_gap_center_local", candidate.endpoint_local)
    )
    center_world = candidate.anchor_position_eval + _rotation(candidate.anchor_yaw_rad) @ reference_local
    matches: list[tuple[float, float, str]] = []
    for index, branch in enumerate(runner.geometry.branches):
        direction_world = np.array([
            math.sin(math.radians(branch.angle_deg)),
            math.cos(math.radians(branch.angle_deg)),
        ])
        direction_local = _rotation(candidate.anchor_yaw_rad).T @ direction_world
        alignment = float(np.dot(candidate.free_axis_local, direction_local))
        vertices = np.asarray(runner.geometry.free_rects[2 + index].vertices)
        distances = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = end - start
            ratio = float(np.clip(np.dot(center_world - start, edge) / np.dot(edge, edge), 0.0, 1.0))
            distances.append(float(np.linalg.norm(center_world - (start + ratio * edge))))
        edge_distance = min(distances)
        if alignment > 0.0 and edge_distance <= 0.12 * candidate.estimated_width:
            matches.append((edge_distance, -alignment, _branch_label(index, branch.angle_deg)))
    return min(matches)[2] if matches else "FALSE"


def _event_time(runner: GeneralCandidateAuditRunner, candidate_id: str, event: str) -> float:
    return next(
        (float(row["timestamp"]) for row in runner.events if row["candidate_id"] == candidate_id and row["event"] == event),
        math.nan,
    )


def _pathway(t_complete: float, t_motion: float) -> str:
    """Classify topology/motion ordering using the requested EXP-039 states."""
    complete, motion = _finite(t_complete), _finite(t_motion)
    if complete and motion:
        return "COMPLETE_BEFORE_MOTION" if t_complete <= t_motion else "MOTION_BEFORE_COMPLETE"
    if complete:
        return "COMPLETE_ONLY"
    if motion:
        return "MOTION_ONLY"
    return "UNRESOLVED"


def _run_worker(spec: tuple[str, int, bool, str]) -> dict[str, Any]:
    """Run one deterministic physical case with the integration adapter."""
    case_id, frames, rear_start, run_role = spec
    runner = GeneralCandidateAuditRunner(case_id, rear_start)
    for frame in range(frames):
        runner.step(frame)
    frozen_summaries = {row["candidate_id"]: row for row in _candidate_summaries(runner, runner.world.time)}
    details = []
    for candidate in runner.candidates:
        summary = frozen_summaries[candidate.candidate_id]
        label = _evaluate_candidate(runner, candidate)
        partial = _event_time(runner, candidate.candidate_id, "PARTIAL_BRANCH_CANDIDATE")
        complete = _event_time(runner, candidate.candidate_id, "COMPLETE_BRANCH_CANDIDATE")
        motion = _event_time(runner, candidate.candidate_id, "MOTION_SUPPORTED")
        descriptor = getattr(candidate, "general_descriptor_record", {})
        evidence = candidate.best_evidence or candidate.last_evidence
        details.append({
            "candidate_id": candidate.candidate_id,
            "matched_branch_eval_only": label,
            "topology_type": candidate.topology_type,
            "t_candidate": candidate.created_time,
            "t_partial": partial,
            "t_complete": complete,
            "t_first_motion_evidence": runner.first_motion_evidence.get(candidate.candidate_id, math.nan),
            "t_motion_supported": motion,
            "validation_pathway": _pathway(complete, motion),
            "final_state": candidate.state,
            "termination_count": len(candidate.endpoint_ids),
            "observed_robot_count": int(evidence.get("observed_robot_count", 0)),
            "directional_sample_count": int(candidate.directional_samples),
            "supporting_robot_count": int(evidence.get("supporting_robot_count", 0)),
            "motion_reliability": float(evidence.get("motion_reliability", 0.0)),
            "motion_tangent_deg": float(evidence.get("motion_direction_local", math.nan)),
            **{
                "relative_opening_angle_deg": descriptor.get("relative_opening_angle_deg", math.nan),
                "gap_width_over_W_hat": descriptor.get("gap_width_over_W_hat", math.nan),
                "free_continuation": descriptor.get("free_continuation", candidate.free_space_evidence),
                "incident_wall_alignment_deg": descriptor.get("incident_wall_alignment_deg", math.nan),
            },
            "frozen_summary": summary,
        })
    signature = {
        "events": runner.events,
        "candidates": [
            {key: row[key] for key in (
                "candidate_id", "topology_type", "t_candidate", "t_partial", "t_complete",
                "t_first_motion_evidence", "t_motion_supported", "final_state",
                "relative_opening_angle_deg", "gap_width_over_W_hat", "free_continuation",
                "incident_wall_alignment_deg", "directional_sample_count",
            )}
            for row in details
        ],
    }
    return {
        "case": case_id,
        "run_role": run_role,
        "frames": frames,
        "world_time": runner.world.time,
        "candidate_count": len(details),
        "details": details,
        "signature": signature,
    }


def _branch_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one evaluation row per GT branch plus each false candidate."""
    geometry = GeometryBuilder.build(result["case"])
    labels = [_branch_label(index, branch.angle_deg) for index, branch in enumerate(geometry.branches)]
    by_label = {label: [] for label in labels}
    false = []
    for candidate in result["details"]:
        if candidate["matched_branch_eval_only"] in by_label:
            by_label[candidate["matched_branch_eval_only"]].append(candidate)
        else:
            false.append(candidate)
    rows = []
    for label in labels:
        candidate = by_label[label][0] if by_label[label] else None
        row = {
            "case": result["case"],
            "candidate_id": "NONE" if candidate is None else candidate["candidate_id"],
            "matched_branch_eval_only": label,
            "is_true_candidate_eval_only": candidate is not None,
            "candidate_created": candidate is not None,
            "complete": candidate is not None and _finite(candidate["t_complete"]),
            "motion_supported": candidate is not None and _finite(candidate["t_motion_supported"]),
            "false_motion_supported_eval_only": False,
        }
        for field in BRANCH_FIELDS:
            if field not in row:
                row[field] = math.nan if candidate is None else candidate.get(field, math.nan)
        if candidate is None:
            row.update({"validation_pathway": "MISSED", "final_state": "MISSED", "topology_type": "MISSED", "termination_count": 0, "observed_robot_count": 0, "directional_sample_count": 0, "supporting_robot_count": 0, "motion_reliability": 0.0})
        rows.append(row)
        # Multiple runtime hypotheses matched to one physical branch count as
        # one true candidate plus false duplicate candidates in evaluation.
        false.extend(by_label[label][1:])
    for candidate in false:
        row = {field: candidate.get(field, math.nan) for field in BRANCH_FIELDS}
        row.update({
            "case": result["case"],
            "matched_branch_eval_only": "FALSE",
            "is_true_candidate_eval_only": False,
            "candidate_created": True,
            "complete": _finite(candidate["t_complete"]),
            "motion_supported": _finite(candidate["t_motion_supported"]),
            "false_motion_supported_eval_only": _finite(candidate["t_motion_supported"]),
        })
        rows.append(row)
    return rows


def _case_summary(case_id: str, rows: list[dict[str, Any]], replay_equal: bool) -> dict[str, Any]:
    true = [row for row in rows if row["matched_branch_eval_only"] != "FALSE"]
    false = [row for row in rows if row["matched_branch_eval_only"] == "FALSE"]
    true_candidates = [row for row in true if row["candidate_created"]]
    complete = [row for row in true_candidates if row["complete"]]
    motion = [row for row in true_candidates if row["motion_supported"]]
    gt = len(true)
    return {
        "case": case_id,
        "gt_branch_count": gt,
        "general_candidate_count": len(true_candidates) + len(false),
        "true_candidate_count": len(true_candidates),
        "false_candidate_count": len(false),
        "candidate_recall": len(true_candidates) / max(1, gt),
        "complete_count": len(complete),
        "complete_recall": len(complete) / max(1, gt),
        "motion_supported_count": len(motion),
        "motion_supported_recall": len(motion) / max(1, gt),
        "false_motion_supported_count": sum(bool(row["motion_supported"]) for row in false),
        "main_replay_exact": replay_equal,
    }


def _smoke_false_diagonal() -> dict[str, Any]:
    """Replay only EXP-035's persisted representative false diagonal."""
    data = _exp035_representative_input(DEFAULT_SOURCE, SAFE_SOURCE)
    rows, _, _ = exp038_evaluate_case(data)
    false_rows = [row for row in rows if row["is_false_gap_eval_only"]]
    return {
        "source_case": data["case"],
        "false_gap_count": len(false_rows),
        "false_general_candidate_count": sum(bool(row["new_candidate_created"]) for row in false_rows),
        "passed": bool(false_rows) and not any(bool(row["new_candidate_created"]) for row in false_rows),
    }


def _run_specs(specs: list[tuple[str, int, bool, str]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_run_worker(spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_worker, specs))


def _plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Save one compact recall comparison without rerunning physics."""
    cases = [row["case"] for row in summaries]
    x = np.arange(len(cases))
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar(x - 0.2, [EXP036_GEOMETRY[case][1] / max(1, EXP036_GEOMETRY[case][0]) for case in cases], 0.4, label="EXP-036")
    axis.bar(x + 0.2, [row["motion_supported_recall"] for row in summaries], 0.4, label="EXP-039")
    axis.set_xticks(x, cases, rotation=25, ha="right")
    axis.set(ylim=(0, 1.08), ylabel="motion-supported recall", title="EXP-036 vs EXP-039 frozen SPH validation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(output: Path, frames: int, m0_frames: int, smoke_m1_frames: int, workers: int) -> dict[str, Any]:
    """Execute gated smoke, main benchmark, and exact deterministic replay."""
    output.mkdir(parents=True, exist_ok=True)
    general_candidate_self_test()
    _candidate_state_self_test()
    diagonal = _smoke_false_diagonal()
    smoke_results = _run_specs([
        ("M0_STRAIGHT", m0_frames, False, "SMOKE"),
        ("M1_CROSS_BASELINE", smoke_m1_frames, True, "SMOKE"),
    ], 1)
    smoke = {row["case"]: row for row in smoke_results}
    smoke_m0_clean = smoke["M0_STRAIGHT"]["candidate_count"] == 0
    smoke_m1_labels = {row["matched_branch_eval_only"] for row in smoke["M1_CROSS_BASELINE"]["details"]}
    smoke_axial = next((row for row in smoke["M1_CROSS_BASELINE"]["details"] if row["matched_branch_eval_only"] == "FORWARD"), None)
    smoke_pass = bool(
        diagonal["passed"]
        and smoke_m0_clean
        and {"LEFT", "RIGHT", "FORWARD"}.issubset(smoke_m1_labels)
        and smoke_axial is not None
        and _finite(smoke_axial["t_complete"])
        and smoke_axial["observed_robot_count"] > 0
    )
    smoke_rows = [{
        "general_candidate_self_test": True,
        "candidate_state_self_test": True,
        "exp035_false_diagonal_pass": diagonal["passed"],
        "m0_candidate_count": smoke["M0_STRAIGHT"]["candidate_count"],
        "m1_candidate_count": smoke["M1_CROSS_BASELINE"]["candidate_count"],
        "m1_side_candidates": sum(label in smoke_m1_labels for label in ("LEFT", "RIGHT")),
        "m1_axial_candidate": smoke_axial is not None,
        "m1_axial_complete": smoke_axial is not None and _finite(smoke_axial["t_complete"]),
        "m1_axial_observed_robot_count": 0 if smoke_axial is None else smoke_axial["observed_robot_count"],
        "m1_axial_directional_sample_count": 0 if smoke_axial is None else smoke_axial["directional_sample_count"],
        "m1_false_candidate_count_eval_only": sum(
            row["matched_branch_eval_only"] == "FALSE"
            for row in smoke["M1_CROSS_BASELINE"]["details"]
        ),
        "smoke_pass": smoke_pass,
    }]
    _write(output / "smoke_test_summary.csv", smoke_rows)
    _write(output / "exp035_false_diagonal_regression.csv", [diagonal])
    if not smoke_pass:
        return {"smoke_pass": False, "smoke": smoke_rows[0], "output": str(output.resolve())}

    main_specs = [
        (case, m0_frames if case == "M0_STRAIGHT" else frames, case != "M0_STRAIGHT", "MAIN")
        for case in BENCHMARK_CASES
    ]
    main_results = _run_specs(main_specs, workers)
    # Persist completed physical results before aggregation or replay.
    with (output / "main_run_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(_canonical(main_results), handle, indent=2)
    main = {row["case"]: row for row in main_results}
    main_branch_rows = [row for case in BENCHMARK_CASES for row in _branch_rows(main[case])]
    preliminary = [_case_summary(case, [row for row in main_branch_rows if row["case"] == case], False) for case in BENCHMARK_CASES]
    main_sane = (
        next(row for row in preliminary if row["case"] == "M0_STRAIGHT")["false_candidate_count"] == 0
        and all(row["true_candidate_count"] > 0 for row in preliminary if row["gt_branch_count"] > 0)
    )
    if not main_sane:
        _write(output / "case_summary.csv", preliminary)
        _write(output / "branch_summary.csv", main_branch_rows, BRANCH_FIELDS)
        return {"smoke_pass": True, "main_sane": False, "output": str(output.resolve())}

    replay_specs = [
        (case, m0_frames if case == "M0_STRAIGHT" else frames, case != "M0_STRAIGHT", "REPLAY")
        for case in BENCHMARK_CASES
    ]
    replay_results = _run_specs(replay_specs, workers)
    replay = {row["case"]: row for row in replay_results}
    replay_equal = {
        case: _canonical(main[case]["signature"]) == _canonical(replay[case]["signature"])
        for case in BENCHMARK_CASES
    }
    case_rows = [_case_summary(case, [row for row in main_branch_rows if row["case"] == case], replay_equal[case]) for case in BENCHMARK_CASES]
    total_gt = sum(row["gt_branch_count"] for row in case_rows)
    total_candidate = sum(row["true_candidate_count"] for row in case_rows)
    total_complete = sum(row["complete_count"] for row in case_rows)
    total_motion = sum(row["motion_supported_count"] for row in case_rows)
    total_false = sum(row["false_candidate_count"] for row in case_rows)
    total_false_motion = sum(row["false_motion_supported_count"] for row in case_rows)
    aggregate = {
        "case": "ALL",
        "gt_branch_count": total_gt,
        "general_candidate_count": total_candidate + total_false,
        "true_candidate_count": total_candidate,
        "false_candidate_count": total_false,
        "candidate_recall": total_candidate / max(1, total_gt),
        "complete_count": total_complete,
        "complete_recall": total_complete / max(1, total_gt),
        "motion_supported_count": total_motion,
        "motion_supported_recall": total_motion / max(1, total_gt),
        "false_motion_supported_count": total_false_motion,
        "main_replay_exact": all(replay_equal.values()),
    }
    case_rows.append(aggregate)

    axial_rows = [
        row for row in main_branch_rows
        if row["case"] in {"M1_CROSS_BASELINE", "M4_ASYMMETRIC_CROSS", "M5_UNEQUAL_WIDTH"}
        and row["matched_branch_eval_only"] == "FORWARD"
    ]
    comparison_rows = []
    for row in case_rows:
        if row["case"] == "ALL":
            old_gt, old_candidate = 13, 10
        else:
            old_gt, old_candidate = EXP036_GEOMETRY[row["case"]]
        old_recall = old_candidate / max(1, old_gt)
        comparison_rows.extend([
            {"case": row["case"], "metric": "candidate_recall", "EXP036": old_recall, "EXP039": row["candidate_recall"], "delta": row["candidate_recall"] - old_recall},
            {"case": row["case"], "metric": "complete_recall", "EXP036": old_recall, "EXP039": row["complete_recall"], "delta": row["complete_recall"] - old_recall},
            {"case": row["case"], "metric": "motion_supported_recall", "EXP036": old_recall, "EXP039": row["motion_supported_recall"], "delta": row["motion_supported_recall"] - old_recall},
            {"case": row["case"], "metric": "false_candidate_count", "EXP036": 0, "EXP039": row["false_candidate_count"], "delta": row["false_candidate_count"]},
            {"case": row["case"], "metric": "false_motion_supported_count", "EXP036": 0, "EXP039": row["false_motion_supported_count"], "delta": row["false_motion_supported_count"]},
        ])
    false_rows = [row for row in main_branch_rows if row["matched_branch_eval_only"] == "FALSE"]
    pathway_counts = Counter(row["validation_pathway"] for row in main_branch_rows if row["candidate_created"] and row["is_true_candidate_eval_only"])
    if total_candidate == 13 and total_complete == 13 and total_motion == 13 and total_false == 0 and total_false_motion == 0:
        verdict = "A_GENERAL_CANDIDATE_INTEGRATION_FULL_RECALL_NO_FALSE_REGRESSION"
    elif total_false > 0 or total_false_motion > 0:
        verdict = "C_AXIAL_RECOVERED_WITH_FALSE_REGRESSION"
    elif all(row["candidate_created"] for row in axial_rows) and total_candidate < 13:
        verdict = "D_EXISTING_BRANCH_REGRESSION"
    elif any(row["candidate_created"] for row in axial_rows):
        verdict = "B_AXIAL_RECOVERY_PARTIAL"
    elif total_candidate > 10 and (total_complete < total_candidate or total_motion < total_candidate):
        verdict = "E_GENERAL_CANDIDATE_NOT_PROPAGATED_TO_VALIDATION"
    else:
        verdict = "F_OTHER"

    _write(output / "case_summary.csv", case_rows)
    _write(output / "branch_summary.csv", main_branch_rows, BRANCH_FIELDS)
    _write(output / "axial_recovery_summary.csv", axial_rows, BRANCH_FIELDS)
    _write(output / "exp036_vs_exp039.csv", comparison_rows)
    _write(output / "false_candidate_audit.csv", false_rows, BRANCH_FIELDS)
    _write(output / "validation_pathway_summary.csv", [{"validation_pathway": key, "count": value} for key, value in sorted(pathway_counts.items())], ["validation_pathway", "count"])
    _write(output / "deterministic_replay.csv", [{"case": case, "exact_match": replay_equal[case]} for case in BENCHMARK_CASES])
    _write(output / "verdict.csv", [{
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "smoke_pass": smoke_pass,
        "main_sane": main_sane,
        "all_replay_exact": all(replay_equal.values()),
        "runtime_GT_map_used": False,
        "map_specific_runtime_logic": False,
        "orientation_specific_runtime_logic": False,
        "detector_modified": False,
        "SPH_motion_modified": False,
    }])
    _plot(output / "exp036_vs_exp039_recall.png", case_rows[:-1])
    return {
        "smoke_pass": True,
        "main_sane": True,
        "verdict": verdict,
        "aggregate": aggregate,
        "case_rows": case_rows,
        "axial_rows": axial_rows,
        "pathways": dict(pathway_counts),
        "replay_equal": replay_equal,
        "diagonal": diagonal,
        "output": str(output.resolve()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--m0-frames", type=int, default=120)
    parser.add_argument("--smoke-m1-frames", type=int, default=300)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args.output, args.frames, args.m0_frames, args.smoke_m1_frames, args.workers)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, **result}, indent=2, default=str))


if __name__ == "__main__":
    main()
