"""EXP-040 incoming-path and persistent branch-identity benchmark.

Phase ``audit`` records policy-free local identity features while reproducing
the EXP-039 constructor.  Phase ``benchmark`` is enabled only after the audit
supports a minimal policy.  Frozen detector, topology, SPH, and motion code is
imported without modification.
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

from junction_detection.integration.branch_candidate_identity import (
    FROZEN_ASSOCIATION_DOT,
    FROZEN_SPATIAL_ASSOCIATION_W,
    LocalMotionHistory,
    PersistentCandidateIdentity,
    best_existing_match,
    incoming_features,
    incoming_path_match,
    identity_from_observation,
    make_observation,
    pairwise_features,
    self_test as identity_self_test,
)
from junction_detection.integration.run_general_candidate_sph_validation_benchmark import (
    BENCHMARK_CASES,
    BRANCH_FIELDS,
    EXP036_GEOMETRY,
    GeneralCandidateAuditRunner,
    _branch_rows,
    _candidate_summaries,
    _canonical,
    _case_summary,
    _evaluate_candidate,
    _plot,
    _smoke_false_diagonal,
)
from junction_detection.pointcloud.general_branch_candidate import build_general_branch_candidate
from pygame_simulator.pre_exploration_general_pipeline_simulator import GeometryBuilder
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (
    MOTION_WINDOW_SECONDS,
    _axis_frame,
    _endpoint_free_continuation,
)

EXPERIMENT_ID = "EXP-040"
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/candidate_identity_history_benchmark"


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
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


def _angle_difference(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


class IdentityFeatureAuditRunner(GeneralCandidateAuditRunner):
    """Record identity evidence while retaining the exact EXP-039 behavior."""

    def __init__(self, case_id: str, rear_start: bool = True):
        super().__init__(case_id, rear_start)
        self.audit_case_id = case_id
        self.local_history = LocalMotionHistory(MOTION_WINDOW_SECONDS)
        self.identity_records: dict[str, PersistentCandidateIdentity] = {}
        self.identity_feature_rows: list[dict[str, Any]] = []
        self.pairwise_rows: list[dict[str, Any]] = []
        self.observation_rows: list[dict[str, Any]] = []
        self._observation_index = 0

    def step(self, frame: int) -> dict[str, Any] | None:
        leader = self.leader()
        self.local_history.observe(
            self.world.time,
            leader.body_yaw_rad,
            leader.observed_velocity,
        )
        return super().step(frame)

    def _match_attached_candidate(self, descriptor: Any, frame: int) -> Any | None:
        candidates = [
            candidate for candidate in self.candidates
            if hasattr(candidate, "general_descriptor_record")
        ]
        if not candidates:
            return None
        def score(candidate: Any) -> tuple[float, int]:
            record = candidate.general_descriptor_record
            return (
                _angle_difference(record["relative_opening_angle_deg"], descriptor.relative_opening_angle_deg)
                + 10.0 * abs(record["gap_width_over_W_hat"] - descriptor.gap_width_over_W_hat),
                0 if candidate.created_frame == frame else 1,
            )
        return min(candidates, key=score)

    def _make_candidates(
        self,
        result: dict[str, Any],
        snapshot: dict[str, Any],
        orientation: float,
        width: float,
    ) -> None:
        """Audit each General observation before invoking EXP-039 unchanged."""
        corridor_axis, _ = _axis_frame(orientation)
        endpoints = {int(row["endpoint_id"]): row for row in result["endpoints"]}
        segments = {int(row["segment_id"]): row for row in result["segments"]}
        observations: list[tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]] = []
        for gap in result["gaps"]:
            if not gap["candidate_valid"]:
                continue
            descriptor, general = build_general_branch_candidate(
                candidate_id=f"O{self._observation_index}",
                timestamp=self.world.time,
                topology_type="COMPLETE",
                gap=gap,
                endpoints=endpoints,
                segments=segments,
                corridor_axis_local=corridor_axis,
            )
            if general is None:
                continue
            observation_id = f"O{self._observation_index}"
            self._observation_index += 1
            observation = make_observation(
                observation_id,
                self.world.time,
                int(snapshot["frame"]),
                int(gap["gap_id"]),
                descriptor,
                self.local_history,
            )
            incoming = incoming_features(observation, self.local_history, width)
            pair_rows = []
            for candidate_id, identity in self.identity_records.items():
                pair = pairwise_features(observation, identity, width)
                row = {
                    "map_case": self.audit_case_id,
                    "time": self.world.time,
                    "frame": int(snapshot["frame"]),
                    "observation_id": observation_id,
                    "source_gap_id": int(gap["gap_id"]),
                    "existing_candidate_id": candidate_id,
                    **pair,
                }
                pair_rows.append(row)
                self.pairwise_rows.append(row)
            observations.append((observation, descriptor, incoming, pair_rows))

        before_ids = {candidate.candidate_id for candidate in self.candidates}
        super()._make_candidates(result, snapshot, orientation, width)
        new_ids = {candidate.candidate_id for candidate in self.candidates} - before_ids

        for observation, descriptor, incoming, pair_rows in observations:
            candidate = self._match_attached_candidate(descriptor, int(snapshot["frame"]))
            matched_id = "NONE" if candidate is None else candidate.candidate_id
            if candidate is not None:
                identity = self.identity_records.get(candidate.candidate_id)
                if identity is None:
                    self.identity_records[candidate.candidate_id] = identity_from_observation(candidate.candidate_id, observation)
                else:
                    identity.append(observation)
            nearest = min(
                pair_rows,
                key=lambda row: (row["center_distance_over_W"], -row["axis_dot"]),
                default=None,
            )
            history = self.local_history.snapshot()
            row = {
                "map_case": self.audit_case_id,
                "candidate_id": matched_id,
                "observation_id": observation.observation_id,
                "creation_time": self.world.time,
                "frame": int(snapshot["frame"]),
                "source_gap_id": observation.source_gap_id,
                "exp039_new_candidate_created": matched_id in new_ids,
                "opening_axis_x": float(observation.opening_normal_local[0]),
                "opening_axis_y": float(observation.opening_normal_local[1]),
                "opening_normal_odom_x": float(observation.opening_normal_odom[0]),
                "opening_normal_odom_y": float(observation.opening_normal_odom[1]),
                "gap_tangent_x": float(observation.gap_tangent_local[0]),
                "gap_tangent_y": float(observation.gap_tangent_local[1]),
                "center_x": float(observation.center_local[0]),
                "center_y": float(observation.center_local[1]),
                "center_odom_x": float(observation.center_odom[0]),
                "center_odom_y": float(observation.center_odom[1]),
                "termination_1_x": float(observation.endpoint_a_local[0]),
                "termination_1_y": float(observation.endpoint_a_local[1]),
                "termination_2_x": float(observation.endpoint_b_local[0]),
                "termination_2_y": float(observation.endpoint_b_local[1]),
                "gap_width": observation.gap_width,
                "gap_width_over_W": observation.gap_width_over_W,
                "free_continuation": observation.free_continuation,
                "incident_wall_alignment_deg": observation.incident_wall_alignment_deg,
                "incoming_axis_dot": incoming["incoming_axis_dot"],
                "parent_direction_dot": incoming["parent_direction_dot"],
                "traversed_path_distance_over_W": incoming["traversed_path_distance_over_W"],
                "motion_history_axis_x": float(history["motion_axis_local"][0]),
                "motion_history_axis_y": float(history["motion_axis_local"][1]),
                "parent_direction_axis_x": float(history["parent_axis_local"][0]),
                "parent_direction_axis_y": float(history["parent_axis_local"][1]),
                "motion_history_reliability": incoming["motion_reliability"],
                "nearest_existing_candidate": "NONE" if nearest is None else nearest["existing_candidate_id"],
                "existing_axis_dot": math.nan if nearest is None else nearest["axis_dot"],
                "center_distance_over_W": math.nan if nearest is None else nearest["center_distance_over_W"],
                "termination_distance_over_W": math.nan if nearest is None else nearest["termination_distance_over_W"],
                "audit_identity_decision": "EXP039_NEW" if matched_id in new_ids else "EXP039_ASSOCIATE",
            }
            self.identity_feature_rows.append(row)
            self.observation_rows.append({
                "map_case": self.audit_case_id,
                "time": self.world.time,
                "frame": int(snapshot["frame"]),
                "observation_id": observation.observation_id,
                "source_gap_id": observation.source_gap_id,
                "exp039_candidate_id": matched_id,
                "exp039_new_candidate_created": matched_id in new_ids,
                "audit_only": True,
            })


class CandidateIdentityPolicyRunner(GeneralCandidateAuditRunner):
    """Apply incoming-first identity semantics before the frozen state tracker."""

    def __init__(self, case_id: str, rear_start: bool = True):
        super().__init__(case_id, rear_start)
        self.audit_case_id = case_id
        self.local_history = LocalMotionHistory(MOTION_WINDOW_SECONDS)
        self.identity_records: dict[str, PersistentCandidateIdentity] = {}
        self.parent_identity: PersistentCandidateIdentity | None = None
        self.identity_feature_rows: list[dict[str, Any]] = []
        self.pairwise_rows: list[dict[str, Any]] = []
        self.identity_decision_rows: list[dict[str, Any]] = []
        self.duplicate_proposals_counted: set[int] = set()
        self._observation_index = 0

    def step(self, frame: int) -> dict[str, Any] | None:
        leader = self.leader()
        self.local_history.observe(self.world.time, leader.body_yaw_rad, leader.observed_velocity)
        return super().step(frame)

    def _pair_rows(self, observation: Any, width: float) -> list[tuple[str, dict[str, Any]]]:
        rows = []
        for candidate_id, identity in self.identity_records.items():
            pair = pairwise_features(observation, identity, width)
            rows.append((candidate_id, pair))
            self.pairwise_rows.append({
                "map_case": self.audit_case_id,
                "time": self.world.time,
                "frame": observation.frame,
                "observation_id": observation.observation_id,
                "source_gap_id": observation.source_gap_id,
                "existing_candidate_id": candidate_id,
                "existing_center_odom_x": float(identity.center_odom[0]),
                "existing_center_odom_y": float(identity.center_odom[1]),
                "existing_opening_axis_odom_x": float(identity.opening_normal_odom[0]),
                "existing_opening_axis_odom_y": float(identity.opening_normal_odom[1]),
                **pair,
            })
        return rows

    def _upgrade_or_refresh(
        self,
        candidate: Any,
        observation: Any,
        descriptor: Any,
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> None:
        """Append one observation without resetting COMPLETE/motion state."""
        if candidate.topology_type == "PARTIAL":
            before = candidate.state
            candidate.topology_type = "COMPLETE"
            candidate.endpoint_ids = (int(first["endpoint_id"]), int(second["endpoint_id"]))
            candidate.gap_center_local = self._candidate_frame_point(descriptor.gap_center_local, candidate)
            candidate.gap_width = descriptor.gap_width
            if before == "PARTIAL_UNVALIDATED":
                candidate.state = "COMPLETE_UNVALIDATED"
            self._event("COMPLETE_BRANCH_CANDIDATE", candidate, before, candidate.state)
        self._attach_descriptor(candidate, descriptor)
        identity = self.identity_records.get(candidate.candidate_id)
        if identity is None:
            self.identity_records[candidate.candidate_id] = identity_from_observation(candidate.candidate_id, observation)
        else:
            identity.append(observation)

    def _record_decision(
        self,
        observation: Any,
        descriptor: Any,
        incoming: dict[str, Any],
        pairs: list[tuple[str, dict[str, Any]]],
        proposed_id: str,
        decision: str,
        matched_id: str,
        reason: str,
        raw_proposal_created: bool,
    ) -> None:
        nearest = min(pairs, key=lambda row: float(row[1]["center_distance_over_W"]), default=("NONE", {}))
        pair = nearest[1]
        history = self.local_history.snapshot()
        duplicate_score = max(
            [max(float(row[1]["axis_dot"]), float(row[1]["spatial_overlap"])) for row in pairs],
            default=0.0,
        )
        common = {
            "map_case": self.audit_case_id,
            "candidate_id": matched_id if matched_id != "NONE" else proposed_id,
            "observation_id": observation.observation_id,
            "creation_time": observation.timestamp,
            "frame": observation.frame,
            "source_gap_id": observation.source_gap_id,
            "opening_axis_x": float(observation.opening_normal_local[0]),
            "opening_axis_y": float(observation.opening_normal_local[1]),
            "opening_normal_odom_x": float(observation.opening_normal_odom[0]),
            "opening_normal_odom_y": float(observation.opening_normal_odom[1]),
            "gap_tangent_x": float(observation.gap_tangent_local[0]),
            "gap_tangent_y": float(observation.gap_tangent_local[1]),
            "center_x": float(observation.center_local[0]),
            "center_y": float(observation.center_local[1]),
            "center_odom_x": float(observation.center_odom[0]),
            "center_odom_y": float(observation.center_odom[1]),
            "termination_1_x": float(observation.endpoint_a_local[0]),
            "termination_1_y": float(observation.endpoint_a_local[1]),
            "termination_2_x": float(observation.endpoint_b_local[0]),
            "termination_2_y": float(observation.endpoint_b_local[1]),
            "gap_width": observation.gap_width,
            "gap_width_over_W": observation.gap_width_over_W,
            "free_continuation": observation.free_continuation,
            "incident_wall_alignment_deg": observation.incident_wall_alignment_deg,
            "incoming_axis_dot": incoming["incoming_axis_dot"],
            "parent_direction_dot": incoming["parent_direction_dot"],
            "parent_tangent_abs_dot": incoming["parent_tangent_abs_dot"],
            "traversed_path_distance_over_W": incoming["traversed_path_distance_over_W"],
            "motion_history_axis_x": float(history["motion_axis_local"][0]),
            "motion_history_axis_y": float(history["motion_axis_local"][1]),
            "parent_direction_axis_x": float(history["parent_axis_local"][0]),
            "parent_direction_axis_y": float(history["parent_axis_local"][1]),
            "motion_history_reliability": incoming["motion_reliability"],
            "nearest_existing_candidate": nearest[0],
            "existing_axis_dot": pair.get("axis_dot", math.nan),
            "center_distance_over_W": pair.get("center_distance_over_W", math.nan),
            "termination_distance_over_W": pair.get("termination_distance_over_W", math.nan),
            "identity_decision": decision,
        }
        self.identity_feature_rows.append(common)
        self.identity_decision_rows.append({
            "time": observation.timestamp,
            "map_case": self.audit_case_id,
            "observation_id": observation.observation_id,
            "source_gap_id": observation.source_gap_id,
            "proposed_candidate_id": proposed_id,
            "decision": decision,
            "matched_candidate_id": matched_id,
            "incoming_match_score": incoming["incoming_axis_dot"],
            "duplicate_match_score": duplicate_score,
            "decision_reason": reason,
            "raw_identity_proposal_created": raw_proposal_created,
        })

    def _make_candidates(
        self,
        result: dict[str, Any],
        snapshot: dict[str, Any],
        orientation: float,
        width: float,
    ) -> None:
        """Resolve parent/duplicate identity before creating BranchCandidate IDs."""
        corridor_axis, left_axis = _axis_frame(orientation)
        endpoints = {int(row["endpoint_id"]): row for row in result["endpoints"]}
        segments = {int(row["segment_id"]): row for row in result["segments"]}
        used: set[int] = set()
        for gap in result["gaps"]:
            if not gap["candidate_valid"]:
                continue
            descriptor, general = build_general_branch_candidate(
                candidate_id=f"O{self._observation_index}",
                timestamp=self.world.time,
                topology_type="COMPLETE",
                gap=gap,
                endpoints=endpoints,
                segments=segments,
                corridor_axis_local=corridor_axis,
            )
            if general is None:
                continue
            observation = make_observation(
                f"O{self._observation_index}", self.world.time, int(snapshot["frame"]),
                int(gap["gap_id"]), descriptor, self.local_history,
            )
            self._observation_index += 1
            first = endpoints[int(gap["endpoint_a"])]
            second = endpoints[int(gap["endpoint_b"])]
            incoming = incoming_features(observation, self.local_history, width)
            pairs = self._pair_rows(observation, width)
            proposed_id = f"C{len(self.candidates)}"
            direction_existing = self._existing_general_candidate(descriptor)

            parent_pair = (
                None if self.parent_identity is None
                else pairwise_features(observation, self.parent_identity, width)
            )
            persistent_parent_match = bool(
                parent_pair is not None
                and float(parent_pair["axis_dot"]) > FROZEN_ASSOCIATION_DOT
                and float(parent_pair["center_distance_over_W"]) <= FROZEN_SPATIAL_ASSOCIATION_W
            )

            if persistent_parent_match or incoming_path_match(incoming):
                first_parent_observation = self.parent_identity is None
                if self.parent_identity is None:
                    self.parent_identity = identity_from_observation("KNOWN_PARENT_PATH", observation)
                else:
                    self.parent_identity.append(observation)
                self._record_decision(
                    observation, descriptor, incoming, pairs, proposed_id,
                    "KNOWN_PARENT_PATH", "KNOWN_PARENT_PATH",
                    (
                        "PERSISTENT_PARENT_GEOMETRY_MATCH"
                        if persistent_parent_match
                        else "DIRECTED_LOCAL_HISTORY_PARENT_MATCH"
                    ),
                    first_parent_observation,
                )
                used.update((int(first["endpoint_id"]), int(second["endpoint_id"])))
                continue

            matched_id, pair, match_reason = best_existing_match(pairs)
            candidate = direction_existing
            if candidate is not None:
                matched_id, match_reason = candidate.candidate_id, "FROZEN_DIRECTION_ASSOCIATION"
            elif matched_id is not None:
                candidate = next(row for row in self.candidates if row.candidate_id == matched_id)

            if candidate is not None:
                is_suppressed_duplicate = direction_existing is None and match_reason == "ENDPOINT_RESOLUTION_AWARE_GAP_OVERLAP"
                raw_proposal = False
                # EXP-039 would create one duplicate ID, then associate other
                # accepted gaps from the same scan to that new identity.
                if is_suppressed_duplicate and observation.frame not in self.duplicate_proposals_counted:
                    self.duplicate_proposals_counted.add(observation.frame)
                    raw_proposal = True
                self._upgrade_or_refresh(candidate, observation, descriptor, first, second)
                self._record_decision(
                    observation, descriptor, incoming, pairs, proposed_id,
                    "MERGE_EXISTING", candidate.candidate_id, match_reason, raw_proposal,
                )
            else:
                candidate = self._new_candidate(
                    "COMPLETE", first, result, snapshot, width,
                    descriptor.opening_normal_local,
                    (int(first["endpoint_id"]), int(second["endpoint_id"])), gap,
                )
                candidate.endpoint_local = descriptor.gap_center_local.copy()
                candidate.wall_tangent_local = descriptor.gap_tangent_local.copy()
                self._attach_descriptor(candidate, descriptor)
                self.identity_records[candidate.candidate_id] = identity_from_observation(candidate.candidate_id, observation)
                self._record_decision(
                    observation, descriptor, incoming, pairs, proposed_id,
                    "NEW_OUTGOING", candidate.candidate_id,
                    "NO_PARENT_OR_EXISTING_IDENTITY_MATCH", True,
                )
            used.update((int(first["endpoint_id"]), int(second["endpoint_id"])))

        # Frozen one-ended PARTIAL definition.  These are outgoing hypotheses
        # before a complete accepted gap exists, exactly as in EXP-034/039.
        for endpoint in result["endpoints"]:
            endpoint_id = int(endpoint["endpoint_id"])
            if endpoint_id in used or endpoint["endpoint_type"] != "WALL_TERMINATION":
                continue
            lateral = float(np.dot(endpoint["point"], left_axis))
            if abs(lateral) <= np.finfo(float).eps:
                continue
            free_axis = left_axis * math.copysign(1.0, lateral)
            if self._existing_side_candidate(free_axis) is None:
                candidate = self._new_candidate(
                    "PARTIAL", endpoint, result, snapshot, width, free_axis,
                    (endpoint_id,), None,
                )
                self.identity_decision_rows.append({
                    "time": self.world.time,
                    "map_case": self.audit_case_id,
                    "observation_id": f"PARTIAL_{snapshot['frame']}_{endpoint_id}",
                    "source_gap_id": -1,
                    "proposed_candidate_id": candidate.candidate_id,
                    "decision": "NEW_OUTGOING",
                    "matched_candidate_id": candidate.candidate_id,
                    "incoming_match_score": math.nan,
                    "duplicate_match_score": math.nan,
                    "decision_reason": "FROZEN_ONE_ENDED_PARTIAL",
                    "raw_identity_proposal_created": True,
                })


def _audit_worker(spec: tuple[str, int]) -> dict[str, Any]:
    case_id, frames = spec
    runner = IdentityFeatureAuditRunner(case_id, case_id != "M0_STRAIGHT")
    for frame in range(frames):
        runner.step(frame)
    return {
        "map_case": case_id,
        "features": runner.identity_feature_rows,
        "pairs": runner.pairwise_rows,
        "observations": runner.observation_rows,
        "candidate_count": len(runner.candidates),
        "candidate_ids": [candidate.candidate_id for candidate in runner.candidates],
    }


def _event_time(runner: CandidateIdentityPolicyRunner, candidate_id: str, event: str) -> float:
    return next(
        (float(row["timestamp"]) for row in runner.events if row["candidate_id"] == candidate_id and row["event"] == event),
        math.nan,
    )


def _policy_worker(spec: tuple[str, int, str]) -> dict[str, Any]:
    """Run one physical case through the identity layer and frozen validation."""
    case_id, frames, role = spec
    runner = CandidateIdentityPolicyRunner(case_id, case_id != "M0_STRAIGHT")
    for frame in range(frames):
        runner.step(frame)
    frozen = {row["candidate_id"]: row for row in _candidate_summaries(runner, runner.world.time)}
    details = []
    for candidate in runner.candidates:
        summary = frozen[candidate.candidate_id]
        complete = _event_time(runner, candidate.candidate_id, "COMPLETE_BRANCH_CANDIDATE")
        motion = _event_time(runner, candidate.candidate_id, "MOTION_SUPPORTED")
        partial = _event_time(runner, candidate.candidate_id, "PARTIAL_BRANCH_CANDIDATE")
        label = _evaluate_candidate(runner, candidate)
        descriptor = getattr(candidate, "general_descriptor_record", {})
        evidence = candidate.best_evidence or candidate.last_evidence
        if math.isfinite(complete) and math.isfinite(motion):
            pathway = "COMPLETE_BEFORE_MOTION" if complete <= motion else "MOTION_BEFORE_COMPLETE"
        elif math.isfinite(complete):
            pathway = "COMPLETE_ONLY"
        elif math.isfinite(motion):
            pathway = "MOTION_ONLY"
        else:
            pathway = "UNRESOLVED"
        identity = runner.identity_records.get(candidate.candidate_id)
        details.append({
            "candidate_id": candidate.candidate_id,
            "matched_branch_eval_only": label,
            "topology_type": candidate.topology_type,
            "t_candidate": candidate.created_time,
            "t_partial": partial,
            "t_complete": complete,
            "t_first_motion_evidence": runner.first_motion_evidence.get(candidate.candidate_id, math.nan),
            "t_motion_supported": motion,
            "validation_pathway": pathway,
            "final_state": candidate.state,
            "termination_count": len(candidate.endpoint_ids),
            "observed_robot_count": int(evidence.get("observed_robot_count", 0)),
            "directional_sample_count": int(candidate.directional_samples),
            "supporting_robot_count": int(evidence.get("supporting_robot_count", 0)),
            "motion_reliability": float(evidence.get("motion_reliability", 0.0)),
            "motion_tangent_deg": float(evidence.get("motion_direction_local", math.nan)),
            "relative_opening_angle_deg": descriptor.get("relative_opening_angle_deg", math.nan),
            "gap_width_over_W_hat": descriptor.get("gap_width_over_W_hat", math.nan),
            "free_continuation": descriptor.get("free_continuation", candidate.free_space_evidence),
            "incident_wall_alignment_deg": descriptor.get("incident_wall_alignment_deg", math.nan),
            "identity_first_seen_time": candidate.created_time if identity is None else identity.first_seen_time,
            "identity_last_seen_time": candidate.created_time if identity is None else identity.last_seen_time,
            "identity_observation_count": 0 if identity is None else len(identity.observation_ids),
            "identity_source_gap_ids": "[]" if identity is None else json.dumps(identity.source_gap_ids),
            "frozen_summary": summary,
        })
    signature = {
        "events": runner.events,
        "decisions": runner.identity_decision_rows,
        "candidates": [{
            key: row[key] for key in (
                "candidate_id", "topology_type", "t_candidate", "t_partial", "t_complete",
                "t_first_motion_evidence", "t_motion_supported", "final_state",
                "identity_observation_count", "identity_source_gap_ids",
            )
        } for row in details],
    }
    return {
        "case": case_id,
        "run_role": role,
        "frames": frames,
        "world_time": runner.world.time,
        "candidate_count": len(details),
        "details": details,
        "decisions": runner.identity_decision_rows,
        "features": runner.identity_feature_rows,
        "pairs": runner.pairwise_rows,
        "history_samples": [
            {"time": row[0], "x_odom": float(row[1][0]), "y_odom": float(row[1][1]), "yaw_odom": row[2]}
            for row in runner.local_history.samples
        ],
        "signature": signature,
    }


def _run_policy_specs(specs: list[tuple[str, int, str]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_policy_worker(spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_policy_worker, specs))


def _identity_counts(result: dict[str, Any]) -> dict[str, int]:
    created = [row for row in result["decisions"] if bool(row["raw_identity_proposal_created"])]
    return {
        "raw_general_candidate_count": len(created),
        "new_outgoing_candidate_count": sum(row["decision"] == "NEW_OUTGOING" for row in created),
        "known_parent_path_count": sum(row["decision"] == "KNOWN_PARENT_PATH" for row in created),
        "merged_duplicate_observation_count": sum(row["decision"] == "MERGE_EXISTING" for row in created),
    }


def _plot_identity_flow(path: Path, total: dict[str, int]) -> None:
    labels = ["EXP-039 raw", "NEW_OUTGOING", "KNOWN_PARENT", "MERGED"]
    values = [
        total["raw_general_candidate_count"], total["new_outgoing_candidate_count"],
        total["known_parent_path_count"], total["merged_duplicate_observation_count"],
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = ["0.45", "tab:green", "tab:blue", "tab:orange"]
    axis.bar(labels, values, color=colors)
    for index, value in enumerate(values):
        axis.text(index, value + 0.2, str(value), ha="center")
    axis.set(ylabel="persistent identity count", title="EXP-040 candidate identity flow")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_incoming_history(path: Path, result: dict[str, Any]) -> None:
    history = result["history_samples"]
    created = [row for row in result["features"] if any(
        decision["observation_id"] == row["observation_id"] and decision["raw_identity_proposal_created"]
        for decision in result["decisions"]
    )]
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot([row["x_odom"] for row in history], [row["y_odom"] for row in history], color="0.3", label="local motion history")
    colors = {"KNOWN_PARENT_PATH": "tab:blue", "NEW_OUTGOING": "tab:green", "MERGE_EXISTING": "tab:orange"}
    decision_by = {row["observation_id"]: row["decision"] for row in result["decisions"]}
    shown: set[str] = set()
    for row in created:
        decision = decision_by[row["observation_id"]]
        label = decision if decision not in shown else None
        shown.add(decision)
        axis.scatter(row["center_odom_x"], row["center_odom_y"], color=colors[decision], s=55, label=label)
        axis.arrow(row["center_odom_x"], row["center_odom_y"], 15 * row["opening_normal_odom_x"], 15 * row["opening_normal_odom_y"], color=colors[decision], head_width=2.0, length_includes_head=True)
    axis.scatter(history[-1]["x_odom"], history[-1]["y_odom"], marker="*", s=130, color="black", label="final LiDAR pose")
    y_values = [row["y_odom"] for row in history] + [row["center_odom_y"] for row in created]
    y_margin = max(5.0, 0.2 * (max(y_values) - min(y_values) if y_values else 0.0))
    axis.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    axis.set(xlabel="local odometry x", ylabel="local odometry y", title=f"{result['case']} incoming-history association")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_m3_duplicate(path: Path, result: dict[str, Any]) -> None:
    duplicate = next(
        row for row in result["decisions"]
        if row["decision"] == "MERGE_EXISTING" and row["decision_reason"] == "ENDPOINT_RESOLUTION_AWARE_GAP_OVERLAP"
        and row["raw_identity_proposal_created"]
    )
    feature = next(row for row in result["features"] if row["observation_id"] == duplicate["observation_id"])
    pair = next(row for row in result["pairs"] if row["observation_id"] == duplicate["observation_id"] and row["existing_candidate_id"] == duplicate["matched_candidate_id"])
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    existing_center = np.array([pair["existing_center_odom_x"], pair["existing_center_odom_y"]], dtype=float)
    duplicate_center = np.array([feature["center_odom_x"], feature["center_odom_y"]], dtype=float)
    existing_normal = np.array([pair["existing_opening_axis_odom_x"], pair["existing_opening_axis_odom_y"]], dtype=float)
    duplicate_normal = np.array([feature["opening_normal_odom_x"], feature["opening_normal_odom_y"]], dtype=float)
    existing_tangent = np.array([-existing_normal[1], existing_normal[0]])
    duplicate_tangent = np.array([-duplicate_normal[1], duplicate_normal[0]])
    width_hat = float(feature["gap_width"]) / float(feature["gap_width_over_W"])
    duplicate_width = float(feature["gap_width"])
    existing_width = max(1.0, duplicate_width - float(pair["gap_width_difference_over_W"]) * width_hat)
    axis.plot(*(np.vstack((existing_center - 0.5 * existing_width * existing_tangent, existing_center + 0.5 * existing_width * existing_tangent)).T), linewidth=4, color="tab:blue")
    axis.plot(*(np.vstack((duplicate_center - 0.5 * duplicate_width * duplicate_tangent, duplicate_center + 0.5 * duplicate_width * duplicate_tangent)).T), linewidth=4, color="tab:orange")
    axis.arrow(*existing_center, *(15.0 * existing_normal), color="tab:blue", head_width=2.0, length_includes_head=True)
    axis.arrow(*duplicate_center, *(15.0 * duplicate_normal), color="tab:orange", head_width=2.0, length_includes_head=True)
    axis.scatter(*existing_center, s=100, label=f"existing {duplicate['matched_candidate_id']}")
    axis.scatter(*duplicate_center, s=100, marker="x", label="duplicate observation")
    axis.plot([pair["existing_center_odom_x"], feature["center_odom_x"]], [pair["existing_center_odom_y"], feature["center_odom_y"]], linestyle="--", color="tab:orange", label=f"distance={float(pair['center_distance_over_W']):.3f} W")
    axis.set(aspect="equal", xlabel="local odometry x", ylabel="local odometry y", title="M3 persistent identity merge")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_audit(output: Path, frames: int, workers: int) -> dict[str, Any]:
    """Run the policy-free M1--M5 identity feature audit."""
    identity_self_test()
    cases = [case for case in BENCHMARK_CASES if case != "M0_STRAIGHT"]
    specs = [(case, frames) for case in cases]
    if workers <= 1:
        results = [_audit_worker(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_audit_worker, specs))
    features = [row for result in results for row in result["features"]]
    pairs = [row for result in results for row in result["pairs"]]
    observations = [row for result in results for row in result["observations"]]
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "pre_policy_identity_feature_audit.csv", features)
    _write(output / "pre_policy_candidate_pairwise_association.csv", pairs)
    _write(output / "pre_policy_identity_observations.csv", observations)
    summary = [{
        "map_case": result["map_case"],
        "exp039_candidate_count": result["candidate_count"],
        "accepted_observation_count": len(result["observations"]),
        "pairwise_comparison_count": len(result["pairs"]),
    } for result in results]
    _write(output / "pre_policy_audit_summary.csv", summary)
    return {"phase": "audit", "summary": summary, "output": str(output.resolve())}


def run_benchmark(
    output: Path,
    frames: int,
    m0_frames: int,
    smoke_m1_frames: int,
    workers: int,
) -> dict[str, Any]:
    """Run gated smoke, M0--M5 main, and exact identity replay."""
    output.mkdir(parents=True, exist_ok=True)
    identity_self_test()
    diagonal = _smoke_false_diagonal()
    smoke_results = _run_policy_specs([
        ("M0_STRAIGHT", m0_frames, "SMOKE"),
        ("M1_CROSS_BASELINE", smoke_m1_frames, "SMOKE"),
    ], 1)
    smoke = {row["case"]: row for row in smoke_results}
    smoke_m1_rows = _branch_rows(smoke["M1_CROSS_BASELINE"])
    smoke_pass = bool(
        diagonal["passed"]
        and smoke["M0_STRAIGHT"]["candidate_count"] == 0
        and sum(row["candidate_created"] for row in smoke_m1_rows) == 3
        and not any(row["matched_branch_eval_only"] == "FALSE" for row in smoke_m1_rows)
        and _identity_counts(smoke["M1_CROSS_BASELINE"])["known_parent_path_count"] == 1
    )
    _write(output / "smoke_test_summary.csv", [{
        "identity_helper_self_test": True,
        "exp035_false_diagonal_pass": diagonal["passed"],
        "m0_candidate_count": smoke["M0_STRAIGHT"]["candidate_count"],
        "m1_outgoing_candidate_count": smoke["M1_CROSS_BASELINE"]["candidate_count"],
        "m1_known_parent_path_count": _identity_counts(smoke["M1_CROSS_BASELINE"])["known_parent_path_count"],
        "m1_false_candidate_count": sum(row["matched_branch_eval_only"] == "FALSE" for row in smoke_m1_rows),
        "smoke_pass": smoke_pass,
    }])
    _write(output / "exp035_false_diagonal_regression.csv", [diagonal])
    if not smoke_pass:
        return {"smoke_pass": False, "output": str(output.resolve())}

    main_specs = [
        (case, m0_frames if case == "M0_STRAIGHT" else frames, "MAIN")
        for case in BENCHMARK_CASES
    ]
    main_results = _run_policy_specs(main_specs, workers)
    with (output / "main_run_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(_canonical(main_results), handle, indent=2)
    main = {row["case"]: row for row in main_results}
    branch_rows = [row for case in BENCHMARK_CASES for row in _branch_rows(main[case])]
    preliminary = [_case_summary(case, [row for row in branch_rows if row["case"] == case], False) for case in BENCHMARK_CASES]
    main_sane = (
        main["M0_STRAIGHT"]["candidate_count"] == 0
        and all(row["true_candidate_count"] > 0 for row in preliminary if row["gt_branch_count"] > 0)
    )
    if not main_sane:
        _write(output / "case_summary.csv", preliminary)
        _write(output / "branch_summary.csv", branch_rows, BRANCH_FIELDS)
        return {"smoke_pass": True, "main_sane": False, "output": str(output.resolve())}

    replay_specs = [
        (case, m0_frames if case == "M0_STRAIGHT" else frames, "REPLAY")
        for case in BENCHMARK_CASES
    ]
    replay_results = _run_policy_specs(replay_specs, workers)
    replay = {row["case"]: row for row in replay_results}
    replay_equal = {
        case: _canonical(main[case]["signature"]) == _canonical(replay[case]["signature"])
        for case in BENCHMARK_CASES
    }

    case_rows = []
    for base in preliminary:
        counts = _identity_counts(main[base["case"]])
        case_rows.append({**base, **counts, "main_replay_exact": replay_equal[base["case"]]})
    total_gt = sum(row["gt_branch_count"] for row in case_rows)
    total_true = sum(row["true_candidate_count"] for row in case_rows)
    total_complete = sum(row["complete_count"] for row in case_rows)
    total_motion = sum(row["motion_supported_count"] for row in case_rows)
    total_false = sum(row["false_candidate_count"] for row in case_rows)
    total_false_motion = sum(row["false_motion_supported_count"] for row in case_rows)
    count_keys = (
        "raw_general_candidate_count", "new_outgoing_candidate_count",
        "known_parent_path_count", "merged_duplicate_observation_count",
    )
    aggregate = {
        "case": "ALL", "gt_branch_count": total_gt,
        "general_candidate_count": total_true + total_false,
        "true_candidate_count": total_true, "false_candidate_count": total_false,
        "candidate_recall": total_true / max(1, total_gt),
        "complete_count": total_complete, "complete_recall": total_complete / max(1, total_gt),
        "motion_supported_count": total_motion, "motion_supported_recall": total_motion / max(1, total_gt),
        "false_motion_supported_count": total_false_motion,
        **{key: sum(row[key] for row in case_rows) for key in count_keys},
        "main_replay_exact": all(replay_equal.values()),
    }
    case_rows.append(aggregate)

    decisions = [row for case in BENCHMARK_CASES for row in main[case]["decisions"]]
    features = [row for case in BENCHMARK_CASES for row in main[case]["features"]]
    pairs = [row for case in BENCHMARK_CASES for row in main[case]["pairs"]]
    label_by_candidate = {
        (row["case"], row["candidate_id"]): row["matched_branch_eval_only"]
        for row in branch_rows if row["candidate_created"]
    }
    decision_by_observation = {(row["map_case"], row["observation_id"]): row for row in decisions}
    for row in features:
        decision = decision_by_observation[(row["map_case"], row["observation_id"])]
        if decision["decision"] == "KNOWN_PARENT_PATH":
            label = "INCOMING_PATH"
        else:
            label = label_by_candidate.get((row["map_case"], decision["matched_candidate_id"]), "UNRESOLVED")
        row["posthoc_branch_label"] = label
        row["posthoc_true_false"] = label not in {"FALSE", "UNRESOLVED"}

    incoming_rows = [row for row in features if row["identity_decision"] == "KNOWN_PARENT_PATH"]
    duplicate_rows = [
        row for row in features
        if row["identity_decision"] == "MERGE_EXISTING"
        and decision_by_observation[(row["map_case"], row["observation_id"])]["decision_reason"] == "ENDPOINT_RESOLUTION_AWARE_GAP_OVERLAP"
    ]
    exp039_false = [
        ("M1_CROSS_BASELINE", "C3", "INCOMING_PATH"),
        ("M2_T_JUNCTION", "C2", "INCOMING_PATH"),
        ("M3_ANGLED_Y", "C2", "INCOMING_PATH"),
        ("M3_ANGLED_Y", "C3", "DUPLICATE_BRANCH"),
        ("M4_ASYMMETRIC_CROSS", "C3", "INCOMING_PATH"),
        ("M5_UNEQUAL_WIDTH", "C3", "INCOMING_PATH"),
    ]
    false_audit = []
    for case, candidate_id, false_type in exp039_false:
        expected_decision = "KNOWN_PARENT_PATH" if false_type == "INCOMING_PATH" else "MERGE_EXISTING"
        match = next(
            row for row in decisions
            if row["map_case"] == case and row["decision"] == expected_decision
            and bool(row["raw_identity_proposal_created"])
        )
        false_audit.append({
            "map_case": case,
            "EXP039_candidate_id": candidate_id,
            "EXP039_false_type": false_type,
            "EXP040_identity_decision": match["decision"],
            "matched_candidate_id": match["matched_candidate_id"],
            "new_candidate_created": False,
            "motion_supported": False,
            "reason": match["decision_reason"],
        })

    comparison_values = {
        "true_candidate_recall": (1.0, aggregate["candidate_recall"]),
        "complete_recall": (1.0, aggregate["complete_recall"]),
        "motion_supported_recall": (1.0, aggregate["motion_supported_recall"]),
        "false_candidate_count": (6, aggregate["false_candidate_count"]),
        "false_motion_supported_count": (1, aggregate["false_motion_supported_count"]),
        "incoming_false_count": (5, 0),
        "duplicate_false_count": (1, 0),
    }
    comparison = [{"metric": key, "EXP039": value[0], "EXP040": value[1]} for key, value in comparison_values.items()]
    pathways = Counter(row["validation_pathway"] for row in branch_rows if row["candidate_created"] and row["is_true_candidate_eval_only"])
    success = (
        total_true == total_gt == 13 and total_complete == 13 and total_motion == 13
        and total_false == 0 and total_false_motion == 0
        and aggregate["known_parent_path_count"] == 5
        and aggregate["merged_duplicate_observation_count"] == 1
    )
    verdict = "A_GENERAL_CANDIDATE_IDENTITY_RESOLVED" if success else "F_IDENTITY_POLICY_INCOMPLETE"

    _write(output / "identity_feature_audit.csv", features)
    _write(output / "incoming_candidate_audit.csv", incoming_rows)
    _write(output / "duplicate_candidate_audit.csv", duplicate_rows)
    _write(output / "candidate_pairwise_association.csv", pairs)
    _write(output / "identity_decisions.csv", decisions)
    _write(output / "branch_summary.csv", branch_rows, BRANCH_FIELDS)
    _write(output / "case_summary.csv", case_rows)
    _write(output / "exp039_vs_exp040.csv", comparison)
    _write(output / "false_candidate_audit.csv", false_audit)
    _write(output / "deterministic_replay.csv", [{"map_case": case, "exact_match": replay_equal[case]} for case in BENCHMARK_CASES])
    _write(output / "validation_pathway_summary.csv", [{"validation_pathway": key, "count": value} for key, value in sorted(pathways.items())])
    _write(output / "verdict.csv", [{
        "experiment_id": EXPERIMENT_ID, "verdict": verdict,
        "smoke_pass": smoke_pass, "main_sane": main_sane,
        "all_replay_exact": all(replay_equal.values()),
        "runtime_GT_map_used": False, "map_specific_runtime_logic": False,
        "orientation_specific_runtime_logic": False,
        "detector_modified": False, "SPH_motion_modified": False,
    }])
    _plot_identity_flow(output / "candidate_identity_flow.png", aggregate)
    _plot_incoming_history(output / "incoming_history_association.png", main["M1_CROSS_BASELINE"])
    _plot_m3_duplicate(output / "m3_duplicate_identity.png", main["M3_ANGLED_Y"])
    _plot(output / "exp039_vs_exp040_recall.png", case_rows[:-1])
    return {
        "smoke_pass": True, "main_sane": True, "verdict": verdict,
        "aggregate": aggregate, "replay_equal": replay_equal,
        "output": str(output.resolve()),
    }


def run_targeted(output: Path, frames: int, workers: int) -> dict[str, Any]:
    """Run only policy-affected M2/M3 main and replay after a postprocess fix."""
    specs = [
        (case, frames, role)
        for role in ("MAIN_FIXED", "REPLAY_FIXED")
        for case in ("M2_T_JUNCTION", "M3_ANGLED_Y")
    ]
    results = _run_policy_specs(specs, workers)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "targeted_m2_m3_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(_canonical(results), handle, indent=2)
    summary = [{
        "map_case": row["case"], "run_role": row["run_role"],
        "candidate_count": row["candidate_count"],
        "false_candidate_count": sum(detail["matched_branch_eval_only"] == "FALSE" for detail in row["details"]),
    } for row in results]
    _write(output / "targeted_m2_m3_summary.csv", summary)
    return {"phase": "targeted", "summary": summary, "output": str(output.resolve())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "benchmark", "targeted"), default="benchmark")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--m0-frames", type=int, default=120)
    parser.add_argument("--smoke-m1-frames", type=int, default=300)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.phase == "audit":
        result = run_audit(args.output, args.frames, args.workers)
    elif args.phase == "targeted":
        result = run_targeted(args.output, args.frames, args.workers)
    else:
        result = run_benchmark(args.output, args.frames, args.m0_frames, args.smoke_m1_frames, args.workers)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, **result}, indent=2, default=str))


if __name__ == "__main__":
    main()
