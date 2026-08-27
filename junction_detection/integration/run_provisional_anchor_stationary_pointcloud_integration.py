"""EXP-049 provisional Anchor to stationary Point-Cloud integration.

The verified EXP-048 front end runs unchanged.  Only after provisional Anchor,
hold confirmation, and the existing stationary-speed condition are all true,
one Anchor-local scan is passed to the existing enhanced Point-Cloud detector,
EXP-033 wall-topology extractor, and EXP-038/039 General Branch Candidate
constructor.  Later stationary scans are retained only for same-pose audit.
GT/map geometry is attached after runtime candidates are final and never feeds
the detector or a state transition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_wall_topology_branch_opening_diagnostic import (  # noqa: E402
    _analyze,
    _gt_mouths_eval,
    _match_candidates_eval,
)
from junction_detection.pointcloud.general_branch_candidate import (  # noqa: E402
    build_general_branch_candidate,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (  # noqa: E402
    detect_openings,
)
from pygame_simulator.lidar_junction_controlled_approach_visualizer import (  # noqa: E402
    BOOTSTRAP_ALIAS,
    M0_ALIAS,
    IntegratedSession,
    replay_signature as frontend_replay_signature,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    LIDAR_MAX_RANGE,
    MIN_SPEED,
)
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (  # noqa: E402
    _axis_frame,
)


EXPERIMENT_ID = "EXP-049"
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/"
    / "provisional_anchor_stationary_pointcloud"
)
PROTECTED_PATHS = (
    "pygame_simulator/pre_exploration_general_pipeline_simulator.py",
    "junction_detection/pointcloud/lidar_profile_junction_detector.py",
    "junction_detection/integration/run_lidar_profile_junction_detection.py",
    "junction_detection/integration/run_active_anchor_transition.py",
    "junction_detection/integration/run_local_controlled_approach_brake_trigger_shadow.py",
    "pygame_simulator/lidar_junction_controlled_approach_visualizer.py",
    "junction_detection/pointcloud/pointcloud_junction_detector_sensor_enhanced.py",
    "junction_detection/integration/run_wall_topology_branch_opening_diagnostic.py",
    "junction_detection/pointcloud/general_branch_candidate.py",
    "pygame_simulator/pre_exploration_wall_topology_sph_validation.py",
    "pygame_simulator/pre_exploration_persistent_partial_sph_validation.py",
    "pygame_simulator/single_junction_sph_dfs_lidar_front_trigger_diagnostics.py",
    "pygame_simulator/single_junction_sph_dfs_provisional_anchor_junction_confirmation.py",
)

PIPELINE_FIELDS = (
    "case_id", "frame", "time", "event", "pipeline_state", "speed",
    "junction_detected", "brake_ready", "braking", "provisional_anchor",
    "anchor_hold_confirmed", "stationary_pointcloud_active",
    "branch_candidates_ready", "runtime_gt_used",
)
SEGMENT_FIELDS = (
    "case_id", "analysis_frame", "segment_id", "beam_start", "beam_end",
    "point_count", "start_x_local", "start_y_local", "end_x_local",
    "end_y_local", "length", "orientation_deg", "fit_residual",
    "endpoint_ids", "termination_endpoint_ids",
)
CANDIDATE_FIELDS = (
    "case_id", "candidate_id", "source_gap_id", "topology_state",
    "center_x_local", "center_y_local", "center_bearing_local",
    "center_range_local", "opening_normal_x_local", "opening_normal_y_local",
    "opening_normal_deg_local", "opening_tangent_x_local",
    "opening_tangent_y_local", "opening_tangent_deg_local", "opening_width",
    "mouth_endpoint_a_x_local", "mouth_endpoint_a_y_local",
    "mouth_endpoint_b_x_local", "mouth_endpoint_b_y_local",
    "termination_count", "wall_support_count", "source_scan_count",
    "existing_detector_confidence", "gap_width_over_W_hat",
    "free_continuation", "incident_wall_alignment_error_deg",
)
EVAL_FIELDS = (
    "case_id", "candidate_id", "matched_gt_branch_eval_only",
    "is_true_outgoing_eval_only", "is_incoming_eval_only",
    "is_duplicate_eval_only", "is_false_candidate_eval_only",
    "angular_error_eval_only", "center_error_eval_only",
    "endpoint_error_eval_only", "matching_reason",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in PROTECTED_PATHS}


def _canonical_float(value: float) -> float | str:
    return value if math.isfinite(value) else "NaN"


def _angle_difference(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _scan_hash(angles: np.ndarray, ranges: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(angles, dtype=np.float64).tobytes())
    digest.update(np.asarray(ranges, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass
class StationaryRuntimeResult:
    analysis_frame: int
    analysis_time: float
    source_snapshot: dict[str, Any]
    topology_result: dict[str, Any]
    angular_openings: list[dict[str, float]]
    candidate_rows: list[dict[str, Any]]
    candidate_internal: dict[str, dict[str, Any]]


class IntegrationRun:
    """Wrap EXP-048 without feeding stationary results back into it."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.frontend = IntegratedSession(case_id)
        self.pipeline_rows: list[dict[str, Any]] = []
        self.stationary_scans: list[dict[str, Any]] = []
        self.runtime_result: StationaryRuntimeResult | None = None
        self.branch_eval_rows: list[dict[str, Any]] = []
        self._frontend_event_index = 0
        self._stationary_active = False
        self._branch_ready = False
        self.pointcloud_invocation_count = 0
        self.pointcloud_invoked_before_hold_count = 0

    def _pipeline_row(
        self,
        event: str,
        frame: int,
        timestamp: float,
        pipeline_state: str,
    ) -> dict[str, Any]:
        world = self.frontend.runner.world
        leader = self.frontend._leader()
        return {
            "case_id": self.case_id,
            "frame": frame,
            "time": timestamp,
            "event": event,
            "pipeline_state": pipeline_state,
            "speed": float(np.linalg.norm(leader.velocity)),
            "junction_detected": self.frontend.detected_latched,
            "brake_ready": self.frontend.brake_trigger_frame is not None,
            "braking": bool(world.braking_active),
            "provisional_anchor": bool(world.provisional_fixed_anchor),
            "anchor_hold_confirmed": (
                self.frontend.anchor_hold_confirmed_frame is not None
            ),
            "stationary_pointcloud_active": self._stationary_active,
            "branch_candidates_ready": self._branch_ready,
            "runtime_gt_used": False,
        }

    def _copy_frontend_events(self) -> None:
        new = self.frontend.events[self._frontend_event_index :]
        self._frontend_event_index = len(self.frontend.events)
        for event in new:
            self.pipeline_rows.append(
                self._pipeline_row(
                    str(event["event"]),
                    int(event["frame"]),
                    float(event["time"]),
                    str(event["pipeline_state"]),
                )
            )

    @staticmethod
    def _runtime_snapshot(snapshot: Any) -> dict[str, Any]:
        """Return only fields accepted by runtime topology inference."""
        margin = np.finfo(float).eps * max(1.0, LIDAR_MAX_RANGE) * 64.0
        return {
            "context": f"ANCHOR_LOCAL_FRAME_{snapshot.physics_frame}",
            "angles": snapshot.angles_deg.copy(),
            "ranges": snapshot.measured_ranges.copy(),
            "hit": snapshot.measured_ranges < LIDAR_MAX_RANGE - margin,
            "max_range": LIDAR_MAX_RANGE,
        }

    def _stationary_conditions(self) -> bool:
        world = self.frontend.runner.world
        speed = float(np.linalg.norm(self.frontend._leader().velocity))
        return bool(
            world.provisional_fixed_anchor
            and self.frontend.anchor_hold_confirmed_frame is not None
            and speed < MIN_SPEED
        )

    def _collect_stationary_scan(self, snapshot: Any) -> None:
        runtime_scan = self._runtime_snapshot(snapshot)
        self.stationary_scans.append(
            {
                "frame": snapshot.physics_frame,
                "time": snapshot.timestamp,
                "scan_sha256": _scan_hash(
                    runtime_scan["angles"], runtime_scan["ranges"]
                ),
                "hit_point_count": int(np.count_nonzero(runtime_scan["hit"])),
                "stationary_speed": snapshot.speed,
            }
        )
        if self.runtime_result is None:
            self._run_stationary_pipeline(snapshot, runtime_scan)

    def _run_stationary_pipeline(
        self, snapshot: Any, runtime_scan: dict[str, Any]
    ) -> None:
        if not self._stationary_conditions():
            self.pointcloud_invoked_before_hold_count += 1
            raise AssertionError("stationary detector invoked before eligible hold")
        self._stationary_active = True
        self.pipeline_rows.append(
            self._pipeline_row(
                "STATIONARY_POINT_CLOUD_ANALYSIS",
                snapshot.physics_frame,
                snapshot.timestamp,
                "STATIONARY_POINT_CLOUD_ANALYSIS",
            )
        )
        self.pointcloud_invocation_count += 1
        angular_openings = list(
            detect_openings(
                runtime_scan["angles"].copy(), runtime_scan["ranges"].copy()
            )
        )
        topology = _analyze(
            runtime_scan["context"],
            runtime_scan,
            float(snapshot.estimated_corridor_width),
        )
        corridor_axis, _ = _axis_frame(float(snapshot.stable_orientation_deg))
        endpoints = {
            int(row["endpoint_id"]): row for row in topology["endpoints"]
        }
        segments = {
            int(row["segment_id"]): row for row in topology["segments"]
        }
        candidate_rows: list[dict[str, Any]] = []
        candidate_internal: dict[str, dict[str, Any]] = {}
        for gap in topology["gaps"]:
            if not gap["candidate_valid"]:
                continue
            candidate_id = f"G{int(gap['gap_id'])}"
            descriptor, candidate = build_general_branch_candidate(
                candidate_id=candidate_id,
                timestamp=float(snapshot.timestamp),
                topology_type="COMPLETE",
                gap=gap,
                endpoints=endpoints,
                segments=segments,
                corridor_axis_local=corridor_axis,
            )
            if candidate is None:
                continue
            first = endpoints[int(gap["endpoint_a"])]
            second = endpoints[int(gap["endpoint_b"])]
            termination_count = sum(
                endpoint["endpoint_type"] == "WALL_TERMINATION"
                for endpoint in (first, second)
            )
            wall_support = int(
                gap["boundary_support_left"] + gap["boundary_support_right"]
            )
            candidate_rows.append(
                {
                    "case_id": self.case_id,
                    "candidate_id": candidate_id,
                    "source_gap_id": int(gap["gap_id"]),
                    "topology_state": str(candidate.topology_type),
                    "center_x_local": float(descriptor.gap_center_local[0]),
                    "center_y_local": float(descriptor.gap_center_local[1]),
                    "center_bearing_local": math.degrees(
                        math.atan2(
                            float(descriptor.gap_center_local[1]),
                            float(descriptor.gap_center_local[0]),
                        )
                    ),
                    "center_range_local": float(
                        np.linalg.norm(descriptor.gap_center_local)
                    ),
                    "opening_normal_x_local": float(
                        descriptor.opening_normal_local[0]
                    ),
                    "opening_normal_y_local": float(
                        descriptor.opening_normal_local[1]
                    ),
                    "opening_normal_deg_local": float(
                        descriptor.opening_normal_deg_local
                    ),
                    "opening_tangent_x_local": float(
                        descriptor.gap_tangent_local[0]
                    ),
                    "opening_tangent_y_local": float(
                        descriptor.gap_tangent_local[1]
                    ),
                    "opening_tangent_deg_local": float(
                        descriptor.gap_tangent_deg_local
                    ),
                    "opening_width": float(descriptor.gap_width),
                    "mouth_endpoint_a_x_local": float(
                        descriptor.endpoint_a_local[0]
                    ),
                    "mouth_endpoint_a_y_local": float(
                        descriptor.endpoint_a_local[1]
                    ),
                    "mouth_endpoint_b_x_local": float(
                        descriptor.endpoint_b_local[0]
                    ),
                    "mouth_endpoint_b_y_local": float(
                        descriptor.endpoint_b_local[1]
                    ),
                    "termination_count": int(termination_count),
                    "wall_support_count": wall_support,
                    "source_scan_count": 1,
                    "existing_detector_confidence": float(
                        candidate.candidate_reliability
                    ),
                    "gap_width_over_W_hat": float(
                        descriptor.gap_width_over_W_hat
                    ),
                    "free_continuation": float(descriptor.free_continuation),
                    "incident_wall_alignment_error_deg": float(
                        descriptor.gap_boundary_wall_alignment_error_deg
                    ),
                }
            )
            candidate_internal[candidate_id] = {
                "gap": gap,
                "descriptor": descriptor,
                "candidate": candidate,
            }
        self.runtime_result = StationaryRuntimeResult(
            analysis_frame=snapshot.physics_frame,
            analysis_time=float(snapshot.timestamp),
            source_snapshot=runtime_scan,
            topology_result=topology,
            angular_openings=angular_openings,
            candidate_rows=candidate_rows,
            candidate_internal=candidate_internal,
        )
        self._branch_ready = True
        self.pipeline_rows.append(
            self._pipeline_row(
                "BRANCH_CANDIDATES_READY",
                snapshot.physics_frame,
                snapshot.timestamp,
                "BRANCH_CANDIDATES_READY",
            )
        )

    def step(self) -> None:
        snapshot = self.frontend.advance_physics_frame()
        self._copy_frontend_events()
        if snapshot is not None and self._stationary_conditions():
            self._collect_stationary_scan(snapshot)

    def run(self, frames: int) -> "IntegrationRun":
        for _ in range(frames):
            self.step()
        if self.runtime_result is not None:
            self._posthoc_evaluate()
        return self

    def _eval_snapshot(self) -> dict[str, Any]:
        runtime = self.runtime_result
        if runtime is None:
            raise RuntimeError("no stationary result")
        snapshot = dict(runtime.source_snapshot)
        snapshot.update(
            {
                "position_eval": self.frontend._leader().position.copy(),
                "yaw_eval": float(self.frontend.runner.world.lidar_yaw_deg),
            }
        )
        return snapshot

    def _posthoc_evaluate(self) -> None:
        runtime = self.runtime_result
        if runtime is None:
            return
        eval_snapshot = self._eval_snapshot()
        mouths = _gt_mouths_eval(
            SimpleNamespace(geometry=self.frontend.runner.geometry), eval_snapshot
        )
        match_rows = _match_candidates_eval(
            self.case_id,
            runtime.topology_result["gaps"],
            runtime.topology_result["endpoints"],
            mouths,
            float(self.frontend.current.estimated_corridor_width),
        )
        matches = {int(row["gap_id"]): row for row in match_rows}
        branch_direction_local: dict[str, float] = {}
        yaw = math.radians(float(eval_snapshot["yaw_eval"]))
        world_to_local_rotation = np.array(
            [[math.cos(yaw), math.sin(yaw)],
             [-math.sin(yaw), math.cos(yaw)]],
            dtype=float,
        )
        labels = {0.0: "FORWARD", -90.0: "RIGHT", 90.0: "LEFT"}
        for index, branch in enumerate(self.frontend.runner.geometry.branches):
            label = labels.get(float(branch.angle_deg), f"BRANCH_{index}")
            direction_world = np.array(
                [
                    math.sin(math.radians(float(branch.angle_deg))),
                    math.cos(math.radians(float(branch.angle_deg))),
                ]
            )
            direction_local = world_to_local_rotation @ direction_world
            branch_direction_local[label] = math.degrees(
                math.atan2(float(direction_local[1]), float(direction_local[0]))
            )
        used: set[str] = set()
        rows: list[dict[str, Any]] = []
        for candidate in runtime.candidate_rows:
            candidate_id = str(candidate["candidate_id"])
            gap_id = int(candidate["source_gap_id"])
            match = matches[gap_id]
            label = str(match["matched_branch_eval"])
            branch_type = str(match["branch_type_eval"])
            duplicate = label not in {"FALSE", "NONE"} and label in used
            if label not in {"FALSE", "NONE"}:
                used.add(label)
            if label in branch_direction_local:
                angular_error = _angle_difference(
                    float(candidate["opening_normal_deg_local"]),
                    branch_direction_local[label],
                )
            else:
                angular_error = math.nan
            true_outgoing = bool(
                match["true_positive_eval"] and branch_type == "OUTGOING"
            )
            incoming = bool(
                match["true_positive_eval"] and branch_type == "INCOMING"
            )
            false = bool(not match["true_positive_eval"])
            rows.append(
                {
                    "case_id": self.case_id,
                    "candidate_id": candidate_id,
                    "matched_gt_branch_eval_only": label,
                    "is_true_outgoing_eval_only": true_outgoing,
                    "is_incoming_eval_only": incoming,
                    "is_duplicate_eval_only": duplicate,
                    "is_false_candidate_eval_only": false,
                    "angular_error_eval_only": angular_error,
                    "center_error_eval_only": float(match["center_error_eval"]),
                    "endpoint_error_eval_only": float(
                        match["endpoint_error_eval"]
                    ),
                    "matching_reason": (
                        "EXISTING_EXP033_ENDPOINT_AND_CENTER_MATCH"
                        if match["true_positive_eval"]
                        else "NO_EXISTING_GT_MATCH_EVAL_ONLY"
                    ),
                }
            )
        self.branch_eval_rows = rows

    def segment_rows(self) -> list[dict[str, Any]]:
        if self.runtime_result is None:
            return []
        result = self.runtime_result.topology_result
        rows = []
        for segment in result["segments"]:
            endpoints = [
                endpoint for endpoint in result["endpoints"]
                if int(segment["segment_id"]) in endpoint["segment_ids"]
            ]
            rows.append(
                {
                    "case_id": self.case_id,
                    "analysis_frame": self.runtime_result.analysis_frame,
                    "segment_id": int(segment["segment_id"]),
                    "beam_start": int(segment["beam_start"]),
                    "beam_end": int(segment["beam_end"]),
                    "point_count": int(segment["point_count"]),
                    "start_x_local": float(segment["start"][0]),
                    "start_y_local": float(segment["start"][1]),
                    "end_x_local": float(segment["end"][0]),
                    "end_y_local": float(segment["end"][1]),
                    "length": float(segment["length"]),
                    "orientation_deg": float(segment["orientation_deg"]),
                    "fit_residual": float(segment["fit_residual"]),
                    "endpoint_ids": json.dumps(
                        [int(endpoint["endpoint_id"]) for endpoint in endpoints]
                    ),
                    "termination_endpoint_ids": json.dumps(
                        [
                            int(endpoint["endpoint_id"])
                            for endpoint in endpoints
                            if endpoint["endpoint_type"] == "WALL_TERMINATION"
                        ]
                    ),
                }
            )
        return rows

    def scan_summary(self) -> dict[str, Any]:
        runtime = self.runtime_result
        return {
            "case_id": self.case_id,
            "anchor_frame": _empty(self.frontend.anchor_enter_frame),
            "analysis_start_frame": (
                "" if runtime is None else runtime.analysis_frame
            ),
            "analysis_start_time": (
                "" if runtime is None else runtime.analysis_time
            ),
            "num_stationary_scans": len(self.stationary_scans),
            "inference_source_scan_count": 0 if runtime is None else 1,
            "num_points": (
                0
                if runtime is None
                else int(np.count_nonzero(runtime.source_snapshot["hit"]))
            ),
            "stationary_scan_signatures_identical": bool(
                self.stationary_scans
                and len({row["scan_sha256"] for row in self.stationary_scans}) == 1
            ),
            "anchor_local_frame_used": runtime is not None,
            "runtime_gt_used": False,
            "detector_name": (
                "detect_openings + EXP033_wall_topology + EXP038_039_GeneralBranchCandidate"
            ),
            "angular_opening_count": (
                0 if runtime is None else len(runtime.angular_openings)
            ),
            "pointcloud_invocation_count": self.pointcloud_invocation_count,
            "pointcloud_invoked_before_hold_count": (
                self.pointcloud_invoked_before_hold_count
            ),
        }

    def case_summary(self, front_end_altered_samples: int | None = None) -> dict[str, Any]:
        runtime = self.runtime_result
        frontend = self.frontend.summary()
        candidates = [] if runtime is None else runtime.candidate_rows
        true_outgoing = sum(
            bool(row["is_true_outgoing_eval_only"])
            and not bool(row["is_duplicate_eval_only"])
            for row in self.branch_eval_rows
        )
        incoming = sum(bool(row["is_incoming_eval_only"]) for row in self.branch_eval_rows)
        duplicate = sum(bool(row["is_duplicate_eval_only"]) for row in self.branch_eval_rows)
        false = sum(bool(row["is_false_candidate_eval_only"]) for row in self.branch_eval_rows)
        segments = [] if runtime is None else runtime.topology_result["segments"]
        endpoints = [] if runtime is None else runtime.topology_result["endpoints"]
        gaps = [] if runtime is None else runtime.topology_result["gaps"]
        return {
            "experiment_id": EXPERIMENT_ID,
            "case_id": self.case_id,
            "ready_frame": frontend["ready_frame"],
            "first_open_frame": frontend["first_open_frame"],
            "detection_frame": frontend["first_detection_frame"],
            "bilateral_entry_frame": frontend["bilateral_entry_frame"],
            "brake_ready_frame": frontend["candidate_b_trigger_frame"],
            "braking_start_frame": frontend["braking_start_frame"],
            "anchor_frame": frontend["anchor_frame"],
            "anchor_x_eval_only": frontend["anchor_x_eval_only"],
            "anchor_y_eval_only": frontend["anchor_y_eval_only"],
            "analysis_start_frame": "" if runtime is None else runtime.analysis_frame,
            "wall_segment_count": len(segments),
            "valid_endpoint_count": sum(bool(row["valid"]) for row in endpoints),
            "wall_termination_count": sum(
                row["endpoint_type"] == "WALL_TERMINATION" for row in endpoints
            ),
            "physical_gap_count": sum(bool(row["candidate_valid"]) for row in gaps),
            "candidate_count": len(candidates),
            "partial_count": sum(row["topology_state"] == "PARTIAL" for row in candidates),
            "complete_count": sum(row["topology_state"] == "COMPLETE" for row in candidates),
            "true_outgoing_recovered": true_outgoing,
            "gt_outgoing_count_eval_only": 3 if self.case_id == BOOTSTRAP_ALIAS else 0,
            "outgoing_recall_eval_only": (
                true_outgoing / 3.0 if self.case_id == BOOTSTRAP_ALIAS else 0.0
            ),
            "incoming_candidate_count": incoming,
            "duplicate_candidate_count": duplicate,
            "false_candidate_count": false,
            "angular_opening_count": (
                0 if runtime is None else len(runtime.angular_openings)
            ),
            "front_end_altered_samples": _empty(front_end_altered_samples),
            "runtime_gt_or_map_used": False,
            "fixed_anchor_transitioned": False,
            "sph_runtime_integrated": False,
            "dfs_integrated": False,
        }


def _empty(value: Any) -> Any:
    return "" if value is None else value


def _write(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(list(rows))


def _frontend_sample_signature(run: IntegrationRun) -> tuple[Any, ...]:
    return tuple(
        (
            row["frame"], row["pipeline_state"], row["corridor_state"],
            row["open_candidate_count"], row["opening_group_count"],
            row["junction_detected"], row["brake_ready"],
            row["braking_active"], row["provisional_anchor"],
            round(float(row["speed"]), 9),
        )
        for row in run.frontend.timeline
    )


def _candidate_signature(run: IntegrationRun) -> tuple[Any, ...]:
    runtime = run.runtime_result
    if runtime is None:
        return ()
    return tuple(
        tuple(
            _canonical_float(float(row[key]))
            if key not in {"candidate_id", "topology_state"}
            else row[key]
            for key in (
                "candidate_id", "topology_state", "center_x_local",
                "center_y_local", "opening_normal_deg_local", "opening_width",
                "existing_detector_confidence",
            )
        )
        for row in runtime.candidate_rows
    )


def _run_signature(run: IntegrationRun) -> tuple[Any, ...]:
    return (
        frontend_replay_signature(run.frontend),
        tuple(row["scan_sha256"] for row in run.stationary_scans),
        _candidate_signature(run),
        tuple(
            (row["candidate_id"], row["matched_gt_branch_eval_only"])
            for row in run.branch_eval_rows
        ),
    )


def _plot_runtime_pointcloud(path: Path, run: IntegrationRun) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runtime = run.runtime_result
    if runtime is None:
        return
    points = runtime.topology_result["points"]
    hit = np.asarray(runtime.source_snapshot["hit"], dtype=bool)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(points[hit, 0], points[hit, 1], s=8, color="0.5", label="stationary LiDAR hits")
    axis.scatter([0.0], [0.0], marker="*", s=140, color="tab:red", label="Anchor origin")
    axis.set(xlabel="anchor-local x", ylabel="anchor-local y", title="Stationary Point Cloud at provisional Anchor")
    axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.2); axis.legend(); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_topology(path: Path, run: IntegrationRun) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runtime = run.runtime_result
    if runtime is None:
        return
    result = runtime.topology_result
    points = result["points"]; hit = np.asarray(runtime.source_snapshot["hit"], dtype=bool)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.scatter(points[hit, 0], points[hit, 1], s=5, color="0.75")
    for segment in result["segments"]:
        axis.plot([segment["start"][0], segment["end"][0]], [segment["start"][1], segment["end"][1]], linewidth=2)
    for endpoint in result["endpoints"]:
        if endpoint["valid"]:
            axis.scatter(endpoint["point"][0], endpoint["point"][1], marker="s", s=55, color="tab:orange")
            axis.text(endpoint["point"][0], endpoint["point"][1], f"E{endpoint['endpoint_id']}")
    for row in runtime.candidate_rows:
        a=np.array([row["mouth_endpoint_a_x_local"],row["mouth_endpoint_a_y_local"]]); b=np.array([row["mouth_endpoint_b_x_local"],row["mouth_endpoint_b_y_local"]]); c=np.array([row["center_x_local"],row["center_y_local"]]); n=np.array([row["opening_normal_x_local"],row["opening_normal_y_local"]])
        axis.plot([a[0],b[0]],[a[1],b[1]],color="tab:green",linewidth=4)
        axis.arrow(c[0],c[1],n[0]*25,n[1]*25,width=.5,color="tab:green",length_includes_head=True)
        axis.text(c[0],c[1],row["candidate_id"],fontweight="bold")
    axis.scatter([0],[0],marker="*",s=120,color="tab:red")
    axis.set(xlabel="anchor-local x",ylabel="anchor-local y",title="Existing wall topology and General Branch Candidates")
    axis.set_aspect("equal",adjustable="box"); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(path,dpi=150); plt.close(figure)


def _plot_eval(path: Path, run: IntegrationRun) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runtime = run.runtime_result
    if runtime is None:
        return
    eval_snapshot = run._eval_snapshot()
    mouths = _gt_mouths_eval(SimpleNamespace(geometry=run.frontend.runner.geometry),eval_snapshot)
    figure,axis=plt.subplots(figsize=(9,8)); result=runtime.topology_result; points=result["points"]; hit=np.asarray(runtime.source_snapshot["hit"],dtype=bool)
    axis.scatter(points[hit,0],points[hit,1],s=5,color="0.75",label="LiDAR hits")
    for mouth in mouths:
        a,b=np.asarray(mouth["a"]),np.asarray(mouth["b"]); axis.plot([a[0],b[0]],[a[1],b[1]],linestyle="--",linewidth=3,label=f"GT {mouth['label']} EVAL ONLY")
    for row in runtime.candidate_rows:
        c=np.array([row["center_x_local"],row["center_y_local"]]); n=np.array([row["opening_normal_x_local"],row["opening_normal_y_local"]]); axis.arrow(c[0],c[1],n[0]*25,n[1]*25,width=.5,color="black",length_includes_head=True); axis.text(c[0],c[1],row["candidate_id"])
    axis.scatter([0],[0],marker="*",s=120,color="tab:red",label="Anchor")
    axis.set(xlabel="anchor-local x",ylabel="anchor-local y",title="Branch candidate matching — GT overlay EVAL ONLY")
    axis.set_aspect("equal",adjustable="box"); axis.grid(alpha=.2); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(path,dpi=150); plt.close(figure)


def _verdict(summary: dict[str, Any], replay_match: bool, hashes_match: bool) -> dict[str, Any]:
    frontend_ok = bool(
        summary["ready_frame"] == 6
        and summary["first_open_frame"] == 30
        and summary["detection_frame"] == 36
        and summary["bilateral_entry_frame"] == 174
        and summary["brake_ready_frame"] == 180
        and summary["braking_start_frame"] == 181
        and summary["anchor_frame"] == 221
        and abs(float(summary["anchor_x_eval_only"]) + 1.777661529818736) <= 1e-6
        and abs(float(summary["anchor_y_eval_only"]) + 40.329315371814026) <= 1e-6
    )
    if not frontend_ok:
        name = "E_FRONTEND_REGRESSION"
    elif int(summary["true_outgoing_recovered"]) == 3 and int(summary["false_candidate_count"]) == 0:
        name = "A_STATIONARY_POINTCLOUD_INTEGRATION_SUCCESS"
    elif int(summary["angular_opening_count"]) >= 3 and int(summary["true_outgoing_recovered"]) < 3:
        name = "C_ANCHOR_GOOD_BUT_DETECTOR_MISSES_BRANCHES"
    elif int(summary["false_candidate_count"]) > 0:
        name = "D_POINTCLOUD_FALSE_CANDIDATES"
    else:
        name = "B_POINTCLOUD_PARTIAL_RECOVERY"
    return {
        "verdict": name,
        "front_end_equivalent": frontend_ok,
        "front_end_altered_samples": summary["front_end_altered_samples"],
        "analysis_after_anchor_hold_only": True,
        "stationary_speed_condition_enforced": True,
        "pointcloud_invoked_before_hold_count": 0,
        "anchor_local_runtime_input_only": True,
        "runtime_gt_or_map_used": False,
        "true_outgoing_recovered": summary["true_outgoing_recovered"],
        "outgoing_recall_eval_only": summary["outgoing_recall_eval_only"],
        "deterministic_replay_match": replay_match,
        "protected_hashes_unchanged": hashes_match,
        "moving_detector_changed": False,
        "candidate_b_changed": False,
        "brake_law_changed": False,
        "fixed_anchor_transitioned": False,
        "sph_integrated": False,
        "dfs_integrated": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",type=int,default=240)
    parser.add_argument("--m0-frames",type=int,default=600)
    parser.add_argument("--include-m0",action="store_true")
    parser.add_argument("--deterministic-replay",action="store_true")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv)
    if args.frames <= 222 or args.m0_frames <= 0:
        parser.error("--frames must exceed 222 and --m0-frames must be positive")
    hashes_before=protected_hashes()
    print("[EXP-049] bootstrap front end + stationary integration",flush=True)
    bootstrap=IntegrationRun(BOOTSTRAP_ALIAS).run(args.frames)
    runs=[bootstrap]
    if args.include_m0:
        print("[EXP-049] M0 negative control",flush=True)
        runs.append(IntegrationRun(M0_ALIAS).run(args.m0_frames))
    replay_match=False; front_end_altered_samples=None
    if args.deterministic_replay:
        print("[EXP-049] deterministic replay",flush=True)
        replay=IntegrationRun(BOOTSTRAP_ALIAS).run(args.frames)
        replay_match=_run_signature(bootstrap)==_run_signature(replay)
        first=_frontend_sample_signature(bootstrap); second=_frontend_sample_signature(replay)
        front_end_altered_samples=sum(a!=b for a,b in zip(first,second))+abs(len(first)-len(second))
    summaries=[]
    for run in runs:
        altered=front_end_altered_samples if run is bootstrap else 0
        summaries.append(run.case_summary(altered))
    summary=summaries[0]
    hashes_after=protected_hashes(); hashes_match=hashes_before==hashes_after
    verdict=_verdict(summary,replay_match,hashes_match)
    m0=next((row for row in summaries if row["case_id"]==M0_ALIAS),None)
    checks={
        "front_end_exact": verdict["front_end_equivalent"],
        "front_end_altered_zero": front_end_altered_samples==0 if args.deterministic_replay else True,
        "analysis_frame_222": summary["analysis_start_frame"]==222,
        "pointcloud_once": bootstrap.pointcloud_invocation_count==1,
        "no_prehold_call": bootstrap.pointcloud_invoked_before_hold_count==0,
        "anchor_local": bootstrap.runtime_result is not None,
        "candidate_csv_nonempty": bool(bootstrap.runtime_result and bootstrap.runtime_result.candidate_rows),
        "runtime_gt_false": all(not row["runtime_gt_used"] for row in bootstrap.pipeline_rows),
        "replay": replay_match if args.deterministic_replay else True,
        "hashes": hashes_match,
        "m0_clear": True if m0 is None else m0["detection_frame"]=="" and m0["candidate_count"]==0 and m0["analysis_start_frame"]=="",
    }
    if not all(checks.values()): raise AssertionError(json.dumps(checks,sort_keys=True))
    args.output.mkdir(parents=True,exist_ok=True)
    _write(args.output/"pipeline_event_timeline.csv",(row for run in runs for row in run.pipeline_rows),PIPELINE_FIELDS)
    _write(args.output/"stationary_scan_summary.csv",(run.scan_summary() for run in runs),bootstrap.scan_summary().keys())
    _write(args.output/"wall_topology_segments.csv",(row for run in runs for row in run.segment_rows()),SEGMENT_FIELDS)
    _write(args.output/"branch_candidates.csv",(row for run in runs for row in ([] if run.runtime_result is None else run.runtime_result.candidate_rows)),CANDIDATE_FIELDS)
    _write(args.output/"branch_candidate_eval.csv",(row for run in runs for row in run.branch_eval_rows),EVAL_FIELDS)
    _write(args.output/"case_summary.csv",summaries,summary.keys())
    _write(args.output/"verdict.csv",[verdict],verdict.keys())
    _write(args.output/"protected_hashes.csv",({"path":path,"sha256_before":hashes_before[path],"sha256_after":hashes_after[path],"unchanged":hashes_before[path]==hashes_after[path]} for path in PROTECTED_PATHS),("path","sha256_before","sha256_after","unchanged"))
    _plot_runtime_pointcloud(args.output/"stationary_pointcloud_at_anchor.png",bootstrap)
    _plot_topology(args.output/"wall_topology_and_candidates.png",bootstrap)
    _plot_eval(args.output/"branch_candidate_eval.png",bootstrap)
    print(f"experiment={EXPERIMENT_ID} verdict={verdict['verdict']} anchor={summary['anchor_frame']} analysis={summary['analysis_start_frame']} segments={summary['wall_segment_count']} terminations={summary['wall_termination_count']} gaps={summary['physical_gap_count']} candidates={summary['candidate_count']} recovered={summary['true_outgoing_recovered']}/3 replay={replay_match} output={args.output.resolve()}")


if __name__=="__main__": main()
