"""EXP-050 physical-gap to General Candidate consistency validation.

The representative EXP-049 Anchor scan is consumed read-only.  Every physical
gap is passed to the existing General Candidate builder without direction or
map labels.  Rigid rotations, input-order permutations, and mouth-endpoint
swaps are diagnostic transforms only; production detector and topology code
are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_provisional_anchor_stationary_pointcloud_integration import (  # noqa: E402
    BOOTSTRAP_ALIAS,
    M0_ALIAS,
    IntegrationRun,
    _run_signature,
)
from junction_detection.pointcloud.general_branch_candidate import (  # noqa: E402
    FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG,
    build_general_branch_candidate,
)
from pygame_simulator.pre_exploration_wall_topology_sph_validation import (  # noqa: E402
    _axis_frame,
)


EXPERIMENT_ID = "EXP-050"
EXPERIMENT_NAME = "General Physical-Gap to Branch-Candidate Consistency Validation"
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/general_gap_candidate_consistency"
)
ROTATIONS = (0, 90, 180, 270)
PROTECTED_PATHS = (
    "pygame_simulator/pre_exploration_general_pipeline_simulator.py",
    "junction_detection/pointcloud/lidar_profile_junction_detector.py",
    "junction_detection/integration/run_lidar_profile_junction_detection.py",
    "pygame_simulator/lidar_junction_controlled_approach_visualizer.py",
    "pygame_simulator/full_junction_pipeline_visualizer.py",
    "junction_detection/integration/run_provisional_anchor_stationary_pointcloud_integration.py",
    "junction_detection/integration/run_wall_topology_branch_opening_diagnostic.py",
    "junction_detection/pointcloud/general_branch_candidate.py",
    "junction_detection/integration/branch_candidate_identity.py",
    "junction_detection/pointcloud/pointcloud_junction_detector_sensor_enhanced.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in PROTECTED_PATHS}


def _write(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    return np.array(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _axial_difference(first: float, second: float) -> float:
    return float(abs((float(first) - float(second) + 90.0) % 180.0 - 90.0))


def _angle(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _rotate_topology(result: dict[str, Any], degrees: float) -> dict[str, Any]:
    """Rigidly rotate one already-observed topology object."""
    matrix = _rotation(degrees)
    transformed = {
        "case": result["case"],
        "snapshot": dict(result["snapshot"]),
        "width_hat": float(result["width_hat"]),
        "points": np.asarray(result["points"], dtype=float) @ matrix.T,
        "segments": [],
        "endpoints": [],
        "gaps": [],
    }
    for source in result["segments"]:
        row = deepcopy(source)
        row["start"] = matrix @ np.asarray(source["start"], dtype=float)
        row["end"] = matrix @ np.asarray(source["end"], dtype=float)
        row["orientation_deg"] = _wrap_degrees(
            float(source["orientation_deg"]) + degrees
        )
        transformed["segments"].append(row)
    for source in result["endpoints"]:
        row = deepcopy(source)
        row["point"] = matrix @ np.asarray(source["point"], dtype=float)
        transformed["endpoints"].append(row)
    for source in result["gaps"]:
        row = deepcopy(source)
        row["gap_center"] = matrix @ np.asarray(source["gap_center"], dtype=float)
        row["gap_orientation_deg"] = _wrap_degrees(
            float(source["gap_orientation_deg"]) + degrees
        )
        row["estimated_direction_local"] = _wrap_degrees(
            float(source["estimated_direction_local"]) + degrees
        )
        transformed["gaps"].append(row)
    return transformed


def _reorder_topology(
    result: dict[str, Any], ordering_case: str
) -> dict[str, Any]:
    transformed = deepcopy(result)
    if ordering_case == "ORIGINAL":
        return transformed
    if ordering_case == "REVERSED_SEGMENTS_AND_GAPS":
        transformed["segments"].reverse()
        transformed["gaps"].reverse()
        return transformed
    if ordering_case == "REVERSED_GAPS":
        transformed["gaps"].reverse()
        return transformed
    if ordering_case == "DETERMINISTIC_SHUFFLE":
        generator = random.Random(50)
        generator.shuffle(transformed["segments"])
        generator.shuffle(transformed["gaps"])
        return transformed
    if ordering_case == "MOUTH_ENDPOINTS_SWAPPED":
        for gap in transformed["gaps"]:
            gap["endpoint_a"], gap["endpoint_b"] = (
                gap["endpoint_b"],
                gap["endpoint_a"],
            )
        return transformed
    raise ValueError(f"unknown ordering case: {ordering_case}")


def _incident_rows(
    case_id: str,
    transform_deg: int,
    gap: dict[str, Any],
    descriptor: Any,
    endpoints: dict[int, dict[str, Any]],
    segments: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tangent = np.asarray(descriptor.gap_tangent_local, dtype=float)
    tangent_deg = float(descriptor.gap_tangent_deg_local)
    for side, endpoint_id in (
        ("A", int(gap["endpoint_a"])),
        ("B", int(gap["endpoint_b"])),
    ):
        endpoint = endpoints[endpoint_id]
        for segment_id in endpoint["segment_ids"]:
            segment = segments[int(segment_id)]
            angle = math.radians(float(segment["orientation_deg"]))
            segment_axis = np.array([math.cos(angle), math.sin(angle)])
            signed_alignment = float(np.dot(tangent, segment_axis))
            rows.append(
                {
                    "case_id": case_id,
                    "transform_deg": transform_deg,
                    "gap_id": int(gap["gap_id"]),
                    "endpoint_side": side,
                    "endpoint_id": endpoint_id,
                    "endpoint_type": str(endpoint["endpoint_type"]),
                    "endpoint_valid": bool(endpoint["valid"]),
                    "incident_segment_id": int(segment_id),
                    "incident_segment_length": float(segment["length"]),
                    "incident_segment_orientation_deg": float(segment["orientation_deg"]),
                    "gap_tangent_deg": tangent_deg,
                    "signed_tangent_alignment": signed_alignment,
                    "absolute_tangent_alignment": abs(signed_alignment),
                    "axial_alignment_error_deg": _axial_difference(
                        float(segment["orientation_deg"]), tangent_deg
                    ),
                    "within_existing_axial_support_rule": _axial_difference(
                        float(segment["orientation_deg"]), tangent_deg
                    ) <= FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG,
                    "existing_resolution_deg": FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG,
                }
            )
    return rows


def _trace_rows(
    case_id: str,
    transform_deg: int,
    gap: dict[str, Any],
    descriptor: Any,
    candidate: Any | None,
    endpoints: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    first = endpoints[int(gap["endpoint_a"])]
    second = endpoints[int(gap["endpoint_b"])]
    mouth_norm = float(
        np.linalg.norm(np.asarray(second["point"]) - np.asarray(first["point"]))
    )
    incident_count = len(descriptor.endpoint_a_wall_tangent_deg) + len(
        descriptor.endpoint_b_wall_tangent_deg
    )
    normal_norm = float(np.linalg.norm(descriptor.opening_normal_local))
    stages = [
        (
            "UPSTREAM_PHYSICAL_GAP_ACCEPTED",
            "PASS" if gap["candidate_valid"] else "FAIL",
            bool(gap["candidate_valid"]),
            "candidate_valid=True",
            str(gap["rejection_reason"]),
        ),
        (
            "MOUTH_GEOMETRY_NONDEGENERATE",
            "PASS" if math.isfinite(mouth_norm) and mouth_norm > 1.0e-12 else "FAIL",
            mouth_norm,
            "> 1e-12 (_unit requirement)",
            "endpoint-to-endpoint mouth vector",
        ),
        (
            "ENDPOINT_VALID_FLAG_GATE",
            "NOT_APPLICABLE",
            f"{first['valid']},{second['valid']}",
            "none",
            "General Candidate builder consumes endpoint geometry/types but does not gate endpoint.valid",
        ),
        (
            "INCIDENT_WALL_AVAILABLE",
            "PASS" if incident_count > 0 else "FAIL",
            incident_count,
            "> 0 (otherwise boundary error is infinity)",
            "incident segment axes from both mouth endpoints",
        ),
        (
            "GAP_TANGENT_AXIAL_WALL_SUPPORT",
            "PASS" if descriptor.geometry_support else "FAIL",
            float(descriptor.gap_boundary_wall_alignment_error_deg),
            f"<= {FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG} deg",
            str(descriptor.rejection_reason),
        ),
        (
            "OPENING_NORMAL_VALID",
            "PASS" if math.isfinite(normal_norm) and abs(normal_norm - 1.0) <= 1.0e-9 else "FAIL",
            normal_norm,
            "unit vector from existing _unit",
            "normal selected by free-space alignment",
        ),
        (
            "TOPOLOGY_TYPE_GATE",
            "NOT_APPLICABLE",
            "COMPLETE",
            "none",
            "topology_type is stored on candidate but not an acceptance gate",
        ),
        (
            "DUPLICATE_IDENTITY_GATE",
            "NOT_APPLICABLE",
            "not evaluated",
            "none",
            "identity is downstream of General Candidate creation",
        ),
        (
            "GENERAL_CANDIDATE_CREATED",
            "PASS" if candidate is not None else "FAIL",
            candidate is not None,
            "descriptor.rejection_reason == NONE",
            str(descriptor.rejection_reason),
        ),
    ]
    return [
        {
            "case_id": case_id,
            "transform_deg": transform_deg,
            "gap_id": int(gap["gap_id"]),
            "stage_name": stage,
            "pass_fail": state,
            "measured_value": measured,
            "reference_value_if_existing": reference,
            "reason": reason,
        }
        for stage, state, measured, reference, reason in stages
    ]


def _evaluate_topology(
    topology: dict[str, Any],
    corridor_axis: np.ndarray,
    case_id: str,
    transform_deg: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = {
        int(row["endpoint_id"]): row for row in topology["endpoints"]
    }
    segments = {int(row["segment_id"]): row for row in topology["segments"]}
    descriptors: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    for gap in topology["gaps"]:
        gap_id = int(gap["gap_id"])
        descriptor, candidate = build_general_branch_candidate(
            candidate_id=f"G{gap_id}",
            timestamp=0.0,
            topology_type="COMPLETE",
            gap=gap,
            endpoints=endpoints,
            segments=segments,
            corridor_axis_local=np.asarray(corridor_axis, dtype=float),
        )
        incident_ids = tuple(
            dict.fromkeys(
                list(endpoints[int(gap["endpoint_a"])]["segment_ids"])
                + list(endpoints[int(gap["endpoint_b"])]["segment_ids"])
            )
        )
        signed_values = []
        for segment_id in incident_ids:
            angle = math.radians(float(segments[int(segment_id)]["orientation_deg"]))
            axis = np.array([math.cos(angle), math.sin(angle)])
            signed_values.append(float(np.dot(descriptor.gap_tangent_local, axis)))
        descriptors.append(
            {
                "case_id": case_id,
                "transform_deg": transform_deg,
                "gap_id": gap_id,
                "mouth_endpoint_a_x": float(descriptor.endpoint_a_local[0]),
                "mouth_endpoint_a_y": float(descriptor.endpoint_a_local[1]),
                "mouth_endpoint_b_x": float(descriptor.endpoint_b_local[0]),
                "mouth_endpoint_b_y": float(descriptor.endpoint_b_local[1]),
                "mouth_center_x": float(descriptor.gap_center_local[0]),
                "mouth_center_y": float(descriptor.gap_center_local[1]),
                "gap_width": float(descriptor.gap_width),
                "gap_tangent_x": float(descriptor.gap_tangent_local[0]),
                "gap_tangent_y": float(descriptor.gap_tangent_local[1]),
                "gap_tangent_deg": float(descriptor.gap_tangent_deg_local),
                "opening_normal_x": float(descriptor.opening_normal_local[0]),
                "opening_normal_y": float(descriptor.opening_normal_local[1]),
                "opening_normal_deg": float(descriptor.opening_normal_deg_local),
                "topology_state": "COMPLETE",
                "endpoint_a_type": str(descriptor.endpoint_a_type),
                "endpoint_b_type": str(descriptor.endpoint_b_type),
                "endpoint_a_valid": bool(endpoints[int(gap["endpoint_a"])]["valid"]),
                "endpoint_b_valid": bool(endpoints[int(gap["endpoint_b"])]["valid"]),
                "incident_wall_segments": json.dumps(incident_ids),
                "incident_wall_count": len(incident_ids),
                "wall_support_count": int(gap["boundary_support_left"] + gap["boundary_support_right"]),
                "boundary_support_a": int(gap["boundary_support_left"]),
                "boundary_support_b": int(gap["boundary_support_right"]),
                "support_asymmetry": abs(int(gap["boundary_support_left"]) - int(gap["boundary_support_right"])),
                "minimum_signed_tangent_alignment": min(signed_values, default=math.nan),
                "maximum_absolute_tangent_alignment": max((abs(value) for value in signed_values), default=math.nan),
                "minimum_axial_alignment_error_deg": float(descriptor.gap_boundary_wall_alignment_error_deg),
                "geometry_support": bool(descriptor.geometry_support),
                "candidate_created": candidate is not None,
                "rejection_reason": str(descriptor.rejection_reason),
            }
        )
        traces.extend(
            _trace_rows(
                case_id, transform_deg, gap, descriptor, candidate, endpoints
            )
        )
        wall_rows.extend(
            _incident_rows(
                case_id,
                transform_deg,
                gap,
                descriptor,
                endpoints,
                segments,
            )
        )
    return descriptors, traces, wall_rows


def _decision_map(rows: Iterable[dict[str, Any]]) -> dict[int, tuple[bool, str]]:
    return {
        int(row["gap_id"]): (
            bool(row["candidate_created"]),
            str(row["rejection_reason"]),
        )
        for row in rows
    }


def _geometry_equivalent(
    base: dict[str, Any], transformed: dict[str, Any], degrees: int
) -> bool:
    matrix = _rotation(degrees)
    return bool(
        math.isclose(float(base["gap_width"]), float(transformed["gap_width"]), abs_tol=1.0e-9)
        and np.allclose(
            matrix @ np.array([base["mouth_center_x"], base["mouth_center_y"]]),
            np.array([transformed["mouth_center_x"], transformed["mouth_center_y"]]),
            atol=1.0e-9,
        )
        and math.isclose(
            _axial_difference(
                float(base["gap_tangent_deg"]) + degrees,
                float(transformed["gap_tangent_deg"]),
            ),
            0.0,
            abs_tol=1.0e-9,
        )
    )


def _canonical_rows(rows: Iterable[dict[str, Any]]) -> str:
    canonical = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, float):
                clean[key] = "NaN" if not math.isfinite(value) else round(value, 9)
            else:
                clean[key] = value
        canonical.append(clean)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _plot_geometry(
    path: Path, topology: dict[str, Any], descriptor_rows: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    hit = np.asarray(topology["snapshot"]["hit"], dtype=bool)
    points = np.asarray(topology["points"])
    axis.scatter(points[hit, 0], points[hit, 1], s=8, color="0.7", label="LiDAR hits")
    for segment in topology["segments"]:
        start, end = np.asarray(segment["start"]), np.asarray(segment["end"])
        axis.plot([start[0], end[0]], [start[1], end[1]], color="tab:blue", linewidth=2)
        axis.text(*(0.5 * (start + end)), f"W{segment['segment_id']}", fontsize=7)
    by_gap = {int(row["gap_id"]): row for row in descriptor_rows}
    for gap in topology["gaps"]:
        row = by_gap[int(gap["gap_id"])]
        first = np.array([row["mouth_endpoint_a_x"], row["mouth_endpoint_a_y"]])
        second = np.array([row["mouth_endpoint_b_x"], row["mouth_endpoint_b_y"]])
        center = np.array([row["mouth_center_x"], row["mouth_center_y"]])
        tangent = np.array([row["gap_tangent_x"], row["gap_tangent_y"]])
        normal = np.array([row["opening_normal_x"], row["opening_normal_y"]])
        color = "tab:green" if row["candidate_created"] else "tab:orange"
        axis.plot([first[0], second[0]], [first[1], second[1]], color=color, linewidth=5)
        axis.quiver(*center, *(normal * 25.0), angles="xy", scale_units="xy", scale=1, color=color)
        axis.quiver(*center, *(tangent * 18.0), angles="xy", scale_units="xy", scale=1, color="tab:purple")
        state = "ACCEPT" if row["candidate_created"] else "REJECT"
        axis.text(center[0], center[1], f"G{gap['gap_id']} {state}", fontsize=8)
    axis.scatter(0.0, 0.0, marker="*", s=130, color="black", label="Anchor-local LiDAR")
    axis.set(
        title="Physical gaps and existing General Candidate decisions",
        xlabel="anchor-local x",
        ylabel="anchor-local y",
        aspect="equal",
        xlim=(-160, 160),
        ylim=(-160, 160),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_evidence(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: int(row["gap_id"]))
    labels = [f"G{row['gap_id']}" for row in rows]
    errors = [float(row["minimum_axial_alignment_error_deg"]) for row in rows]
    supports = [int(row["wall_support_count"]) for row in rows]
    colors = ["tab:green" if row["candidate_created"] else "tab:orange" for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    axes[0].bar(labels, errors, color=colors)
    axes[0].axhline(FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG, color="tab:red", linestyle="--", label=f"existing {FROZEN_ENDPOINT_FRAME_RESOLUTION_DEG:g} deg rule")
    axes[0].set(ylabel="minimum axial error [deg]", title="Accepted vs rejected physical evidence")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, supports, color=colors)
    axes[1].set(xlabel="physical gap ID", ylabel="boundary wall support count")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_rotation(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = sorted({int(row["base_gap_id"]) for row in rows})
    matrix = np.zeros((len(gaps), len(ROTATIONS)))
    labels: list[list[str]] = [["" for _ in ROTATIONS] for _ in gaps]
    for row in rows:
        y = gaps.index(int(row["base_gap_id"]))
        x = ROTATIONS.index(int(row["rotation_deg"]))
        matrix[y, x] = 1.0 if row["candidate_created"] else 0.0
        labels[y][x] = "PASS" if row["candidate_created"] else "REJECT"
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    for y in range(len(gaps)):
        for x in range(len(ROTATIONS)):
            axis.text(x, y, labels[y][x], ha="center", va="center", fontsize=8)
    axis.set(
        title="Rigid-rotation decision consistency",
        xlabel="rigid rotation [deg]",
        ylabel="physical gap identity",
        xticks=range(len(ROTATIONS)),
        xticklabels=ROTATIONS,
        yticks=range(len(gaps)),
        yticklabels=[f"G{gap}" for gap in gaps],
    )
    figure.colorbar(image, ax=axis, ticks=(0, 1), label="candidate created")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_experiment(
    output: Path, frames: int, m0_frames: int
) -> dict[str, Any]:
    hashes_before = protected_hashes()
    representative = IntegrationRun(BOOTSTRAP_ALIAS).run(frames)
    runtime = representative.runtime_result
    if runtime is None:
        raise AssertionError("representative stationary Point Cloud was not invoked")
    base_topology = runtime.topology_result
    snapshot = representative.frontend.current
    base_corridor, _ = _axis_frame(float(snapshot.stable_orientation_deg))

    descriptor_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    by_rotation: dict[int, list[dict[str, Any]]] = {}
    for degrees in ROTATIONS:
        topology = _rotate_topology(base_topology, degrees)
        corridor = _rotation(degrees) @ base_corridor
        descriptors, traces, walls = _evaluate_topology(
            topology, corridor, BOOTSTRAP_ALIAS, degrees
        )
        descriptor_rows.extend(descriptors)
        trace_rows.extend(traces)
        wall_rows.extend(walls)
        by_rotation[degrees] = descriptors

    base_rows = by_rotation[0]
    base_decisions = _decision_map(base_rows)
    base_by_gap = {int(row["gap_id"]): row for row in base_rows}
    rotation_rows: list[dict[str, Any]] = []
    for degrees, rows in by_rotation.items():
        for row in rows:
            gap_id = int(row["gap_id"])
            decision = (bool(row["candidate_created"]), str(row["rejection_reason"]))
            rotation_rows.append(
                {
                    "base_gap_id": gap_id,
                    "rotation_deg": degrees,
                    "candidate_created": decision[0],
                    "rejection_reason": decision[1],
                    "matches_base_decision": decision == base_decisions[gap_id],
                    "geometry_equivalent": _geometry_equivalent(
                        base_by_gap[gap_id], row, degrees
                    ),
                }
            )

    ordering_cases = (
        "ORIGINAL",
        "REVERSED_SEGMENTS_AND_GAPS",
        "REVERSED_GAPS",
        "DETERMINISTIC_SHUFFLE",
        "MOUTH_ENDPOINTS_SWAPPED",
    )
    ordering_rows: list[dict[str, Any]] = []
    ordering_decisions: dict[str, dict[int, tuple[bool, str]]] = {}
    for ordering_case in ordering_cases:
        topology = _reorder_topology(base_topology, ordering_case)
        rows, _, _ = _evaluate_topology(
            topology, base_corridor, BOOTSTRAP_ALIAS, 0
        )
        decisions = _decision_map(rows)
        ordering_decisions[ordering_case] = decisions
        row_by_gap = {int(row["gap_id"]): row for row in rows}
        for gap_id in sorted(base_decisions):
            decision = decisions[gap_id]
            base = base_by_gap[gap_id]
            transformed = row_by_gap[gap_id]
            ordering_rows.append(
                {
                    "ordering_case": ordering_case,
                    "gap_id": gap_id,
                    "candidate_created": decision[0],
                    "rejection_reason": decision[1],
                    "matches_reference": decision == base_decisions[gap_id],
                    "tangent_axis_matches_reference": math.isclose(
                        _axial_difference(
                            float(base["gap_tangent_deg"]),
                            float(transformed["gap_tangent_deg"]),
                        ),
                        0.0,
                        abs_tol=1.0e-9,
                    ),
                    "representation_case": ordering_case == "MOUTH_ENDPOINTS_SWAPPED",
                }
            )

    replay = IntegrationRun(BOOTSTRAP_ALIAS).run(frames)
    replay_runtime = replay.runtime_result
    replay_topology = replay_runtime.topology_result
    replay_snapshot = replay.frontend.current
    replay_corridor, _ = _axis_frame(float(replay_snapshot.stable_orientation_deg))
    replay_rows, _, _ = _evaluate_topology(
        replay_topology, replay_corridor, BOOTSTRAP_ALIAS, 0
    )
    replay_match = bool(
        _run_signature(representative) == _run_signature(replay)
        and _canonical_rows(base_rows) == _canonical_rows(replay_rows)
    )

    m0 = IntegrationRun(M0_ALIAS).run(m0_frames)
    front = representative.frontend.summary()
    expected_front = {
        "ready_frame": 6,
        "first_open_frame": 30,
        "first_detection_frame": 36,
        "bilateral_entry_frame": 174,
        "candidate_b_trigger_frame": 180,
        "braking_start_frame": 181,
        "stop_frame": 216,
        "anchor_frame": 221,
    }
    front_equivalent = all(front[key] == value for key, value in expected_front.items())
    anchor_equivalent = bool(
        abs(float(front["anchor_x_eval_only"]) + 1.777661529818736) <= 1.0e-6
        and abs(float(front["anchor_y_eval_only"]) + 40.329315371814026) <= 1.0e-6
    )
    pointcloud_equivalent = bool(
        runtime.analysis_frame == 222
        and int(np.count_nonzero(runtime.source_snapshot["hit"])) == 228
        and len(base_topology["segments"]) == 6
        and sum(bool(row["valid"]) for row in base_topology["endpoints"]) == 3
        and len(base_topology["gaps"]) == 3
        and len(runtime.candidate_rows) == 2
    )
    m0_pass = bool(
        m0.frontend.first_detection_frame is None
        and m0.frontend.anchor_enter_frame is None
        and m0.pointcloud_invocation_count == 0
        and m0.runtime_result is None
    )
    rotation_pass = all(
        bool(row["matches_base_decision"] and row["geometry_equivalent"])
        for row in rotation_rows
    )
    order_only_pass = all(
        bool(row["matches_reference"])
        for row in ordering_rows
        if not row["representation_case"]
    )
    representation_pass = all(
        bool(row["matches_reference"] and row["tangent_axis_matches_reference"])
        for row in ordering_rows
        if row["representation_case"]
    )

    classifications = []
    for gap_id, decision in sorted(base_decisions.items()):
        rotation_ok = all(
            row["matches_base_decision"]
            for row in rotation_rows
            if int(row["base_gap_id"]) == gap_id
        )
        order_ok = all(
            row["matches_reference"]
            for row in ordering_rows
            if int(row["gap_id"]) == gap_id and not row["representation_case"]
        )
        representation_ok = all(
            row["matches_reference"]
            for row in ordering_rows
            if int(row["gap_id"]) == gap_id and row["representation_case"]
        )
        if not rotation_ok:
            classification = "DIRECTION_DEPENDENT_FAILURE"
        elif not order_ok:
            classification = "ORDER_DEPENDENT_FAILURE"
        elif not representation_ok:
            classification = "REPRESENTATION_DEPENDENT_FAILURE"
        else:
            classification = "CONSISTENT_ACCEPT" if decision[0] else "CONSISTENT_REJECT"
        classifications.append(
            {
                "gap_id": gap_id,
                "base_candidate_created": decision[0],
                "base_rejection_reason": decision[1],
                "classification": classification,
            }
        )

    if not rotation_pass:
        verdict = "B_ROTATION_DEPENDENT_CANDIDATE_FAILURE"
    elif not order_only_pass:
        verdict = "C_ORDER_DEPENDENT_CANDIDATE_FAILURE"
    elif not representation_pass:
        verdict = "D_REPRESENTATION_DEPENDENT_CANDIDATE_FAILURE"
    else:
        verdict = "A_GENERAL_CANDIDATE_RULE_DIRECTION_INVARIANT"

    checks = {
        "front_equivalent": front_equivalent,
        "anchor_equivalent": anchor_equivalent,
        "pointcloud_equivalent": pointcloud_equivalent,
        "base_decisions_match_exp049": sorted(base_decisions.values()).count((True, "NONE")) == 2 and sorted(base_decisions.values()).count((False, "NO_INCIDENT_WALL_SUPPORT_ALONG_GAP_TANGENT")) == 1,
        "rotation_pass": rotation_pass,
        "ordering_pass": order_only_pass,
        "representation_pass": representation_pass,
        "m0_pass": m0_pass,
        "replay_match": replay_match,
        "runtime_gt_map_not_used": True,
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks, sort_keys=True))

    summary_rows = [
        {
            "experiment_id": EXPERIMENT_ID,
            "case_id": BOOTSTRAP_ALIAS,
            "ready_frame": front["ready_frame"],
            "first_open_frame": front["first_open_frame"],
            "detection_frame": front["first_detection_frame"],
            "bilateral_frame": front["bilateral_entry_frame"],
            "brake_ready_frame": front["candidate_b_trigger_frame"],
            "braking_frame": front["braking_start_frame"],
            "stop_frame": front["stop_frame"],
            "anchor_frame": front["anchor_frame"],
            "pointcloud_frame": runtime.analysis_frame,
            "hit_points": int(np.count_nonzero(runtime.source_snapshot["hit"])),
            "wall_segments": len(base_topology["segments"]),
            "valid_endpoints": sum(bool(row["valid"]) for row in base_topology["endpoints"]),
            "physical_gaps": len(base_topology["gaps"]),
            "general_candidates": len(runtime.candidate_rows),
            "rotation_consistent": rotation_pass,
            "ordering_consistent": order_only_pass,
            "representation_consistent": representation_pass,
            "deterministic_replay": replay_match,
            "runtime_gt_map_used": False,
        },
        {
            "experiment_id": EXPERIMENT_ID,
            "case_id": M0_ALIAS,
            "ready_frame": m0.frontend.first_ready_frame,
            "first_open_frame": m0.frontend.first_open_frame,
            "detection_frame": m0.frontend.first_detection_frame,
            "bilateral_frame": m0.frontend.bilateral_entry_frame,
            "brake_ready_frame": m0.frontend.brake_trigger_frame,
            "braking_frame": m0.frontend.braking_start_frame,
            "stop_frame": m0.frontend.stop_frame,
            "anchor_frame": m0.frontend.anchor_enter_frame,
            "pointcloud_frame": "",
            "hit_points": 0,
            "wall_segments": 0,
            "valid_endpoints": 0,
            "physical_gaps": 0,
            "general_candidates": 0,
            "rotation_consistent": "NOT_APPLICABLE",
            "ordering_consistent": "NOT_APPLICABLE",
            "representation_consistent": "NOT_APPLICABLE",
            "deterministic_replay": "NOT_APPLICABLE",
            "runtime_gt_map_used": False,
        },
    ]
    verdict_row = {
        "verdict": verdict,
        **checks,
        "accepted_gap_count": sum(decision[0] for decision in base_decisions.values()),
        "rejected_gap_count": sum(not decision[0] for decision in base_decisions.values()),
        "gap_classifications": json.dumps(classifications, sort_keys=True),
        "production_changed": False,
        "detector_changed": False,
    }

    descriptor_fields = tuple(descriptor_rows[0])
    trace_fields = tuple(trace_rows[0])
    wall_fields = tuple(wall_rows[0])
    rotation_fields = tuple(rotation_rows[0])
    ordering_fields = tuple(ordering_rows[0])
    summary_fields = tuple(summary_rows[0])
    _write(output / "gap_descriptor.csv", descriptor_rows, descriptor_fields)
    _write(output / "candidate_acceptance_trace.csv", trace_rows, trace_fields)
    _write(output / "gap_wall_support.csv", wall_rows, wall_fields)
    _write(output / "rotation_consistency.csv", rotation_rows, rotation_fields)
    _write(output / "ordering_consistency.csv", ordering_rows, ordering_fields)
    _write(output / "case_summary.csv", summary_rows, summary_fields)
    _write(output / "verdict.csv", [verdict_row], tuple(verdict_row))
    _plot_geometry(output / "gap_candidate_geometry.png", base_topology, base_rows)
    _plot_evidence(output / "accepted_vs_rejected_gap_evidence.png", base_rows)
    _plot_rotation(output / "rotation_consistency.png", rotation_rows)

    hashes_after = protected_hashes()
    hash_rows = [
        {
            "path": path,
            "sha256_before": hashes_before[path],
            "sha256_after": hashes_after[path],
            "unchanged": hashes_before[path] == hashes_after[path],
        }
        for path in hashes_before
    ]
    _write(
        output / "protected_hashes.csv",
        hash_rows,
        ("path", "sha256_before", "sha256_after", "unchanged"),
    )
    if hashes_before != hashes_after:
        raise AssertionError("protected source hash changed during EXP-050")
    return {
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "gap_decisions": base_decisions,
        "classifications": classifications,
        "rotation_consistent": rotation_pass,
        "ordering_consistent": order_only_pass,
        "representation_consistent": representation_pass,
        "deterministic_replay": replay_match,
        "m0_pass": m0_pass,
        "protected_hashes_unchanged": True,
        "output": str(output),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--m0-frames", type=int, default=600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.frames <= 222:
        parser.error("--frames must include stationary analysis frame 222")
    if args.m0_frames <= 0:
        parser.error("--m0-frames must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_experiment(args.output, args.frames, args.m0_frames)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
