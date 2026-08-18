"""Production-like inlet/turning extension of the general physical SPH benchmark.

Robots start inside one incoming corridor, move toward the confirmed Anchor,
receive a Branch-agnostic persistent exploration heading after entering the
central region, and are guided into outgoing corridors only by SPH interaction
and physical wall collision.  GT Branch tangents are used by environment walls
and evaluation labels, never by the control force or stable-motion estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from pygame_simulator.general_junction_sph_benchmark import (
    EPSILON,
    BranchGeometry,
    JunctionGeometry,
    SphParameters,
    _spiky_kernel,
    collision_mask_sanity,
    normalize_angle_deg,
    validate_geometry,
)


@dataclass(frozen=True)
class InletBenchmarkGeometry:
    """One general Junction plus the evaluation identity of its inlet."""

    geometry: JunctionGeometry
    incoming_branch_id: str

    @property
    def incoming_branch(self) -> BranchGeometry:
        return next(
            branch
            for branch in self.geometry.branches
            if branch.branch_id == self.incoming_branch_id
        )

    @property
    def outgoing_branches(self) -> tuple[BranchGeometry, ...]:
        return tuple(
            branch
            for branch in self.geometry.branches
            if branch.branch_id != self.incoming_branch_id
        )

    @property
    def incoming_travel_angle_deg(self) -> float:
        """World heading from the inlet toward the Anchor (evaluation label)."""
        return normalize_angle_deg(self.incoming_branch.angle_deg + 180.0)

    def turn_angle_deg(self, branch: BranchGeometry) -> float:
        return abs(normalize_angle_deg(
            branch.angle_deg - self.incoming_travel_angle_deg
        ))

    def turn_severity(self, branch: BranchGeometry) -> str:
        angle = self.turn_angle_deg(branch)
        if angle <= 45.0:
            return "shallow"
        if angle <= 105.0:
            return "medium"
        return "sharp"


@dataclass
class InletSphResult:
    """Physical trajectory data emitted by one inlet scenario."""

    case_row: dict[str, Any]
    segment_rows: list[dict[str, Any]] = field(default_factory=list)
    crossing_rows: list[dict[str, Any]] = field(default_factory=list)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    final_positions: np.ndarray | None = None
    entered_junction_count: int = 0
    collision_count: int = 0


def _rotate(vector: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cosines, sines = np.cos(angles), np.sin(angles)
    return np.column_stack((
        vector[:, 0] * cosines - vector[:, 1] * sines,
        vector[:, 0] * sines + vector[:, 1] * cosines,
    ))


def _initial_inlet_particles(
    case: InletBenchmarkGeometry,
    parameters: SphParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack robots in the inlet without reading an outgoing Branch tangent."""
    geometry = case.geometry
    inlet = case.incoming_branch
    rng = np.random.default_rng(geometry.seed)
    spacing = parameters.robot_radius * 2.25
    usable_half_width = inlet.width * 0.5 - parameters.robot_radius * 1.25
    laterals = np.arange(-usable_half_width, usable_half_width + spacing * 0.25, spacing)
    # Staggered rows lose one edge slot on alternating rows.
    row_count = (
        math.ceil(
            parameters.particle_count
            / max(len(laterals) - 1, 1)
        )
        + 2
    )
    usable_length = inlet.length - parameters.robot_radius * 4.0
    axial_spacing = min(spacing * 0.92, usable_length / max(row_count, 1))
    coordinates = []
    for row in range(row_count):
        axial = geometry.central_radius + inlet.length - parameters.robot_radius * 2.0 - row * axial_spacing
        offset = 0.5 * spacing if row % 2 else 0.0
        for lateral in laterals:
            shifted = lateral + offset
            if abs(shifted) > usable_half_width:
                continue
            coordinates.append(
                geometry.center_array
                + inlet.tangent * axial
                + inlet.normal * shifted
            )
            if len(coordinates) >= parameters.particle_count:
                break
        if len(coordinates) >= parameters.particle_count:
            break
    if len(coordinates) < parameters.particle_count:
        raise RuntimeError("incoming corridor cannot hold requested particles")
    positions = np.asarray(coordinates, dtype=float)
    to_anchor = geometry.center_array - positions
    directions = to_anchor / np.maximum(np.linalg.norm(to_anchor, axis=1, keepdims=True), EPSILON)
    velocities = directions * parameters.drive_speed * 0.25

    # A stratified, seeded fan is defined relative to each realized incoming
    # heading.  It contains no Branch angle and rotates with the inlet.
    offsets = np.linspace(-math.radians(165.0), math.radians(165.0), len(positions))
    rng.shuffle(offsets)
    exploration_headings = _rotate(directions, offsets)
    return positions, velocities, exploration_headings


def _turn_metadata(
    case: InletBenchmarkGeometry,
    branch: BranchGeometry,
) -> tuple[float, str]:
    return case.turn_angle_deg(branch), case.turn_severity(branch)


def simulate_inlet_sph_case(
    case: InletBenchmarkGeometry,
    parameters: SphParameters | None = None,
) -> InletSphResult:
    """Run one inlet → Anchor → turn → straight physical SPH scenario."""
    params = parameters or SphParameters(steps=1600)
    geometry = case.geometry
    valid, reason = validate_geometry(geometry, params.robot_radius)
    if not valid:
        raise ValueError(f"invalid geometry {geometry.case_id}: {reason}")
    sanity = collision_mask_sanity(geometry, params.robot_radius)
    if not sanity["pass"]:
        raise AssertionError(f"collision mask sanity failed: {sanity}")

    positions, velocities, exploration_headings = _initial_inlet_particles(
        case, params
    )
    sample_positions = positions.copy()
    previous_positions = positions.copy()
    entered_junction = np.zeros(len(positions), dtype=bool)
    previous_membership: list[str | None] = [case.incoming_branch_id] * len(positions)
    update_counts: dict[tuple[str, int], int] = {}
    latest_motion_angles: dict[int, float] = {}
    crossing_seen: set[tuple[str, int]] = set()
    segments: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    reference_density: float | None = None
    collision_count = 0
    minimum_pair_distance = float("inf")

    for step in range(params.steps):
        differences = positions[:, None, :] - positions[None, :, :]
        distance_sq = np.sum(differences * differences, axis=2)
        distances = np.sqrt(np.maximum(distance_sq, EPSILON))
        np.fill_diagonal(distances, 0.0)
        pair_distances = distances[np.triu_indices(len(positions), 1)]
        minimum_pair_distance = min(
            minimum_pair_distance,
            float(np.min(pair_distances)),
        )
        kernel = _spiky_kernel(distances, params.smoothing_length)
        kernel[distances > params.smoothing_length] = 0.0
        density = np.sum(kernel, axis=1)
        if reference_density is None:
            reference_density = max(
                float(np.median(density)) * params.rest_density_scale,
                EPSILON,
            )
        ratio = density / reference_density
        pressure = (
            params.pressure_gain
            * density
            * np.maximum(ratio * ratio - 1.0, 0.0)
        )

        q = np.maximum(0.0, 1.0 - distances / params.smoothing_length)
        gradient_magnitude = (
            -30.0 / (math.pi * params.smoothing_length**3) * q**2
        )
        gradient_magnitude[
            (distances <= EPSILON)
            | (distances > params.smoothing_length)
        ] = 0.0
        pair_directions = differences / np.maximum(
            distances[:, :, None], EPSILON
        )
        gradients = pair_directions * gradient_magnitude[:, :, None]
        coefficient = (
            pressure[:, None]
            / np.maximum(density[:, None] ** 2, EPSILON)
            + pressure[None, :]
            / np.maximum(density[None, :] ** 2, EPSILON)
        )
        pressure_force = np.sum(
            -coefficient[:, :, None] * gradients,
            axis=1,
        )
        kernel_zero = float(_spiky_kernel(
            np.asarray([0.0]), params.smoothing_length
        )[0])
        normalized_kernel = kernel / max(kernel_zero, EPSILON)
        viscosity_force = params.viscosity_gain * np.sum(
            (velocities[None, :, :] - velocities[:, None, :])
            * normalized_kernel[:, :, None],
            axis=1,
        )
        overlap = np.maximum(
            0.0,
            params.robot_radius * 2.0 - distances,
        )
        overlap[
            (distances <= EPSILON)
            | (distances >= params.robot_radius * 2.0)
        ] = 0.0
        overlap_force = params.overlap_gain * np.sum(
            pair_directions * overlap[:, :, None],
            axis=1,
        )

        relative = positions - geometry.center_array
        radii = np.linalg.norm(relative, axis=1)
        entered_junction |= radii <= geometry.central_radius * 0.64
        to_anchor = geometry.center_array - positions
        anchor_direction = to_anchor / np.maximum(
            np.linalg.norm(to_anchor, axis=1, keepdims=True),
            EPSILON,
        )
        desired_direction = np.where(
            entered_junction[:, None],
            exploration_headings,
            anchor_direction,
        )
        drive_force = params.drive_gain * (
            desired_direction * params.drive_speed - velocities
        )
        acceleration = (
            pressure_force
            + viscosity_force
            + overlap_force
            + drive_force
            - params.linear_drag * velocities
        )
        velocities += acceleration * params.dt
        speeds = np.linalg.norm(velocities, axis=1)
        over_speed = speeds > params.maximum_speed
        velocities[over_speed] *= (
            params.maximum_speed / speeds[over_speed]
        )[:, None]
        proposed = positions + velocities * params.dt
        for robot_id in range(len(proposed)):
            if not bool(geometry.contains(
                proposed[robot_id], params.robot_radius
            )[0]):
                collision_count += 1
                corrected, wall_normal = geometry.project_point(
                    proposed[robot_id], params.robot_radius
                )
                proposed[robot_id] = corrected
                normal_speed = float(
                    velocities[robot_id] @ wall_normal
                )
                if normal_speed > 0.0:
                    velocities[robot_id] -= (
                        wall_normal * normal_speed * 1.15
                    )
                velocities[robot_id] *= 0.82
        previous_positions[:] = positions
        positions = proposed

        # Sparse evaluation-only traces retain the physical inlet and Junction
        # transit for plots. They are not consumed by either tangent estimator.
        if step % (params.sample_every * 2) == 0:
            for robot_id, point in enumerate(positions):
                traces.append({
                    "case_id": geometry.case_id,
                    "robot_id": robot_id,
                    "frame": step,
                    "time": step * params.dt,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "entered_junction": bool(entered_junction[robot_id]),
                })

        memberships: list[str | None] = []
        membership_data: list[tuple[BranchGeometry | None, float, float]] = []
        for robot_id, point in enumerate(positions):
            branch, axial, lateral = geometry.evaluation_branch(
                point, params.robot_radius
            )
            if (
                not entered_junction[robot_id]
                or branch is None
                or branch.branch_id == case.incoming_branch_id
            ):
                memberships.append(None)
                membership_data.append((None, float("nan"), float("nan")))
                continue
            memberships.append(branch.branch_id)
            membership_data.append((branch, axial, lateral))
            key = (branch.branch_id, robot_id)
            if key not in crossing_seen:
                crossing_seen.add(key)
                crossing = sample_positions[robot_id].copy()
                turn_angle, severity = _turn_metadata(case, branch)
                crossings.append({
                    "case_id": geometry.case_id,
                    "branch_id": branch.branch_id,
                    "robot_id": robot_id,
                    "frame": step,
                    "time": step * params.dt,
                    "crossing_x": float(crossing[0]),
                    "crossing_y": float(crossing[1]),
                    "outward_dx": float(
                        positions[robot_id, 0]
                        - previous_positions[robot_id, 0]
                    ),
                    "outward_dy": float(
                        positions[robot_id, 1]
                        - previous_positions[robot_id, 1]
                    ),
                    "incoming_angle_deg": (
                        case.incoming_travel_angle_deg
                    ),
                    "outgoing_gt_angle_deg": branch.angle_deg,
                    "turn_angle_deg": turn_angle,
                    "turn_severity": severity,
                })

        if step > 0 and step % params.sample_every == 0:
            displacements = positions - sample_positions
            lengths = np.linalg.norm(displacements, axis=1)
            angles = np.degrees(np.arctan2(
                displacements[:, 1],
                displacements[:, 0],
            ))
            current_angles = {
                robot_id: normalize_angle_deg(float(angles[robot_id]))
                for robot_id, branch_id in enumerate(memberships)
                if branch_id is not None
                and lengths[robot_id] > params.robot_radius * 0.08
            }
            for robot_id, branch_id in enumerate(memberships):
                branch, axial, lateral = membership_data[robot_id]
                if branch_id is None or branch is None:
                    continue
                if (
                    previous_membership[robot_id] not in {None, branch_id}
                    or lengths[robot_id] <= params.robot_radius * 0.08
                    or axial >= branch.length * 0.96
                ):
                    continue
                neighbor_delta = positions - positions[robot_id]
                neighbor_distance = np.linalg.norm(
                    neighbor_delta, axis=1
                )
                neighbor_ids = [
                    other_id
                    for other_id, other_branch in enumerate(memberships)
                    if other_id != robot_id
                    and other_branch == branch_id
                    and neighbor_distance[other_id]
                    <= params.neighbor_radius
                    and other_id in current_angles
                ]
                neighbor_angles = [
                    current_angles[other_id]
                    for other_id in neighbor_ids
                ]
                if neighbor_angles:
                    radians = np.radians(neighbor_angles)
                    mean_cos = float(np.mean(np.cos(radians)))
                    mean_sin = float(np.mean(np.sin(radians)))
                    neighbor_mean = normalize_angle_deg(math.degrees(
                        math.atan2(mean_sin, mean_cos)
                    ))
                    resultant = min(
                        1.0, math.hypot(mean_cos, mean_sin)
                    )
                    dispersion = (
                        180.0
                        if resultant <= EPSILON
                        else math.degrees(math.sqrt(max(
                            0.0, -2.0 * math.log(resultant)
                        )))
                    )
                else:
                    neighbor_mean = float("nan")
                    resultant = 0.0
                    dispersion = float("nan")
                key = (branch_id, robot_id)
                update_counts[key] = update_counts.get(key, 0) + 1
                motion_angle = current_angles[robot_id]
                latest_motion_angles[robot_id] = motion_angle
                turn_angle, severity = _turn_metadata(case, branch)
                if axial <= branch.width:
                    phase = "JUNCTION_TURNING"
                elif axial <= branch.width * 2.0:
                    phase = "BRANCH_STRAIGHTENING"
                else:
                    phase = "BRANCH_FLOW"
                segments.append({
                    "environment_id": geometry.case_id,
                    "case_id": geometry.case_id,
                    "seed": geometry.seed,
                    "topology": geometry.topology,
                    "junction_rotation_deg": geometry.rotation_deg,
                    "incoming_branch_id": case.incoming_branch_id,
                    "incoming_branch_angle_deg": (
                        case.incoming_travel_angle_deg
                    ),
                    "branch_id": branch_id,
                    "outgoing_branch_id": branch_id,
                    "gt_branch_angle_deg": branch.angle_deg,
                    "turn_angle_deg": turn_angle,
                    "turn_severity": severity,
                    "robot_id": robot_id,
                    "frame": step,
                    "time": step * params.dt,
                    "lifecycle_phase": phase,
                    "previous_x": float(sample_positions[robot_id, 0]),
                    "previous_y": float(sample_positions[robot_id, 1]),
                    "current_x": float(positions[robot_id, 0]),
                    "current_y": float(positions[robot_id, 1]),
                    "dx": float(displacements[robot_id, 0]),
                    "dy": float(displacements[robot_id, 1]),
                    "displacement_length": float(lengths[robot_id]),
                    "motion_angle_deg": motion_angle,
                    "observed_speed": float(
                        lengths[robot_id]
                        / (params.sample_every * params.dt)
                    ),
                    "segment_update_count": update_counts[key],
                    "anchor_confirmed": True,
                    "branch_discovered": sum(
                        row["branch_id"] == branch_id
                        for row in crossings
                    ) >= 4,
                    "neighbor_count": len(neighbor_ids),
                    "neighbor_motion_mean_deg": neighbor_mean,
                    "neighbor_motion_dispersion_deg": dispersion,
                    "local_heading_consensus": resultant,
                    "progress_fraction": max(
                        0.0, min(1.0, axial / branch.length)
                    ),
                    "local_axial": axial,
                    "local_lateral": lateral,
                    "information_scope": (
                        "REALIZED_SELF_MOTION_AND_LOCAL_NEIGHBOR_MOTION"
                    ),
                })
            sample_positions = positions.copy()
            previous_membership = memberships

    outgoing = case.outgoing_branches
    case_row = {
        "case_id": geometry.case_id,
        "topology": geometry.topology,
        "seed": geometry.seed,
        "branch_count": len(geometry.branches),
        "outgoing_branch_count": len(outgoing),
        "incoming_branch_id": case.incoming_branch_id,
        "incoming_branch_outward_angle_deg": (
            case.incoming_branch.angle_deg
        ),
        "incoming_travel_angle_deg": (
            case.incoming_travel_angle_deg
        ),
        "outgoing_angles_deg": str([
            branch.angle_deg for branch in outgoing
        ]),
        "turn_angles_deg": str([
            case.turn_angle_deg(branch) for branch in outgoing
        ]),
        "turn_severities": str([
            case.turn_severity(branch) for branch in outgoing
        ]),
        "branch_lengths": str([
            branch.length for branch in geometry.branches
        ]),
        "branch_widths": str([
            branch.width for branch in geometry.branches
        ]),
        "rotation_deg": geometry.rotation_deg,
        "length_group": geometry.length_group,
        "corridor_width_group": geometry.width_group,
        "central_radius": geometry.central_radius,
        "source": "GEOMETRY_GENERAL_PHYSICAL_SPH_INLET",
        "source_family": geometry.source_family,
        "valid": True,
        "invalid_reason": "",
        "particle_count": params.particle_count,
        "steps": params.steps,
        "dt": params.dt,
        "entered_junction_count": int(np.sum(entered_junction)),
        "collision_count": collision_count,
        "minimum_pair_distance": minimum_pair_distance,
        "segment_count": len(segments),
        "crossing_count": len(crossings),
    }
    return InletSphResult(
        case_row=case_row,
        segment_rows=segments,
        crossing_rows=crossings,
        trace_rows=traces,
        final_positions=positions,
        entered_junction_count=int(np.sum(entered_junction)),
        collision_count=collision_count,
    )


def _branches(
    angles: Sequence[float],
    lengths: Sequence[float],
    widths: Sequence[float],
) -> tuple[BranchGeometry, ...]:
    return tuple(
        BranchGeometry(
            f"B{index}",
            normalize_angle_deg(angle),
            float(length),
            float(width),
        )
        for index, (angle, length, width) in enumerate(zip(
            angles, lengths, widths
        ))
    )


def create_inlet_benchmark_geometries(
    seed: int = 20260817,
) -> tuple[list[InletBenchmarkGeometry], list[dict[str, Any]]]:
    """Create a balanced inlet design spanning geometry and turn severity."""
    specs: list[dict[str, Any]] = []
    canonical = (-90.0, -45.0, 20.0, 90.0, 160.0)
    for rotation in (0.0, 30.0, 60.0, 120.0):
        for seed_offset in (0, 1):
            angles = [angle + rotation for angle in canonical[:4]]
            specs.append({
                "case_id": (
                    f"inlet_rotated_r{int(rotation):03d}_s{seed_offset}"
                ),
                "topology": "4-way",
                "seed": seed + seed_offset,
                "angles": angles,
                "lengths": [105.0] * 4,
                "widths": [24.0] * 4,
                "rotation": rotation,
                "length_group": "nominal",
                "width_group": "nominal",
                "radius": 42.0,
                "family": "INLET_ROTATION",
            })
    topology_angles = {
        "3-way": (-90.0, 20.0, 90.0),
        "4-way": (-90.0, -45.0, 20.0, 90.0),
        "5-way": canonical,
    }
    for topology, angles in topology_angles.items():
        for seed_offset in (0, 1):
            specs.append({
                "case_id": f"inlet_arbitrary_{topology}_s{seed_offset}",
                "topology": topology,
                "seed": seed + 100 + seed_offset,
                "angles": list(angles),
                "lengths": [105.0] * len(angles),
                "widths": [20.0] * len(angles),
                "rotation": 0.0,
                "length_group": "nominal",
                "width_group": "nominal",
                "radius": 44.0,
                "family": "INLET_TOPOLOGY",
            })
    for index, (group, length) in enumerate((
        ("short", 70.0),
        ("nominal", 105.0),
        ("long", 165.0),
    )):
        specs.append({
            "case_id": f"inlet_length_{group}",
            "topology": "4-way",
            "seed": seed + 200 + index,
            "angles": list(canonical[:4]),
            "lengths": [105.0, length, length, length],
            "widths": [24.0] * 4,
            "rotation": 0.0,
            "length_group": group,
            "width_group": "nominal",
            "radius": 42.0,
            "family": "INLET_LENGTH",
        })
    for index, (group, width, radius) in enumerate((
        ("narrow", 16.0, 38.0),
        ("nominal", 24.0, 42.0),
        ("wide", 32.0, 50.0),
    )):
        specs.append({
            "case_id": f"inlet_width_{group}",
            "topology": "4-way",
            "seed": seed + 300 + index,
            "angles": list(canonical[:4]),
            "lengths": [105.0] * 4,
            "widths": [width] * 4,
            "rotation": 0.0,
            "length_group": "nominal",
            "width_group": group,
            "radius": radius,
            "family": "INLET_WIDTH",
        })
    specs.append({
        "case_id": "inlet_stress_long_sharp",
        "topology": "4-way",
        "seed": seed + 999,
        "angles": [-90.0, -40.0, 20.0, 90.0],
        "lengths": [126.0, 256.424384, 150.0, 150.0],
        "widths": [84.0] * 4,
        "rotation": 0.0,
        "length_group": "mixed-long-stress",
        "width_group": "production-scale",
        "radius": 115.0,
        "family": "INLET_LONG_STRESS",
    })

    cases: list[InletBenchmarkGeometry] = []
    rejected: list[dict[str, Any]] = []
    for spec in specs:
        geometry = JunctionGeometry(
            case_id=spec["case_id"],
            topology=spec["topology"],
            seed=spec["seed"],
            center=(0.0, 0.0),
            central_radius=spec["radius"],
            branches=_branches(
                spec["angles"], spec["lengths"], spec["widths"]
            ),
            rotation_deg=spec["rotation"],
            length_group=spec["length_group"],
            width_group=spec["width_group"],
            source_family=spec["family"],
        )
        valid, reason = validate_geometry(geometry)
        if valid:
            cases.append(InletBenchmarkGeometry(geometry, "B0"))
        else:
            rejected.append({
                "case_id": geometry.case_id,
                "topology": geometry.topology,
                "seed": geometry.seed,
                "valid": False,
                "invalid_reason": reason,
                "source": "GEOMETRY_GENERAL_PHYSICAL_SPH_INLET",
            })
    return cases, rejected


def run_inlet_geometry_synthetic_test() -> None:
    """Validate topology coverage, turn groups, and collision masks."""
    cases, rejected = create_inlet_benchmark_geometries(7)
    assert cases and not rejected
    assert {case.geometry.topology for case in cases} >= {
        "3-way", "4-way", "5-way"
    }
    severities = {
        case.turn_severity(branch)
        for case in cases
        for branch in case.outgoing_branches
    }
    assert severities >= {"shallow", "medium", "sharp"}
    assert all(
        collision_mask_sanity(case.geometry)["pass"]
        for case in cases
    )


if __name__ == "__main__":
    run_inlet_geometry_synthetic_test()
    case = create_inlet_benchmark_geometries(7)[0][0]
    result = simulate_inlet_sph_case(
        case,
        SphParameters(steps=700, particle_count=64),
    )
    if result.entered_junction_count <= 0:
        raise AssertionError("inlet smoke run never entered the Junction")
    print("general_junction_sph_inlet_benchmark tests: PASS")
