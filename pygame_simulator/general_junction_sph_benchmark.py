"""Geometry-general physical SPH trajectory benchmark.

This is a diagnostics-only physical particle simulator.  It reuses the
production prototype's compact spiky kernel, compressive pressure, local
velocity-consensus viscosity, finite robot radius, and collision response
principles, while replacing the hard-coded LEFT/UP/RIGHT rectangles with a
data-driven union of a central disk and oriented corridor rectangles.

The benchmark intentionally stops before Guard/Shepherd/DFS/Handoff logic.
Robots receive only an Anchor-relative radial exploration drive; Branch
tangents are used by wall construction and evaluation labels, never by robot
control or trajectory sample selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


EPSILON = 1.0e-12


def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _unit(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.asarray([math.cos(radians), math.sin(radians)], dtype=float)


@dataclass(frozen=True)
class BranchGeometry:
    """One oriented corridor attached to a shared central region."""

    branch_id: str
    angle_deg: float
    length: float
    width: float

    @property
    def tangent(self) -> np.ndarray:
        return _unit(self.angle_deg)

    @property
    def normal(self) -> np.ndarray:
        tangent = self.tangent
        return np.asarray([-tangent[1], tangent[0]], dtype=float)


@dataclass(frozen=True)
class JunctionGeometry:
    """Union of a central disk and arbitrary oriented Branch corridors."""

    case_id: str
    topology: str
    seed: int
    center: tuple[float, float]
    central_radius: float
    branches: tuple[BranchGeometry, ...]
    rotation_deg: float
    length_group: str
    width_group: str
    source_family: str

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Return membership in the walkable union with robot-centre margin."""
        array = np.atleast_2d(np.asarray(points, dtype=float))
        relative = array - self.center_array
        central_limit = max(self.central_radius - margin, 0.0)
        allowed = np.sum(relative * relative, axis=1) <= central_limit**2
        overlap = self.central_radius * 0.32
        for branch in self.branches:
            axial = relative @ branch.tangent
            lateral = relative @ branch.normal
            allowed |= (
                (axial >= self.central_radius - overlap)
                & (axial <= self.central_radius + branch.length - margin)
                & (np.abs(lateral) <= max(0.0, branch.width * 0.5 - margin))
            )
        return allowed

    def project_point(self, point: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
        """Project an outside robot centre to the nearest union component.

        Returns the corrected position and an approximate outward wall normal.
        The nearest-component projection makes collision handling independent
        of Branch labels and world axes.
        """
        point = np.asarray(point, dtype=float)
        if bool(self.contains(point, margin)[0]):
            return point.copy(), np.zeros(2, dtype=float)
        center = self.center_array
        relative = point - center
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        radius = max(self.central_radius - margin, EPSILON)
        distance = float(np.linalg.norm(relative))
        radial = relative / max(distance, EPSILON)
        disk_point = center + radial * radius
        candidates.append((float(np.linalg.norm(point - disk_point)), disk_point, radial))

        overlap = self.central_radius * 0.32
        for branch in self.branches:
            tangent, normal = branch.tangent, branch.normal
            axial = float(relative @ tangent)
            lateral = float(relative @ normal)
            minimum_axial = self.central_radius - overlap
            maximum_axial = self.central_radius + branch.length - margin
            half_width = max(branch.width * 0.5 - margin, EPSILON)
            clipped_axial = min(max(axial, minimum_axial), maximum_axial)
            clipped_lateral = min(max(lateral, -half_width), half_width)
            candidate = center + tangent * clipped_axial + normal * clipped_lateral
            delta = point - candidate
            delta_length = float(np.linalg.norm(delta))
            wall_normal = delta / max(delta_length, EPSILON)
            candidates.append((delta_length, candidate, wall_normal))
        _, corrected, wall_normal = min(candidates, key=lambda item: item[0])
        return corrected, wall_normal

    def evaluation_branch(self, point: np.ndarray, margin: float = 0.0) -> tuple[BranchGeometry | None, float, float]:
        """Return GT membership and local coordinates for evaluation only."""
        relative = np.asarray(point, dtype=float) - self.center_array
        candidates = []
        for branch in self.branches:
            axial_from_center = float(relative @ branch.tangent)
            axial = axial_from_center - self.central_radius
            lateral = float(relative @ branch.normal)
            if (
                axial >= -self.central_radius * 0.20
                and axial <= branch.length
                and abs(lateral) <= branch.width * 0.5 + margin
            ):
                candidates.append((axial, -abs(lateral), branch, lateral))
        if not candidates:
            return None, float("nan"), float("nan")
        axial, _, branch, lateral = max(candidates, key=lambda item: (item[0], item[1]))
        return branch, axial, lateral


@dataclass(frozen=True)
class SphParameters:
    """Dimensionally consistent physical parameters for the benchmark."""

    robot_radius: float = 1.35
    smoothing_length: float = 8.0
    pressure_gain: float = 520.0
    rest_density_scale: float = 0.78
    viscosity_gain: float = 5.5
    overlap_gain: float = 180.0
    drive_speed: float = 16.0
    drive_gain: float = 2.8
    linear_drag: float = 0.45
    maximum_speed: float = 25.0
    dt: float = 0.0125
    steps: int = 760
    sample_every: int = 8
    neighbor_radius: float = 12.0
    particle_count: int = 112


@dataclass
class SphCaseResult:
    """Raw physical observations from one completed simulation."""

    case_row: dict[str, Any]
    segment_rows: list[dict[str, Any]] = field(default_factory=list)
    crossing_rows: list[dict[str, Any]] = field(default_factory=list)
    final_positions: np.ndarray | None = None
    collision_count: int = 0
    minimum_pair_distance: float = float("inf")


def validate_geometry(geometry: JunctionGeometry, robot_radius: float = 1.35) -> tuple[bool, str]:
    """Reject overlapping mouths and corridors too narrow for robot centres."""
    if len(geometry.branches) < 3:
        return False, "TOPOLOGY_REQUIRES_AT_LEAST_THREE_BRANCHES"
    for branch in geometry.branches:
        if branch.length <= robot_radius * 8.0:
            return False, f"{branch.branch_id}:BRANCH_TOO_SHORT"
        if branch.width <= robot_radius * 4.0:
            return False, f"{branch.branch_id}:CORRIDOR_TOO_NARROW"
    ordered = sorted(normalize_angle_deg(branch.angle_deg) for branch in geometry.branches)
    gaps = [
        (ordered[(index + 1) % len(ordered)] - ordered[index]) % 360.0
        for index in range(len(ordered))
    ]
    maximum_width = max(branch.width for branch in geometry.branches)
    required_gap = math.degrees(2.0 * math.asin(min(0.95, (maximum_width * 0.5 + robot_radius * 2.0) / geometry.central_radius)))
    if min(gaps) < required_gap:
        return False, f"MOUTH_OVERLAP:min_gap={min(gaps):.3f}<required={required_gap:.3f}"
    return True, "VALID"


def collision_mask_sanity(geometry: JunctionGeometry, robot_radius: float = 1.35) -> dict[str, Any]:
    """Check centre connectivity, Branch centre-lines, walls, and end caps."""
    center = geometry.center_array
    checks = [bool(geometry.contains(center, robot_radius)[0])]
    outside_checks = []
    for branch in geometry.branches:
        for fraction in np.linspace(0.0, 0.95, 12):
            point = center + branch.tangent * (geometry.central_radius + branch.length * fraction)
            checks.append(bool(geometry.contains(point, robot_radius)[0]))
        wall_point = center + branch.tangent * (geometry.central_radius + branch.length * 0.5) + branch.normal * (branch.width * 0.5 + robot_radius * 1.2)
        outside_checks.append(not bool(geometry.contains(wall_point, robot_radius)[0]))
        end_point = center + branch.tangent * (geometry.central_radius + branch.length + robot_radius * 1.2)
        outside_checks.append(not bool(geometry.contains(end_point, robot_radius)[0]))
    return {
        "inside_checks": len(checks),
        "outside_checks": len(outside_checks),
        "inside_pass": all(checks),
        "outside_pass": all(outside_checks),
        "pass": all(checks) and all(outside_checks),
    }


def _spiky_kernel(distances: np.ndarray, smoothing_length: float) -> np.ndarray:
    """Vectorized form of production's 2-D compact spiky kernel."""
    q = np.maximum(0.0, 1.0 - distances / smoothing_length)
    return 10.0 / (math.pi * smoothing_length**2) * q**3


def _initial_particles(geometry: JunctionGeometry, parameters: SphParameters) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible compressed Anchor-centred swarm.

    Initial headings are radial with seeded perturbations; they are rotated
    with the geometry but never derived from any Branch tangent.
    """
    rng = np.random.default_rng(geometry.seed)
    spacing = parameters.robot_radius * 2.25
    extent = geometry.central_radius * 0.72
    coordinates = []
    y = -extent
    while y <= extent:
        x_offset = 0.5 * spacing if int(round((y + extent) / spacing)) % 2 else 0.0
        x = -extent + x_offset
        while x <= extent:
            if x * x + y * y <= extent**2:
                coordinates.append((x, y))
            x += spacing
        y += spacing * math.sqrt(3.0) * 0.5
    if len(coordinates) < parameters.particle_count:
        raise RuntimeError("central region cannot hold requested benchmark particles")
    selected = np.sort(rng.choice(
        len(coordinates),
        size=parameters.particle_count,
        replace=False,
    ))
    positions = np.asarray([coordinates[index] for index in selected], dtype=float)
    rotation = math.radians(geometry.rotation_deg)
    rotation_matrix = np.asarray([[math.cos(rotation), -math.sin(rotation)], [math.sin(rotation), math.cos(rotation)]])
    positions = positions @ rotation_matrix.T + geometry.center_array
    relative = positions - geometry.center_array
    radial_norm = np.linalg.norm(relative, axis=1, keepdims=True)
    radial = relative / np.maximum(radial_norm, EPSILON)
    jitter = rng.normal(0.0, 0.055, size=positions.shape[0])
    cos_jitter, sin_jitter = np.cos(jitter), np.sin(jitter)
    directions = np.column_stack((
        radial[:, 0] * cos_jitter - radial[:, 1] * sin_jitter,
        radial[:, 0] * sin_jitter + radial[:, 1] * cos_jitter,
    ))
    velocities = directions * parameters.drive_speed * 0.20
    return positions, velocities


def simulate_sph_case(
    geometry: JunctionGeometry,
    parameters: SphParameters | None = None,
) -> SphCaseResult:
    """Run one geometry through physical SPH and collect real trajectories."""
    params = parameters or SphParameters()
    valid, reason = validate_geometry(geometry, params.robot_radius)
    if not valid:
        raise ValueError(f"invalid geometry {geometry.case_id}: {reason}")
    sanity = collision_mask_sanity(geometry, params.robot_radius)
    if not sanity["pass"]:
        raise AssertionError(f"collision mask sanity failed for {geometry.case_id}: {sanity}")

    positions, velocities = _initial_particles(geometry, params)
    sample_positions = positions.copy()
    previous_positions = positions.copy()
    previous_membership: list[str | None] = [None] * len(positions)
    update_counts: dict[tuple[str, int], int] = {}
    latest_motion_angles: dict[int, float] = {}
    crossing_seen: set[tuple[str, int]] = set()
    segment_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    collision_count = 0
    minimum_pair_distance = float("inf")
    reference_density: float | None = None

    for step in range(params.steps):
        differences = positions[:, None, :] - positions[None, :, :]
        distance_sq = np.sum(differences * differences, axis=2)
        distances = np.sqrt(np.maximum(distance_sq, EPSILON))
        np.fill_diagonal(distances, 0.0)
        pair_distances = distances[np.triu_indices(len(positions), 1)]
        minimum_pair_distance = min(minimum_pair_distance, float(np.min(pair_distances)))
        kernel = _spiky_kernel(distances, params.smoothing_length)
        kernel[distances > params.smoothing_length] = 0.0
        density = np.sum(kernel, axis=1)
        if reference_density is None:
            reference_density = max(float(np.median(density)) * params.rest_density_scale, EPSILON)
        ratio = density / reference_density
        pressure = params.pressure_gain * density * np.maximum(ratio * ratio - 1.0, 0.0)

        q = np.maximum(0.0, 1.0 - distances / params.smoothing_length)
        gradient_magnitude = -30.0 / (math.pi * params.smoothing_length**3) * q**2
        gradient_magnitude[(distances <= EPSILON) | (distances > params.smoothing_length)] = 0.0
        directions = differences / np.maximum(distances[:, :, None], EPSILON)
        gradients = directions * gradient_magnitude[:, :, None]
        coefficient = pressure[:, None] / np.maximum(density[:, None] ** 2, EPSILON) + pressure[None, :] / np.maximum(density[None, :] ** 2, EPSILON)
        pressure_force = np.sum(-coefficient[:, :, None] * gradients, axis=1)
        normalized_kernel = kernel / max(float(_spiky_kernel(np.asarray([0.0]), params.smoothing_length)[0]), EPSILON)
        viscosity_force = params.viscosity_gain * np.sum((velocities[None, :, :] - velocities[:, None, :]) * normalized_kernel[:, :, None], axis=1)

        overlap = np.maximum(0.0, params.robot_radius * 2.0 - distances)
        overlap[(distances <= EPSILON) | (distances >= params.robot_radius * 2.0)] = 0.0
        overlap_force = params.overlap_gain * np.sum(directions * overlap[:, :, None], axis=1)

        relative = positions - geometry.center_array
        radial_norm = np.linalg.norm(relative, axis=1, keepdims=True)
        radial = relative / np.maximum(radial_norm, EPSILON)
        # At the exact Anchor, preserve the particle's own realized heading.
        velocity_norm = np.linalg.norm(velocities, axis=1, keepdims=True)
        fallback = velocities / np.maximum(velocity_norm, EPSILON)
        desired_direction = np.where(radial_norm > params.robot_radius, radial, fallback)
        drive_force = params.drive_gain * (desired_direction * params.drive_speed - velocities)
        acceleration = pressure_force + viscosity_force + overlap_force + drive_force - params.linear_drag * velocities
        velocities += acceleration * params.dt
        speeds = np.linalg.norm(velocities, axis=1)
        over_speed = speeds > params.maximum_speed
        velocities[over_speed] *= (params.maximum_speed / speeds[over_speed])[:, None]
        proposed = positions + velocities * params.dt
        for robot_id in range(len(proposed)):
            if not bool(geometry.contains(proposed[robot_id], params.robot_radius)[0]):
                collision_count += 1
                corrected, wall_normal = geometry.project_point(proposed[robot_id], params.robot_radius)
                proposed[robot_id] = corrected
                normal_speed = float(velocities[robot_id] @ wall_normal)
                if normal_speed > 0.0:
                    velocities[robot_id] -= wall_normal * normal_speed * 1.15
                velocities[robot_id] *= 0.82
        previous_positions[:] = positions
        positions = proposed

        current_membership: list[str | None] = []
        for robot_id, point in enumerate(positions):
            branch, axial, lateral = geometry.evaluation_branch(point, params.robot_radius)
            branch_id = None if branch is None else branch.branch_id
            current_membership.append(branch_id)
            if branch is not None:
                # Production infers mouth samples from Anchor-relative swarm
                # expansion, not from a known geometric mouth plane.  Retain
                # the last fixed-rate local observation when the robot first
                # joins an evaluation Branch.  This is deliberately not
                # projected onto the GT mouth plane.
                key = (branch.branch_id, robot_id)
                if key not in crossing_seen:
                    crossing = sample_positions[robot_id].copy()
                    crossing_seen.add(key)
                    crossing_rows.append({
                        "case_id": geometry.case_id,
                        "branch_id": branch.branch_id,
                        "robot_id": robot_id,
                        "frame": step,
                        "time": step * params.dt,
                        "crossing_x": float(crossing[0]),
                        "crossing_y": float(crossing[1]),
                        "outward_dx": float(positions[robot_id, 0] - previous_positions[robot_id, 0]),
                        "outward_dy": float(positions[robot_id, 1] - previous_positions[robot_id, 1]),
                    })

        if step > 0 and step % params.sample_every == 0:
            sample_displacements = positions - sample_positions
            sample_lengths = np.linalg.norm(sample_displacements, axis=1)
            sample_angles = np.degrees(np.arctan2(sample_displacements[:, 1], sample_displacements[:, 0]))
            for robot_id, branch_id in enumerate(current_membership):
                if branch_id is None or previous_membership[robot_id] not in {None, branch_id}:
                    continue
                branch = next(item for item in geometry.branches if item.branch_id == branch_id)
                _, axial, lateral = geometry.evaluation_branch(positions[robot_id], params.robot_radius)
                if sample_lengths[robot_id] <= params.robot_radius * 0.08 or axial >= branch.length * 0.96:
                    continue
                neighbor_delta = positions - positions[robot_id]
                neighbor_distance = np.linalg.norm(neighbor_delta, axis=1)
                neighbor_ids = [
                    other_id for other_id, other_branch in enumerate(current_membership)
                    if other_id != robot_id
                    and other_branch == branch_id
                    and neighbor_distance[other_id] <= params.neighbor_radius
                    and other_id in latest_motion_angles
                ]
                neighbor_angles = [latest_motion_angles[other_id] for other_id in neighbor_ids]
                if neighbor_angles:
                    radians = np.radians(neighbor_angles)
                    mean_cos, mean_sin = float(np.mean(np.cos(radians))), float(np.mean(np.sin(radians)))
                    neighbor_mean = normalize_angle_deg(math.degrees(math.atan2(mean_sin, mean_cos)))
                    neighbor_resultant = min(1.0, math.hypot(mean_cos, mean_sin))
                    neighbor_dispersion = 180.0 if neighbor_resultant <= EPSILON else math.degrees(math.sqrt(max(0.0, -2.0 * math.log(neighbor_resultant))))
                else:
                    neighbor_mean, neighbor_resultant, neighbor_dispersion = float("nan"), 0.0, float("nan")
                key = (branch_id, robot_id)
                update_counts[key] = update_counts.get(key, 0) + 1
                angle = normalize_angle_deg(float(sample_angles[robot_id]))
                latest_motion_angles[robot_id] = angle
                segment_rows.append({
                    "environment_id": geometry.case_id,
                    "case_id": geometry.case_id,
                    "seed": geometry.seed,
                    "branch_id": branch_id,
                    "gt_branch_angle_deg": branch.angle_deg,
                    "robot_id": robot_id,
                    "frame": step,
                    "time": step * params.dt,
                    "previous_x": float(sample_positions[robot_id, 0]),
                    "previous_y": float(sample_positions[robot_id, 1]),
                    "current_x": float(positions[robot_id, 0]),
                    "current_y": float(positions[robot_id, 1]),
                    "dx": float(sample_displacements[robot_id, 0]),
                    "dy": float(sample_displacements[robot_id, 1]),
                    "displacement_length": float(sample_lengths[robot_id]),
                    "motion_angle_deg": angle,
                    "observed_speed": float(sample_lengths[robot_id] / (params.sample_every * params.dt)),
                    "segment_update_count": update_counts[key],
                    "lifecycle_phase": "JUNCTION_TURNING" if axial <= branch.width else "BRANCH_FLOW",
                    "anchor_confirmed": True,
                    "branch_discovered": len([row for row in crossing_rows if row["branch_id"] == branch_id]) >= 4,
                    "neighbor_count": len(neighbor_ids),
                    "neighbor_motion_mean_deg": neighbor_mean,
                    "neighbor_motion_dispersion_deg": neighbor_dispersion,
                    "local_heading_consensus": neighbor_resultant,
                    "progress_fraction": max(0.0, min(1.0, axial / branch.length)),
                    "local_axial": axial,
                    "local_lateral": lateral,
                    "information_scope": "REALIZED_SELF_MOTION_AND_LOCAL_NEIGHBOR_MOTION",
                })
            sample_positions = positions.copy()
            previous_membership = current_membership

    case_row = {
        "case_id": geometry.case_id,
        "topology": geometry.topology,
        "seed": geometry.seed,
        "branch_count": len(geometry.branches),
        "branch_angles_deg": str([branch.angle_deg for branch in geometry.branches]),
        "branch_lengths": str([branch.length for branch in geometry.branches]),
        "branch_widths": str([branch.width for branch in geometry.branches]),
        "rotation_deg": geometry.rotation_deg,
        "length_group": geometry.length_group,
        "corridor_width_group": geometry.width_group,
        "central_radius": geometry.central_radius,
        "source": "GEOMETRY_GENERAL_PHYSICAL_SPH",
        "source_family": geometry.source_family,
        "valid": True,
        "invalid_reason": "",
        "particle_count": params.particle_count,
        "steps": params.steps,
        "dt": params.dt,
        "collision_count": collision_count,
        "minimum_pair_distance": minimum_pair_distance,
        "segment_count": len(segment_rows),
        "crossing_count": len(crossing_rows),
    }
    return SphCaseResult(
        case_row=case_row,
        segment_rows=segment_rows,
        crossing_rows=crossing_rows,
        final_positions=positions,
        collision_count=collision_count,
        minimum_pair_distance=minimum_pair_distance,
    )


def _branches(angles: Sequence[float], lengths: Sequence[float], widths: Sequence[float]) -> tuple[BranchGeometry, ...]:
    return tuple(
        BranchGeometry(f"B{index}", normalize_angle_deg(angle), float(length), float(width))
        for index, (angle, length, width) in enumerate(zip(angles, lengths, widths))
    )


def create_physical_benchmark_geometries(seed: int = 20260817) -> tuple[list[JunctionGeometry], list[dict[str, Any]]]:
    """Create valid physical cases and retain rejected specifications/reasons."""
    specifications: list[dict[str, Any]] = []
    cross = (0.0, 90.0, 180.0, -90.0)
    for rotation in (0.0, 30.0, 60.0, 120.0):
        for seed_offset in (0, 1):
            specifications.append({
                "case_id": f"rotated_cross_r{int(rotation):03d}_s{seed_offset}",
                "topology": "4-way", "seed": seed + seed_offset,
                "angles": [angle + rotation for angle in cross],
                "lengths": [110.0] * 4, "widths": [24.0] * 4,
                "rotation": rotation, "length_group": "nominal",
                "width_group": "nominal", "radius": 38.0,
                "family": "ROTATION_INVARIANCE",
            })
    topology_specs = {
        "3-way": (-120.0, 0.0, 125.0),
        "4-way": (-145.0, -35.0, 55.0, 155.0),
        "5-way": (-150.0, -75.0, 0.0, 70.0, 145.0),
    }
    for topology, angles in topology_specs.items():
        for seed_offset in (0, 1):
            specifications.append({
                "case_id": f"arbitrary_{topology}_s{seed_offset}",
                "topology": topology, "seed": seed + 100 + seed_offset,
                "angles": list(angles), "lengths": [110.0] * len(angles),
                "widths": [20.0] * len(angles), "rotation": 0.0,
                "length_group": "nominal", "width_group": "nominal",
                "radius": 42.0, "family": "ARBITRARY_TOPOLOGY",
            })
    for index, (group, length) in enumerate((
        ("short", 70.0),
        ("nominal", 110.0),
        ("long", 165.0),
    )):
        specifications.append({
            "case_id": f"length_{group}", "topology": "4-way",
            "seed": seed + 200 + index, "angles": list(cross),
            "lengths": [length] * 4, "widths": [24.0] * 4,
            "rotation": 0.0, "length_group": group,
            "width_group": "nominal", "radius": 38.0,
            "family": "LENGTH_SWEEP",
        })
    for index, (group, width, radius) in enumerate((
        ("narrow", 16.0, 34.0),
        ("nominal", 24.0, 38.0),
        ("wide", 32.0, 46.0),
    )):
        specifications.append({
            "case_id": f"width_{group}", "topology": "4-way",
            "seed": seed + 300 + index, "angles": list(cross),
            "lengths": [110.0] * 4, "widths": [width] * 4,
            "rotation": 0.0, "length_group": "nominal",
            "width_group": group, "radius": radius,
            "family": "WIDTH_SWEEP",
        })
    specifications.append({
        "case_id": "stress_case_right_long", "topology": "4-way",
        "seed": seed + 999, "angles": [0.0, 90.0, 180.0, -90.0],
        "lengths": [256.424384, 126.0, 126.0, 126.0],
        "widths": [84.0] * 4, "rotation": 0.0,
        "length_group": "mixed-right-long", "width_group": "production-scale",
        "radius": 68.0, "family": "RIGHT_STRESS_ABSTRACTION",
    })

    geometries: list[JunctionGeometry] = []
    rejected: list[dict[str, Any]] = []
    for spec in specifications:
        geometry = JunctionGeometry(
            case_id=spec["case_id"], topology=spec["topology"], seed=spec["seed"],
            center=(0.0, 0.0), central_radius=spec["radius"],
            branches=_branches(spec["angles"], spec["lengths"], spec["widths"]),
            rotation_deg=spec["rotation"], length_group=spec["length_group"],
            width_group=spec["width_group"], source_family=spec["family"],
        )
        valid, reason = validate_geometry(geometry)
        if valid:
            geometries.append(geometry)
        else:
            rejected.append({
                "case_id": geometry.case_id, "topology": geometry.topology,
                "seed": geometry.seed, "branch_count": len(geometry.branches),
                "branch_angles_deg": str([branch.angle_deg for branch in geometry.branches]),
                "branch_lengths": str([branch.length for branch in geometry.branches]),
                "branch_widths": str([branch.width for branch in geometry.branches]),
                "rotation_deg": geometry.rotation_deg,
                "length_group": geometry.length_group,
                "corridor_width_group": geometry.width_group,
                "central_radius": geometry.central_radius,
                "source": "GEOMETRY_GENERAL_PHYSICAL_SPH",
                "source_family": geometry.source_family,
                "valid": False, "invalid_reason": reason,
            })
    return geometries, rejected


def run_geometry_synthetic_test() -> None:
    """Validate arbitrary geometry rejection and analytical collision masks."""
    geometries, rejected = create_physical_benchmark_geometries(7)
    assert geometries
    assert {geometry.topology for geometry in geometries} >= {"3-way", "4-way", "5-way"}
    assert all(collision_mask_sanity(geometry)["pass"] for geometry in geometries)
    invalid = JunctionGeometry(
        case_id="invalid", topology="3-way", seed=0, center=(0.0, 0.0),
        central_radius=20.0,
        branches=_branches((0.0, 5.0, 180.0), (50.0, 50.0, 50.0), (18.0, 18.0, 18.0)),
        rotation_deg=0.0, length_group="short", width_group="nominal",
        source_family="SYNTHETIC_INVALID",
    )
    valid, reason = validate_geometry(invalid)
    assert not valid and reason.startswith("MOUTH_OVERLAP")
    assert rejected == []


if __name__ == "__main__":
    run_geometry_synthetic_test()
    geometry = create_physical_benchmark_geometries(7)[0][0]
    result = simulate_sph_case(geometry, SphParameters(steps=80, particle_count=48))
    if not result.segment_rows:
        raise AssertionError("SPH smoke run produced no physical trajectory segments")
    print("general_junction_sph_benchmark synthetic and smoke tests: PASS")
