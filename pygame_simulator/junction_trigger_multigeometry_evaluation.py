"""EVALUATION-ONLY multi-geometry Junction-suspicion benchmark.

NOT PRODUCTION. NO DFS. NO GLOBAL MAP INFORMATION IS PROVIDED TO THE DETECTOR.
This compact harness models deterministic corridor motion, SPH-scale robot
repulsion, and geometry-derived wall interaction. GT is evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "junction_detection/integration/output/junction_trigger_multigeometry_corrected"
CASE_IDS = ("M0_STRAIGHT", "M1_CROSS_BASELINE", "M2_T_JUNCTION", "M3_ANGLED_Y", "M4_ASYMMETRIC_CROSS", "M5_UNEQUAL_WIDTH")

# Fixed across all maps. These retain the prior diagnostic scales, but this is
# minimal swarm dynamics rather than the production pressure/viscosity solver.
SUPPORT = 22.0 * 0.70
ROBOT_RADIUS = 1.60 * 0.70
DT = 0.05
SAMPLE_INTERVAL = 0.10
FRONT_QUANTILE = 0.68
MIN_SPEED = 1.2
LATERAL_MIN_DELTA = (4.5 * 0.70) ** 2
LATERAL_RATIO = 1.28
BASELINE_ALPHA = 0.035
GAP_THRESHOLD_DEG = 120.0
FORWARD_FORCE, NEIGHBOR_FORCE = 20.0, 0.22
WALL_FORCE, WALL_INFLUENCE, MAX_SPEED = 3.0, 12.0, 18.0

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class Branch:
    """Geometry/GT-only outgoing corridor description."""
    angle_deg: float
    width: float
    length: float


@dataclass(frozen=True)
class OrientedRect:
    """Convex free-space primitive represented by four vertices."""
    vertices: tuple[Point, Point, Point, Point]

    def contains(self, point: np.ndarray, tolerance: float = 1e-8) -> bool:
        vertices = [np.asarray(value) for value in self.vertices]
        crosses = []
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            edge = end - start
            crosses.append(float(edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])))
        return min(crosses) >= -tolerance or max(crosses) <= tolerance


@dataclass(frozen=True)
class GeometryCase:
    """Single geometry source for physics, drawing, LiDAR walls, and GT."""
    case_id: str
    incoming_width: float
    incoming_length: float
    junction_width: float
    junction_height: float
    branches: tuple[Branch, ...]
    free_space_rects: tuple[OrientedRect, ...]
    wall_segments: tuple[Segment, ...]
    entrance_progress: float | None
    approach_distance: float

    def contains(self, point: np.ndarray) -> bool:
        return any(rect.contains(point) for rect in self.free_space_rects)


def _oriented_rect(center: np.ndarray, direction: np.ndarray, width: float, length: float) -> OrientedRect:
    direction = direction / np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    start, end = center - direction * length / 2, center + direction * length / 2
    return OrientedRect(tuple(tuple(value) for value in (
        start - lateral * width / 2, end - lateral * width / 2,
        end + lateral * width / 2, start + lateral * width / 2,
    )))


def _intersection_t(a, b, c, d):
    r, s = b - a, d - c
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) < 1e-10:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / cross)
    u = float((q[0] * r[1] - q[1] * r[0]) / cross)
    return min(1.0, max(0.0, t)) if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9 else None


def _external_boundary(rects: tuple[OrientedRect, ...]) -> tuple[Segment, ...]:
    """Extract an oriented-rectangle union boundary without extra packages.

    Primitive edges are split at intersections. A piece remains only when
    small probes on its two sides disagree about free-space membership, which
    removes internal corridor/Junction entrance caps.
    """
    edges = []
    for rect in rects:
        vertices = [np.asarray(value) for value in rect.vertices]
        edges.extend(zip(vertices, vertices[1:] + vertices[:1]))
    inside = lambda point: any(rect.contains(point, 1e-7) for rect in rects)
    pieces = []
    for a, b in edges:
        cuts = [0.0, 1.0]
        for c, d in edges:
            value = _intersection_t(a, b, c, d)
            if value is not None:
                cuts.append(value)
        cuts = sorted(set(round(value, 10) for value in cuts))
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / max(np.linalg.norm(edge), 1e-12)
        for t0, t1 in zip(cuts, cuts[1:]):
            if t1 - t0 < 1e-8:
                continue
            start, end = a + edge * t0, a + edge * t1
            mid = (start + end) / 2
            if inside(mid + normal * 1e-4) != inside(mid - normal * 1e-4):
                pieces.append((tuple(start), tuple(end)))
    unique = {}
    for start, end in pieces:
        key = tuple(sorted((tuple(round(v, 7) for v in start), tuple(round(v, 7) for v in end))))
        unique[key] = (start, end)
    return tuple(unique.values())


def _direction(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.array([math.sin(radians), math.cos(radians)])


def make_geometry(case_id: str) -> GeometryCase:
    """Construct connected free space in the incoming-forward (+y) frame."""
    width, incoming_length, junction_size, branch_length = 84.0, 180.0, 84.0, 126.0
    definitions = {
        "M0_STRAIGHT": (),
        "M1_CROSS_BASELINE": (Branch(0, width, branch_length), Branch(-90, width, branch_length), Branch(90, width, branch_length)),
        "M2_T_JUNCTION": (Branch(-90, width, branch_length), Branch(90, width, branch_length)),
        "M3_ANGLED_Y": (Branch(-60, width, branch_length), Branch(60, width, branch_length)),
        "M4_ASYMMETRIC_CROSS": (Branch(0, width, branch_length), Branch(-90, width * .75, branch_length), Branch(90, width, branch_length)),
        "M5_UNEQUAL_WIDTH": (Branch(0, width * .65, branch_length), Branch(-90, width, branch_length), Branch(90, width * 1.35, branch_length)),
    }
    if case_id not in definitions:
        raise ValueError(f"unknown map case: {case_id}")
    if case_id == "M0_STRAIGHT":
        # Same corridor and start, with no reachable cap in 600 frames.
        rects, entrance = (_oriented_rect(np.array([0., 175.]), np.array([0., 1.]), width, 710.),), None
    else:
        entrance = -junction_size / 2
        rect_list = [
            _oriented_rect(np.array([0., entrance - incoming_length / 2]), np.array([0., 1.]), width, incoming_length),
            _oriented_rect(np.zeros(2), np.array([0., 1.]), junction_size, junction_size),
        ]
        for branch in definitions[case_id]:
            direction = _direction(branch.angle_deg)
            start = junction_size / 2 - 2.0  # overlap removes internal cap
            rect_list.append(_oriented_rect(direction * (start + branch.length / 2), direction, branch.width, branch.length + 4))
        rects = tuple(rect_list)
    return GeometryCase(case_id, width, incoming_length, junction_size, junction_size,
                        definitions[case_id], rects, _external_boundary(rects), entrance, width / 2.0)


@dataclass
class Robot:
    robot_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    observed_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


def initial_swarm(count: int = 48) -> list[Robot]:
    """Return the identical deterministic formation for every map."""
    return [Robot(robot_id, np.array([(robot_id % 8 - 3.5) * 7., -155. + (robot_id // 8) * 7.])) for robot_id in range(count)]


def _nearest_point(point: np.ndarray, segment: Segment):
    a, b = np.asarray(segment[0]), np.asarray(segment[1])
    edge = b - a
    t = float(np.clip(np.dot(point - a, edge) / max(np.dot(edge, edge), 1e-12), 0, 1))
    nearest = a + edge * t
    return nearest, float(np.linalg.norm(point - nearest))


def _inward_normal(geometry, nearest, segment):
    a, b = np.asarray(segment[0]), np.asarray(segment[1])
    edge = b - a
    normal = np.array([-edge[1], edge[0]]) / max(np.linalg.norm(edge), 1e-12)
    return normal if geometry.contains(nearest + normal * 1e-3) else -normal


def integrate_step(robots: list[Robot], geometry: GeometryCase) -> tuple[int, int]:
    """Advance one fixed step without consulting branch metadata or map ID."""
    positions = np.array([robot.position for robot in robots])
    penetration_events = 0
    for index, robot in enumerate(robots):
        force = np.array([0., FORWARD_FORCE])
        offsets = positions[index] - positions
        distances = np.linalg.norm(offsets, axis=1)
        for peer in np.where((distances > 1e-9) & (distances < SUPPORT))[0]:
            force += offsets[peer] / distances[peer] * (SUPPORT - distances[peer]) * NEIGHBOR_FORCE
        choices = [(*_nearest_point(robot.position, wall), wall) for wall in geometry.wall_segments]
        nearest, distance, wall = min(choices, key=lambda item: item[1])
        if distance < WALL_INFLUENCE:
            force += _inward_normal(geometry, nearest, wall) * (WALL_INFLUENCE - distance) * WALL_FORCE
        robot.velocity = .90 * robot.velocity + force * DT
        speed = float(np.linalg.norm(robot.velocity))
        if speed > MAX_SPEED:
            robot.velocity *= MAX_SPEED / speed
        proposed = robot.position + robot.velocity * DT
        if not geometry.contains(proposed):
            penetration_events += 1
            choices = [(*_nearest_point(proposed, candidate), candidate) for candidate in geometry.wall_segments]
            nearest, _, wall = min(choices, key=lambda item: item[1])
            inward = _inward_normal(geometry, nearest, wall)
            proposed = nearest + inward * (ROBOT_RADIUS + 1e-4)
            outward_speed = float(np.dot(robot.velocity, -inward))
            if outward_speed > 0:
                robot.velocity += inward * outward_speed
        robot.position, robot.observed_velocity = proposed, robot.velocity.copy()
    return sum(not geometry.contains(robot.position) for robot in robots), penetration_events


class Detector:
    """Map-independent detector receiving robot states only."""
    def __init__(self):
        self.previous_forward = None
        self.previous_variance = None
        self.baseline = self.dwell = 0.0
        self.previous_boundary_ids = set()
        self.previous_component_count = 0
        self.previous_boundary_centroid = None

    def sample(self, robots: list[Robot], timestamp: float) -> dict | None:
        moving = [robot for robot in robots if np.linalg.norm(robot.observed_velocity) >= MIN_SPEED]
        if not moving:
            return None
        forward = np.sum([robot.observed_velocity for robot in moving], axis=0)
        if np.linalg.norm(forward) <= 1e-12:
            return None
        forward /= np.linalg.norm(forward)
        if self.previous_forward is not None and np.dot(forward, self.previous_forward) < 0:
            forward = -forward
        self.previous_forward = forward.copy()
        center = np.mean([robot.position for robot in robots], axis=0)
        projections = np.array([np.dot(robot.position - center, forward) for robot in robots])
        front = [robot for robot, projection in zip(robots, projections) if projection >= np.quantile(projections, FRONT_QUANTILE)]
        front_center = np.mean([robot.position for robot in front], axis=0)
        lateral_axis = np.array([-forward[1], forward[0]])
        lateral = np.array([np.dot(robot.position - front_center, lateral_axis) for robot in front])
        longitudinal = np.array([np.dot(robot.position - front_center, forward) for robot in front])
        front_lateral_velocity = np.array([np.dot(robot.observed_velocity, lateral_axis) for robot in front])
        all_lateral = np.array([np.dot(robot.position - center, lateral_axis) for robot in robots])
        all_longitudinal = np.array([np.dot(robot.position - center, forward) for robot in robots])
        speeds = np.array([np.linalg.norm(robot.observed_velocity) for robot in robots])
        front_speeds = np.array([np.linalg.norm(robot.observed_velocity) for robot in front])
        variance = float(np.mean(lateral * lateral))
        if self.baseline <= 1e-12:
            self.baseline = max(variance, LATERAL_MIN_DELTA)
        delta, ratio = variance - self.baseline, variance / max(self.baseline, 1e-12)
        expanding = delta >= LATERAL_MIN_DELTA and ratio >= LATERAL_RATIO
        self.dwell = self.dwell + SAMPLE_INTERVAL if expanding else 0.0
        if not expanding and variance < self.baseline:
            self.baseline += BASELINE_ALPHA * (variance - self.baseline)
        rate = 0.0 if self.previous_variance is None else (variance - self.previous_variance) / SAMPLE_INTERVAL
        self.previous_variance = variance

        neighbor_map = {}
        boundary = []
        for robot in robots:
            neighbors = [peer for peer in robots if peer is not robot and np.linalg.norm(peer.position-robot.position) <= SUPPORT]
            neighbor_map[robot.robot_id] = neighbors
            bearings = sorted(math.atan2(peer.position[1]-robot.position[1], peer.position[0]-robot.position[0]) % (2*math.pi) for peer in neighbors)
            gaps = ([b-a for a,b in zip(bearings, bearings[1:])] + [bearings[0]+2*math.pi-bearings[-1]]) if len(bearings) >= 2 else [2*math.pi]
            if math.degrees(max(gaps)) >= GAP_THRESHOLD_DEG:
                boundary.append(robot)
        by_id = {robot.robot_id: robot for robot in boundary}
        remaining, components = set(by_id), []
        while remaining:
            group = {remaining.pop()}; queue = list(group)
            while queue:
                current = by_id[queue.pop()]
                for candidate in tuple(remaining):
                    if np.linalg.norm(current.position - by_id[candidate].position) <= SUPPORT:
                        remaining.remove(candidate); group.add(candidate); queue.append(candidate)
            components.append(group)
        components.sort(key=lambda group: (-len(group), min(group)))
        sizes = [len(group) for group in components]
        # Observable swarm centroid, not the GT/global origin.
        component_bearings = sorted(math.degrees(math.atan2(*(np.mean([by_id[value].position for value in group], axis=0)-center)[::-1])) % 360 for group in components)
        separations = ([b-a for a,b in zip(component_bearings, component_bearings[1:])] + [component_bearings[0]+360-component_bearings[-1]]) if len(component_bearings)>1 else []
        ids = set(by_id)
        retention = len(ids & self.previous_boundary_ids) / len(self.previous_boundary_ids) if self.previous_boundary_ids else 1.0
        split = self.previous_component_count < 2 <= len(components)
        self.previous_boundary_ids, self.previous_component_count = ids, len(components)

        boundary_positions = np.array([robot.position for robot in boundary]) if boundary else np.empty((0, 2))
        if boundary:
            relative = boundary_positions - center
            boundary_lateral = relative @ lateral_axis
            boundary_longitudinal = relative @ forward
            boundary_angles = np.unwrap(np.arctan2(boundary_lateral, boundary_longitudinal))
            covariance = np.cov(relative.T, bias=True) if len(boundary) > 1 else np.zeros((2, 2))
            eigenvalues = np.linalg.eigvalsh(covariance)
            anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-12))
            boundary_centroid = np.mean(boundary_positions, axis=0)
            centroid_displacement = 0.0 if self.previous_boundary_centroid is None else float(np.linalg.norm(boundary_centroid - self.previous_boundary_centroid))
            self.previous_boundary_centroid = boundary_centroid.copy()
        else:
            boundary_lateral = boundary_longitudinal = np.array([])
            boundary_angles = np.array([])
            anisotropy = centroid_displacement = 0.0

        # Motion bearing deviations are expressed in the robot-derived common
        # forward frame. No map axis or branch direction is consulted.
        moving_deviations = []
        lateral_velocities = []
        for robot in moving:
            longitudinal_velocity = float(np.dot(robot.observed_velocity, forward))
            lateral_velocity = float(np.dot(robot.observed_velocity, lateral_axis))
            lateral_velocities.append(lateral_velocity)
            moving_deviations.append(math.atan2(lateral_velocity, longitudinal_velocity))
        unwrapped_motion = np.unwrap(np.asarray(moving_deviations)) if moving_deviations else np.array([])
        motion_iqr = float(np.percentile(unwrapped_motion, 75) - np.percentile(unwrapped_motion, 25)) if len(unwrapped_motion) else 0.0
        degrees = np.array([len(neighbor_map[robot.robot_id]) for robot in robots], dtype=float)
        nearest = []
        for robot in robots:
            distances = [np.linalg.norm(peer.position - robot.position) for peer in robots if peer is not robot]
            nearest.append(min(distances) if distances else 0.0)
        graph_remaining = {robot.robot_id for robot in robots}
        graph_components = []
        while graph_remaining:
            group = {graph_remaining.pop()}; queue = list(group)
            while queue:
                current = queue.pop()
                for peer in neighbor_map[current]:
                    if peer.robot_id in graph_remaining:
                        graph_remaining.remove(peer.robot_id); group.add(peer.robot_id); queue.append(peer.robot_id)
            graph_components.append(group)
        component_probabilities = np.asarray(sizes, dtype=float) / max(sum(sizes), 1) if sizes else np.array([])
        component_entropy = float(-np.sum(component_probabilities * np.log(component_probabilities))) if len(component_probabilities) else 0.0
        return {
            "timestamp": timestamp, "front_cohort_size": len(front),
            "front_cohort_ids": tuple(robot.robot_id for robot in front),
            "front_cohort_center_x": float(front_center[0]), "front_cohort_center_y": float(front_center[1]),
            "runtime_forward_x": float(forward[0]), "runtime_forward_y": float(forward[1]),
            "front_cohort_lateral_variance": variance, "front_cohort_baseline": self.baseline,
            "front_cohort_expansion_ratio": ratio, "front_cohort_variance_rate": rate,
            "front_cohort_expansion_dwell": self.dwell, "front_cohort_sustained_marker": bool(self.dwell > 0),
            "front_cohort_lateral_span": float(np.ptp(lateral)),
            "front_cohort_longitudinal_span": float(np.ptp(longitudinal)),
            "front_cohort_mean_abs_lateral_velocity": float(np.mean(np.abs(front_lateral_velocity))),
            "front_cohort_lateral_velocity_variance": float(np.var(front_lateral_velocity)),
            "swarm_lateral_span": float(np.ptp(all_lateral)), "swarm_longitudinal_span": float(np.ptp(all_longitudinal)),
            "mean_speed": float(np.mean(speeds)), "front_cohort_mean_speed": float(np.mean(front_speeds)),
            "motion_bearing_mean_deg": math.degrees(float(np.mean(unwrapped_motion))) if len(unwrapped_motion) else 0.0,
            "motion_bearing_std_deg": math.degrees(float(np.std(unwrapped_motion))) if len(unwrapped_motion) else 0.0,
            "motion_bearing_iqr_deg": math.degrees(motion_iqr),
            "positive_lateral_flow_fraction": float(np.mean(np.asarray(lateral_velocities) > 0.0)) if lateral_velocities else 0.0,
            "negative_lateral_flow_fraction": float(np.mean(np.asarray(lateral_velocities) < 0.0)) if lateral_velocities else 0.0,
            "boundary_ids": tuple(robot.robot_id for robot in boundary),
            "boundary_component_ids": tuple(tuple(sorted(group)) for group in components),
            "boundary_count": len(boundary), "boundary_fraction": len(boundary)/len(robots),
            "component_count": len(components), "largest_component_size": sizes[0] if sizes else 0,
            "largest_component_fraction": sizes[0]/max(len(boundary),1) if sizes else 0.0,
            "second_component_size": sizes[1] if len(sizes)>1 else 0,
            "second_component_fraction": sizes[1]/max(len(boundary),1) if len(sizes)>1 else 0.0,
            "min_component_angular_separation": min(separations) if separations else math.nan,
            "boundary_membership_retention": retention, "first_component_split": int(split),
            "component_match_collision_count": 0,
            "boundary_spatial_span": float(np.linalg.norm(np.ptp(boundary_positions, axis=0))) if len(boundary_positions) else 0.0,
            "boundary_lateral_span": float(np.ptp(boundary_lateral)) if len(boundary_lateral) else 0.0,
            "boundary_longitudinal_span": float(np.ptp(boundary_longitudinal)) if len(boundary_longitudinal) else 0.0,
            "boundary_angular_spread_deg": math.degrees(float(np.ptp(boundary_angles))) if len(boundary_angles) else 0.0,
            "boundary_anisotropy": anisotropy, "boundary_centroid_displacement": centroid_displacement,
            "boundary_component_size_entropy": component_entropy,
            "largest_to_total_boundary_ratio": sizes[0]/max(len(boundary), 1) if sizes else 0.0,
            "second_to_largest_ratio": sizes[1]/max(sizes[0], 1) if len(sizes) > 1 else 0.0,
            "mean_neighbor_count": float(np.mean(degrees)), "median_neighbor_count": float(np.median(degrees)),
            "neighbor_degree_std": float(np.std(degrees)), "mean_nearest_neighbor_distance": float(np.mean(nearest)),
            "neighbor_graph_component_count": len(graph_components),
            "neighbor_graph_largest_component_fraction": max(map(len, graph_components), default=0)/len(robots),
        }


def evaluation_phase(geometry: GeometryCase, robots: list[Robot], row: dict) -> str:
    """Compute post-hoc GT phase without feeding Detector or motion."""
    if geometry.entrance_progress is None:
        return "CORRIDOR_ONLY"
    frontmost = max(float(robot.position[1]) for robot in robots)
    if frontmost < geometry.entrance_progress - geometry.approach_distance:
        return "CORRIDOR"
    if frontmost < geometry.entrance_progress:
        return "OPENING_APPROACH"
    if row["front_cohort_center_y"] < geometry.entrance_progress:
        return "BOUNDARY_CROSSING"
    return "JUNCTION_REGION"


def split_episodes(rows: list[dict]) -> list[dict]:
    """Return maximal contiguous component_count >= 2 intervals."""
    episodes, active = [], []
    for row in rows + [{"component_count": 0}]:
        if row["component_count"] >= 2:
            active.append(row)
        elif active:
            separations = [item["min_component_angular_separation"] for item in active if math.isfinite(item["min_component_angular_separation"])]
            episodes.append({"phase": active[0]["phase"], "start_time": active[0]["timestamp"], "end_time": active[-1]["timestamp"], "duration": len(active)*SAMPLE_INTERVAL,
                "max_component_count": max(item["component_count"] for item in active), "mean_largest_fraction": float(np.mean([item["largest_component_fraction"] for item in active])),
                "max_second_fraction": max(item["second_component_fraction"] for item in active), "mean_boundary_fraction": float(np.mean([item["boundary_fraction"] for item in active])),
                "mean_min_angular_separation": float(np.mean(separations)) if separations else math.nan, "mean_retention": float(np.mean([item["boundary_membership_retention"] for item in active]))})
            active = []
    return episodes


def run_case(geometry: GeometryCase, frames: int, observer=None):
    robots, detector, rows, penetration_total = initial_swarm(), Detector(), [], 0
    for frame in range(frames):
        outside, penetration = integrate_step(robots, geometry); penetration_total += penetration
        if frame % round(SAMPLE_INTERVAL/DT) == 0:
            row = detector.sample(robots, frame*DT)
            if row:
                entrance = geometry.entrance_progress
                row.update({"map_case": geometry.case_id, "frame": frame, "phase": evaluation_phase(geometry, robots, row),
                    "evaluation_frontmost_progress": max(float(robot.position[1]) for robot in robots),
                    "evaluation_front_center_progress": row["front_cohort_center_y"],
                    "evaluation_entrance_progress": entrance if entrance is not None else math.nan,
                    "gt_frontmost_crossed": bool(entrance is not None and max(float(robot.position[1]) for robot in robots) >= entrance),
                    "gt_front_center_crossed": bool(entrance is not None and row["front_cohort_center_y"] >= entrance),
                    "outside_free_space_robot_count": outside, "wall_penetration_event_count": penetration_total})
                rows.append(row)
        if observer and observer(frame, robots, detector, rows[-1] if rows else None) is False:
            break
    return rows, split_episodes(rows)


def _first_crossing(rows, field, entrance):
    return math.nan if entrance is None else next((row["timestamp"] for row in rows if row[field] >= entrance), math.nan)


def _longest(episodes, phases: Iterable[str]):
    return max((episode["duration"] for episode in episodes if episode["phase"] in set(phases)), default=0.0)


def summarize_case(geometry, rows, episodes):
    corridor = [row for row in rows if row["phase"] in ("CORRIDOR", "CORRIDOR_ONLY")]
    opening = [row for row in rows if row["phase"] in ("OPENING_APPROACH", "BOUNDARY_CROSSING", "JUNCTION_REGION")]
    frontmost = _first_crossing(rows, "evaluation_frontmost_progress", geometry.entrance_progress)
    front_center = _first_crossing(rows, "evaluation_front_center_progress", geometry.entrance_progress)
    lateral = next((row["timestamp"] for row in rows if row["front_cohort_sustained_marker"]), math.nan)
    mean = lambda values: float(np.mean(values)) if values else math.nan
    return {"map_case": geometry.case_id,
        "geometry_parameters": f"width={geometry.incoming_width};junction={geometry.junction_width}x{geometry.junction_height};branches={[(b.angle_deg,b.width,b.length) for b in geometry.branches]}",
        "sample_count": len(rows), "frontmost_crossing": frontmost, "front_center_crossing": front_center,
        "front_lateral_sustained_onset": lateral,
        "delta_lateral_frontmost": lateral-frontmost if math.isfinite(lateral) and math.isfinite(frontmost) else math.nan,
        "delta_lateral_front_center": lateral-front_center if math.isfinite(lateral) and math.isfinite(front_center) else math.nan,
        "first_component_split": next((row["timestamp"] for row in rows if row["first_component_split"]), math.nan),
        "corridor_component_ge2_fraction": mean([row["component_count"]>=2 for row in corridor]),
        "opening_component_ge2_fraction": mean([row["component_count"]>=2 for row in opening]),
        "corridor_largest_component_fraction": mean([row["largest_component_fraction"] for row in corridor]),
        "opening_largest_component_fraction": mean([row["largest_component_fraction"] for row in opening]),
        "longest_corridor_split_duration": _longest(episodes, ("CORRIDOR","CORRIDOR_ONLY")),
        "longest_opening_split_duration": _longest(episodes, ("OPENING_APPROACH","BOUNDARY_CROSSING","JUNCTION_REGION")),
        "max_outside_free_space_robot_count": max((row["outside_free_space_robot_count"] for row in rows), default=0),
        "wall_penetration_event_count": max((row["wall_penetration_event_count"] for row in rows), default=0),
        "qualitative_result": "STRAIGHT_CONTROL" if geometry.entrance_progress is None else "OBSERVED"}


def _safe(row):
    return {key:value for key,value in row.items() if key not in ("front_cohort_ids","boundary_ids","boundary_component_ids")}


def save_case(output, geometry, rows, episodes):
    folder = output/geometry.case_id; folder.mkdir(parents=True, exist_ok=True)
    safe = [_safe(row) for row in rows]
    with (folder/"timeline.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(safe[0]) if safe else ["map_case"]); writer.writeheader(); writer.writerows(safe)
    fields=["phase","start_time","end_time","duration","max_component_count","mean_largest_fraction","max_second_fraction","mean_boundary_fraction","mean_min_angular_separation","mean_retention"]
    with (folder/"split_episodes.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(episodes)


def run_batch(frames: int, output: Path):
    output.mkdir(parents=True, exist_ok=True); all_rows=[]; summaries=[]; all_episodes=[]
    for case_id in CASE_IDS:
        geometry=make_geometry(case_id); rows,episodes=run_case(geometry,frames); save_case(output,geometry,rows,episodes)
        all_rows.extend(rows); summaries.append(summarize_case(geometry,rows,episodes)); all_episodes.extend(dict(map_case=case_id,**episode) for episode in episodes)
    with (output/"multigeometry_run_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    fields=["map_case","phase","sample_count","mean_lateral_ratio","component_ge2_fraction","mean_largest_fraction"]
    with (output/"multigeometry_phase_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for case_id in CASE_IDS:
            for phase in ("CORRIDOR_ONLY","CORRIDOR","OPENING_APPROACH","BOUNDARY_CROSSING","JUNCTION_REGION"):
                selected=[row for row in all_rows if row["map_case"]==case_id and row["phase"]==phase]
                writer.writerow({"map_case":case_id,"phase":phase,"sample_count":len(selected),
                    "mean_lateral_ratio":float(np.mean([row["front_cohort_expansion_ratio"] for row in selected])) if selected else math.nan,
                    "component_ge2_fraction":float(np.mean([row["component_count"]>=2 for row in selected])) if selected else math.nan,
                    "mean_largest_fraction":float(np.mean([row["largest_component_fraction"] for row in selected])) if selected else math.nan})
    event_fields=["map_case","front_lateral_sustained_onset","first_component_split","frontmost_crossing","front_center_crossing"]
    with (output/"multigeometry_event_alignment.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=event_fields); writer.writeheader(); writer.writerows({key:summary[key] for key in event_fields} for summary in summaries)
    episode_fields=["map_case","phase","start_time","end_time","duration","max_component_count","mean_largest_fraction","max_second_fraction","mean_boundary_fraction","mean_min_angular_separation","mean_retention"]
    with (output/"multigeometry_split_episodes.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=episode_fields); writer.writeheader(); writer.writerows(all_episodes)
    figure,axes=plt.subplots(2,2,figsize=(12,8))
    for case_id in CASE_IDS:
        selected=[row for row in all_rows if row["map_case"]==case_id]; time=[row["timestamp"] for row in selected]
        axes[0,0].plot(time,[row["front_cohort_expansion_ratio"] for row in selected],label=case_id)
        axes[0,1].plot(time,[row["component_count"] for row in selected],label=case_id)
        axes[1,0].plot(time,[row["largest_component_fraction"] for row in selected],label=case_id)
        axes[1,1].plot(time,[row["outside_free_space_robot_count"] for row in selected],label=case_id)
    for axis,title in zip(axes.flat,("FRONT_COHORT expansion ratio","120° component count","largest component fraction","outside free-space robots")):
        axis.set_title(title); axis.set_xlabel("time [s]"); axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=7); figure.suptitle("Corrected evaluation-only multi-geometry benchmark — GT used only for evaluation")
    figure.tight_layout(); figure.savefig(output/"multigeometry_validation.png",dpi=150); plt.close(figure)
    return summaries


def run_gui(case_id: str, frames: int, show_trails=False):
    """Run observation-only Pygame rendering; Clock never changes physics DT."""
    import pygame
    pygame.init(); screen=pygame.display.set_mode((1000,760)); pygame.display.set_caption("Evaluation-only Junction trigger geometry")
    clock,font=pygame.time.Clock(),pygame.font.Font(None,23); cases=list(CASE_IDS); current=cases.index(case_id)
    geometry,robots,detector=make_geometry(case_id),initial_swarm(),Detector(); frame=0; paused=False; gt=True; running=True; rows=[]; trails=[]
    def reset(selected):
        nonlocal geometry,robots,detector,frame,rows,trails,paused
        geometry,robots,detector=make_geometry(selected),initial_swarm(),Detector(); frame=0; rows=[]; trails=[]; paused=False
    world_to_screen=lambda point:(int(500+point[0]*2),int(650-(point[1]+180)*2))
    while running and (frames<=0 or frame<frames):
        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: running=False
                elif event.key==pygame.K_SPACE: paused=not paused
                elif event.key==pygame.K_r: reset(cases[current])
                elif event.key==pygame.K_g: gt=not gt
                elif pygame.K_1<=event.key<=pygame.K_6: current=event.key-pygame.K_1; reset(cases[current])
        if not paused:
            outside,penetration=integrate_step(robots,geometry)
            if frame%2==0:
                row=detector.sample(robots,frame*DT)
                if row: row.update({"phase":evaluation_phase(geometry,robots,row),"outside_free_space_robot_count":outside,"wall_penetration_event_count":penetration}); rows.append(row)
            if show_trails and rows: trails.append(np.array([rows[-1]["front_cohort_center_x"],rows[-1]["front_cohort_center_y"]])); trails[:]=trails[-160:]
            frame+=1
        screen.fill((20,23,29))
        for rect in geometry.free_space_rects: pygame.draw.polygon(screen,(52,58,66),[world_to_screen(point) for point in rect.vertices])
        for start,end in geometry.wall_segments: pygame.draw.line(screen,(225,225,225),world_to_screen(start),world_to_screen(end),3)
        if gt and geometry.entrance_progress is not None:
            y=geometry.entrance_progress; pygame.draw.line(screen,(250,190,45),world_to_screen((-geometry.incoming_width/2,y)),world_to_screen((geometry.incoming_width/2,y)),2)
        if show_trails and len(trails)>1: pygame.draw.lines(screen,(120,210,255),False,[world_to_screen(point) for point in trails],2)
        latest=rows[-1] if rows else None; front_ids=set(latest["front_cohort_ids"]) if latest else set(); component_by_id={}
        if latest:
            for index,component in enumerate(latest["boundary_component_ids"]):
                for robot_id in component: component_by_id[robot_id]=index
        colors=((255,90,90),(255,180,65),(210,85,255),(80,230,180))
        for robot in robots:
            color=(70,235,130) if robot.robot_id in front_ids else (85,145,230)
            if robot.robot_id in component_by_id: color=colors[component_by_id[robot.robot_id]%len(colors)]
            pygame.draw.circle(screen,color,world_to_screen(robot.position),max(3,round(ROBOT_RADIUS*2)))
        phase=latest["phase"] if latest else "INITIALIZING"
        hud=[f"Map: {geometry.case_id}",f"t={frame*DT:.2f}s  frame={frame}  {'PAUSED' if paused else 'RUNNING'}",f"phase={phase}  GT EVALUATION ONLY","SPACE pause | R reset | G GT overlay | 1-6 map | ESC quit"]
        if latest: hud += [f"front cohort={latest['front_cohort_size']}  variance={latest['front_cohort_lateral_variance']:.2f}",f"ratio={latest['front_cohort_expansion_ratio']:.3f}  rate={latest['front_cohort_variance_rate']:.2f}  dwell={latest['front_cohort_expansion_dwell']:.2f}",f"boundary={latest['boundary_count']}  components={latest['component_count']}  largest={latest['largest_component_fraction']:.3f}",f"outside={latest['outside_free_space_robot_count']}  wall contacts={latest['wall_penetration_event_count']}"]
        for index,text in enumerate(hud): screen.blit(font.render(text,True,(240,240,240)),(12,10+index*23))
        pygame.display.flip(); clock.tick(60)
    pygame.quit()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--map-case",choices=CASE_IDS); parser.add_argument("--frames",type=int,default=600)
    mode=parser.add_mutually_exclusive_group(); mode.add_argument("--gui",action="store_true"); mode.add_argument("--headless",action="store_true")
    parser.add_argument("--show-trails",action="store_true"); parser.add_argument("--output",type=Path,default=Path(os.environ.get("MULTIGEOM_OUTPUT",DEFAULT_OUTPUT)))
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.gui: run_gui(args.map_case or "M1_CROSS_BASELINE",args.frames,args.show_trails)
    elif args.map_case:
        geometry=make_geometry(args.map_case); rows,episodes=run_case(geometry,args.frames); save_case(args.output,geometry,rows,episodes); print(summarize_case(geometry,rows,episodes))
    else:
        summaries=run_batch(args.frames,args.output); print(f"output={args.output.resolve()} cases={len(summaries)}")


if __name__ == "__main__":
    main()
