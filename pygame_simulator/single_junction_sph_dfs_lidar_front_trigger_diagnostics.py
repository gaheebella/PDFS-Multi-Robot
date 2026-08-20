"""Evaluation-only heterogeneous LiDAR/front-cohort diagnostic.

The production simulator is executed unchanged under a bounded headless run.
This wrapper traces ``AnchorShadowManager.update`` after its existing local
expansion calculation, assigns one fixed hardware identity once, and records
local-observable cheap scan summaries.  It does not elect an Anchor, stop a
robot, or call Point Cloud confirmation as a trigger.
"""

from __future__ import annotations

import csv
import math
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = ROOT / "pygame_simulator/single_junction_sph_dfs_provisional_anchor_junction_confirmation.py"
OUTPUT = Path(os.environ.get(
    "SPH_DFS_DIAGNOSTIC_OUTPUT",
    str(ROOT / "junction_detection/integration/output/lidar_front_trigger_diagnostics"),
))


def _cross_geometry() -> tuple[list[tuple[int, int]], float, tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """Return the unchanged benchmark polygon, corridor width, and GT center."""
    center = (400, 350)
    scale = 0.70
    width = round(120 * scale)
    half = width // 2
    normal = round(180 * scale)
    right = normal * 2
    base = round(normal * (2.0 / 3.0))
    points = [
        (center[0] - half, center[1] - half - normal),
        (center[0] + half, center[1] - half - normal),
        (center[0] + half, center[1] - half),
        (center[0] + half + right, center[1] - half),
        (center[0] + half + right, center[1] + half),
        (center[0] + half, center[1] + half),
        (center[0] + half, center[1] + half + base),
        (center[0] - half, center[1] + half + base),
        (center[0] - half, center[1] + half),
        (center[0] - half - normal, center[1] + half),
        (center[0] - half - normal, center[1] - half),
        (center[0] - half, center[1] - half),
    ]
    opening_boundary = ((center[0] - half, center[1] + half), (center[0] + half, center[1] + half))
    return points, float(width), center, opening_boundary


class DiagnosticCollector:
    """Collect local diagnostics from the existing simulator update callback."""

    def __init__(self, simulate_lidar):
        self.simulate_lidar = simulate_lidar
        self.rows: list[dict] = []
        configured = os.environ.get("SPH_DFS_LIDAR_ROBOT_ID", "27")
        self.lidar_id: int | None = int(configured) if configured else None
        self.previous_ranges: np.ndarray | None = None
        self.previous_yaw: float | None = None
        self.previous_forward = None
        self.previous_heading = None
        self.lateral_states = {
            "BROAD_OBSERVED": {"previous_variance": None, "baseline": 0.0, "dwell": 0.0},
            "FRONT_COHORT": {"previous_variance": None, "baseline": 0.0, "dwell": 0.0},
            "FRONT_LOCAL_REDUCED": {"previous_variance": None, "baseline": 0.0, "dwell": 0.0},
        }
        self.local_lateral_states = {
            "LOCAL_FRONT_SURFACE": {"previous_variance": None, "baseline": 0.0, "dwell": 0.0},
            "LOCAL_FRONT_COHORT": {"previous_variance": None, "baseline": 0.0, "dwell": 0.0},
        }
        self.previous_local_membership: set[int] = set()
        self.local_membership_rows: list[dict] = []
        self.front_maintenance_dwell = 0.0
        self.previous_frontmost_signed_boundary: float | None = None
        self.previous_observed_signed_boundary: float | None = None
        self.last_sample_timestamp: float | None = None
        self.sample_interval_s = 0.1
        self.polygon, self.max_range, self.gt_center, self.opening_boundary = _cross_geometry()
        self.initial_setup_offset_y = float(os.environ.get("SPH_DFS_DIAGNOSTIC_START_OFFSET_Y", "20.0"))

    def _front_features(self, robots, globals_dict):
        """Recompute the existing local front/cohort geometry every sample."""
        min_speed = float(globals_dict.get("JUNCTION_COHORT_MIN_SPEED", 1.2))
        moving = [r.observed_velocity for r in robots if r.observed_velocity.length() >= min_speed]
        if not moving:
            return (), None, None, ()
        forward = sum(moving, type(moving[0])())
        if forward.length_squared() <= 1e-12:
            return (), None, None, ()
        forward = forward.normalize()
        if self.previous_forward is not None and forward.dot(self.previous_forward) < 0.0:
            # Resolve local motion's 180-degree sign ambiguity by temporal
            # continuity; no global/GT heading is used.
            forward = -forward
        self.previous_forward = forward
        center = sum((r.position for r in robots), type(robots[0].position)()) / max(len(robots), 1)
        projections = [(r.position - center).dot(forward) for r in robots]
        quantile = float(globals_dict.get("JUNCTION_FRONT_QUANTILE", 0.68))
        threshold = float(np.quantile(projections, quantile))
        front = tuple(r for r, value in zip(robots, projections) if value >= threshold)
        front_center = sum((r.position for r in front), type(robots[0].position)()) / max(len(front), 1)
        radius = float(globals_dict.get("JUNCTION_OBSERVATION_RADIUS", 84.0 * 1.35))
        observed = tuple(r for r in robots if r.position.distance_to(front_center) <= radius)
        return front, forward, projections, observed

    def _local_front_topology(self, mobile, globals_dict, timestamp):
        """Build a shadow front surface from existing SPH support neighbors.

        Each robot uses its own observed-velocity direction.  The support
        radius is the production ``SMOOTHING_LENGTH``; no front-specific
        radius, angle, quantile, GT geometry, or robot-count threshold is
        introduced here.
        """
        min_speed = float(globals_dict.get("JUNCTION_COHORT_MIN_SPEED", 1.2))
        support = float(globals_dict.get("SMOOTHING_LENGTH", 22.0 * 0.70))
        surface = []
        neighbor_map = {}
        direction_unavailable = 0
        for robot in mobile:
            velocity = robot.observed_velocity
            direction_available = velocity.length() >= min_speed
            neighbors = [
                peer for peer in mobile
                if peer is not robot and robot.position.distance_to(peer.position) <= support
            ]
            neighbor_map[robot] = neighbors
            forward_count = 0
            if direction_available:
                forward = velocity.normalize()
                forward_count = sum((peer.position - robot.position).dot(forward) > 0.0 for peer in neighbors)
                if forward_count == 0:
                    surface.append(robot)
            else:
                direction_unavailable += 1
            self.local_membership_rows.append({
                "timestamp": float(timestamp), "robot_id": robot.robot_id,
                "speed": float(velocity.length()), "direction_available": direction_available,
                "sph_neighbor_count": len(neighbors), "forward_neighbor_count": forward_count,
                "is_local_front_surface": False,
            })
        robot_by_id = {robot.robot_id: robot for robot in mobile}
        surface_ids = {robot.robot_id for robot in surface}
        cohort_ids = set(surface_ids)
        for robot in surface:
            cohort_ids.update(peer.robot_id for peer in neighbor_map[robot])
        for row in self.local_membership_rows[-len(mobile):]:
            row["is_local_front_surface"] = row["robot_id"] in surface_ids
            row["is_local_front_cohort"] = row["robot_id"] in cohort_ids
        current_ids = cohort_ids
        union = self.previous_local_membership | current_ids
        overlap = len(self.previous_local_membership & current_ids) / len(union) if union else 1.0
        self.previous_local_membership = current_ids
        return tuple(robot_by_id[robot_id] for robot_id in surface_ids), tuple(robot_by_id[robot_id] for robot_id in cohort_ids), direction_unavailable, overlap

    def _lateral_features(self, observed, forward, globals_dict, timestamp, state):
        """Apply the unchanged lateral math to one independent cohort state."""
        if not observed:
            return 0.0, state["baseline"], 0.0, 1.0, state["dwell"], 0.0, 0.0, 0.0, 0.0
        center = sum((r.position for r in observed), type(observed[0].position)()) / len(observed)
        lateral = type(forward)(-forward.y, forward.x)
        values = [(r.position - center).dot(lateral) for r in observed]
        variance = float(sum(v * v for v in values) / len(values))
        min_delta = float(globals_dict.get("JUNCTION_LATERAL_EXPANSION_MIN", (4.5 * 0.70) ** 2))
        ratio_limit = float(globals_dict.get("JUNCTION_LATERAL_EXPANSION_RATIO", 1.28))
        alpha = float(globals_dict.get("JUNCTION_BASELINE_ALPHA", 0.035))
        if state["baseline"] <= 1e-12:
            state["baseline"] = max(variance, min_delta)
        delta = variance - state["baseline"]
        ratio = variance / max(state["baseline"], 1e-12)
        expanding = delta >= min_delta and ratio >= ratio_limit
        dt = 0.0 if not self.rows else max(timestamp - self.rows[-1]["timestamp"], 1e-12)
        state["dwell"] = state["dwell"] + dt if expanding else 0.0
        if not expanding and variance < state["baseline"]:
            state["baseline"] += alpha * (variance - state["baseline"])
        rate = 0.0 if state["previous_variance"] is None else (variance - state["previous_variance"]) / max(dt, 1e-12)
        state["previous_variance"] = variance
        p10, p90 = np.percentile(values, [10.0, 90.0])
        return variance, state["baseline"], delta, ratio, state["dwell"], rate, min(values), max(values), float(p90 - p10)

    def _evaluation_geometry(self, position):
        """Return GT opening distance/phase; never used by runtime control."""
        (x0, y), (x1, _) = self.opening_boundary
        clamped_x = min(max(position.x, x0), x1)
        boundary_distance = math.hypot(position.x - clamped_x, position.y - y)
        width = self.max_range
        signed_from_boundary = position.y - y
        center_distance = math.hypot(position.x - self.gt_center[0], position.y - self.gt_center[1])
        if signed_from_boundary > 2.0 * width:
            phase = "CORRIDOR"
        elif signed_from_boundary > 0.0:
            phase = "OPENING_APPROACH"
        elif center_distance <= 2.0 * width:
            phase = "JUNCTION_REGION"
        else:
            phase = "POST_MIN_DISTANCE"
        inside_zone = signed_from_boundary <= 0.0 and x0 <= position.x <= x1
        return boundary_distance, center_distance, phase, inside_zone

    def _sph_evaluation_geometry(self, front, observed):
        """Compute cohort progress/phase using GT geometry for evaluation only."""
        boundary_y = self.opening_boundary[0][1]
        front_center = sum((r.position for r in front), type(front[0].position)()) / max(len(front), 1)
        observed_center = sum((r.position for r in observed), type(front[0].position)()) / max(len(observed), 1)
        relevant = observed or front
        frontmost = min(relevant, key=lambda r: r.position.y)
        front_distance = self._evaluation_geometry(front_center)[0]
        observed_distance = self._evaluation_geometry(observed_center)[0]
        frontmost_distance = self._evaluation_geometry(frontmost.position)[0]
        frontmost_signed = frontmost.position.y - boundary_y
        observed_signed = observed_center.y - boundary_y
        def crossed_fraction(cohort):
            """Return boundary-crossed fraction for evaluation only.

            The opening span is used to avoid counting robots that are below
            the horizontal boundary but outside the opening itself.  This is
            post-hoc GT bookkeeping and is never consumed by runtime control.
            """
            if not cohort:
                return 0.0
            crossed = sum(
                r.position.y <= boundary_y
                and self.opening_boundary[0][0] <= r.position.x <= self.opening_boundary[1][0]
                for r in cohort
            )
            return float(crossed / len(cohort))
        front_crossed_fraction = crossed_fraction(front)
        observed_crossed_fraction = crossed_fraction(observed)
        previous = self.previous_frontmost_signed_boundary
        previous_observed = self.previous_observed_signed_boundary
        self.previous_frontmost_signed_boundary = frontmost_signed
        self.previous_observed_signed_boundary = observed_signed
        if frontmost_signed > 0.0 and observed_signed > 0.0:
            phase = "SPH_CORRIDOR"
        elif frontmost_signed <= 0.0 and observed_signed > 0.0:
            phase = "SPH_OPENING_APPROACH"
        elif previous_observed is not None and previous_observed > 0.0 and observed_signed <= 0.0:
            phase = "SPH_BOUNDARY_CROSSING"
        elif observed_signed <= 0.0 and self._evaluation_geometry(observed_center)[1] <= 2.0 * self.max_range:
            phase = "SPH_JUNCTION_REGION"
        else:
            phase = "SPH_POST_BOUNDARY"
        return (
            front_center, observed_center, front_distance, observed_distance,
            frontmost_distance, phase, front_crossed_fraction,
            observed_crossed_fraction,
            frontmost_signed, observed_signed,
        )

    def _local_front_maintenance(self, lidar, robots, forward, dt, globals_dict):
        """Apply a weak bias only while local neighbors show persistent burial."""
        comm_range = float(globals_dict.get("COMM_RANGE", 54.0 * 0.70))
        robot_radius = float(globals_dict.get("ROBOT_RADIUS", 1.60 * 0.70))
        peers = [
            peer for peer in robots
            if peer is not lidar and peer.role == "NORMAL"
            and peer.connected_to_base
            and lidar.position.distance_to(peer.position) <= comm_range
        ]
        forward_rows, rear_rows = [], []
        for peer in peers:
            longitudinal = (peer.position - lidar.position).dot(forward)
            if longitudinal > robot_radius:
                forward_rows.append((longitudinal, peer))
            elif longitudinal < -robot_radius:
                rear_rows.append((-longitudinal, peer))
        forward_speeds = [peer.observed_velocity.dot(forward) for _, peer in forward_rows]
        own_forward_speed = lidar.observed_velocity.dot(forward)
        mean_forward_speed = float(np.mean(forward_speeds)) if forward_speeds else None
        # A local burial signature is a nearby forward layer moving faster than
        # the LiDAR robot while a rear layer still preserves connectivity.
        buried_signature = (
            bool(forward_rows) and bool(rear_rows)
            and mean_forward_speed is not None
            and own_forward_speed < mean_forward_speed
        )
        state = "FRONT" if not forward_rows else (
            "BURIED" if buried_signature else "UNCERTAIN"
        )
        self.front_maintenance_dwell = (
            self.front_maintenance_dwell + dt if state == "BURIED" else 0.0
        )
        required = float(globals_dict.get("JUNCTION_EXPANSION_DWELL_TIME", 0.14))
        active = state == "BURIED" and self.front_maintenance_dwell >= required and bool(rear_rows)
        push_speed = float(globals_dict.get("PRESSURE_PUSH_MAX_SPEED", 42.0 * 0.70))
        bias = forward * (0.08 * push_speed) if active else type(forward)()
        if active:
            lidar.velocity += bias * dt
        ranges = [value for value, _ in forward_rows]
        return {
            "local_forward_neighbor_count": len(forward_rows),
            "local_rear_neighbor_count": len(rear_rows),
            "nearest_forward_neighbor_range": min(ranges) if ranges else "",
            "mean_forward_neighbor_range": float(np.mean(ranges)) if ranges else "",
            "local_front_state": state,
            "front_maintenance_active": active,
            "front_maintenance_bias": float(bias.length()),
            "local_neighbor_connectivity_count": len(peers),
            "front_neighbor_mean_forward_speed": mean_forward_speed if mean_forward_speed is not None else "",
        }

    def sample(self, manager, robots, timestamp, dt, globals_dict):
        if (
            self.last_sample_timestamp is not None
            and timestamp - self.last_sample_timestamp < self.sample_interval_s - 1e-9
        ):
            return
        self.last_sample_timestamp = float(timestamp)
        mobile = tuple(r for r in robots if r.role == "NORMAL" and r.connected_to_base)
        if not mobile:
            return
        front, forward, projections, observed = self._front_features(mobile, globals_dict)
        if not front or forward is None:
            return
        local_surface, local_cohort, direction_unavailable, local_overlap = self._local_front_topology(
            mobile, globals_dict, timestamp
        )
        if self.lidar_id is None:
            # Hardware identity is assigned once from the initial front surface;
            # it is never re-elected on later frames.
            self.lidar_id = max(front, key=lambda r: r.position.dot(forward)).robot_id
        lidar = next((r for r in robots if r.robot_id == self.lidar_id), None)
        if lidar is None:
            return
        front_center = sum((r.position for r in front), type(lidar.position)()) / len(front)
        lidar_projection = (lidar.position - front_center).dot(forward)
        front_projections = [(r.position - front_center).dot(forward) for r in front]
        rank = sum(value > lidar_projection for value in front_projections) + 1
        global_projection = (lidar.position - front_center).dot(forward)
        all_progress = [(r.position - front_center).dot(forward) for r in mobile]
        frontier_max = max(all_progress)
        global_rank = sum(value > global_projection for value in all_progress) + 1
        maintenance = self._local_front_maintenance(lidar, robots, forward, dt, globals_dict)
        opening_distance, center_distance, evaluation_phase, inside_zone = self._evaluation_geometry(lidar.position)
        (
            front_center,
            observed_center,
            front_center_boundary_distance,
            observed_center_boundary_distance,
            frontmost_boundary_distance,
            sph_phase,
            front_crossed_fraction,
            observed_crossed_fraction,
            frontmost_signed_boundary,
            observed_signed_boundary,
        ) = self._sph_evaluation_geometry(front, observed)
        # All three sets are selected from local robot geometry only.  The
        # reduced radius is tied to the existing communication/sensing scale,
        # rather than an experiment-specific tuned distance.
        comm_range = float(globals_dict.get("COMM_RANGE", 54.0 * 0.70))
        observation_radius = float(globals_dict.get("JUNCTION_OBSERVATION_RADIUS", 84.0 * 1.35))
        reduced_radius = min(comm_range, observation_radius)
        front_local = tuple(
            robot for robot in mobile
            if robot.position.distance_to(front_center) <= reduced_radius
        )
        cohorts = {
            "BROAD_OBSERVED": observed,
            "FRONT_COHORT": front,
            "FRONT_LOCAL_REDUCED": front_local,
        }

        velocity = lidar.observed_velocity
        raw_heading = ""
        heading_flip_corrected = False
        if velocity.length_squared() > 1e-12:
            heading = velocity.normalize()
            raw_heading = math.degrees(math.atan2(heading.y, heading.x))
            if self.previous_heading is not None and heading.dot(self.previous_heading) < 0.0:
                heading = -heading
                heading_flip_corrected = True
            self.previous_heading = heading
            yaw = math.degrees(math.atan2(heading.y, heading.x))
            self.previous_yaw = yaw
        elif self.previous_yaw is not None:
            yaw = self.previous_yaw
        else:
            yaw = math.degrees(math.atan2(forward.y, forward.x))
        scan = self.simulate_lidar(
            polygon_points=self.polygon,
            anchor_xy=(lidar.position.x, lidar.position.y),
            anchor_reference_yaw_deg=yaw,
            max_range_world_units=self.max_range,
        )
        angles = scan.angles_deg
        left = (angles >= -135.0) & (angles <= -45.0)
        right = (angles >= 45.0) & (angles <= 135.0)
        left_ranges = scan.ranges[left]
        right_ranges = scan.ranges[right]
        scan_change = ""
        if self.previous_ranges is not None:
            scan_change = float(np.mean(np.abs(scan.ranges - self.previous_ranges)))
        self.previous_ranges = scan.ranges.copy()
        cohort_features = {
            name: self._lateral_features(
                cohort, forward, globals_dict, timestamp, self.lateral_states[name]
            ) for name, cohort in cohorts.items()
        }
        local_features = {
            name: self._lateral_features(
                cohort, forward, globals_dict, timestamp, self.local_lateral_states[name]
            ) for name, cohort in {
                "LOCAL_FRONT_SURFACE": local_surface,
                "LOCAL_FRONT_COHORT": local_cohort,
            }.items()
        }
        variance, baseline, variance_delta, expansion_ratio, expansion_dwell, variance_rate, lateral_min, lateral_max, lateral_p90_span = cohort_features["BROAD_OBSERVED"]
        row = {
            "timestamp": float(timestamp),
            "lidar_robot_id": self.lidar_id,
            "lidar_role": "FRONT_LIDAR_LEADER",
            "front_cohort_robot_count": len(front),
            "lidar_is_in_front_cohort": lidar in front,
            "lidar_front_rank": rank,
            "front_projection_relative_to_cohort": float(lidar_projection),
            "global_frontier_max_projection": float(frontier_max),
            "lidar_global_frontier_gap": float(frontier_max - global_projection),
            "lidar_global_progress_rank": global_rank,
            "raw_local_heading_deg": raw_heading,
            "stable_local_heading_deg": yaw,
            "heading_flip_corrected": heading_flip_corrected,
            "lidar_local_forward_speed": float(lidar.observed_velocity.dot(forward)),
            **maintenance,
            "lidar_minus_front_mean_speed": (
                "" if maintenance["front_neighbor_mean_forward_speed"] == ""
                else float(lidar.observed_velocity.dot(forward))
                - maintenance["front_neighbor_mean_forward_speed"]
            ),
            "lateral_variance": variance,
            "lateral_baseline": baseline,
            "lateral_variance_delta": variance_delta,
            "lateral_expansion_ratio": expansion_ratio,
            "lateral_expansion_dwell": expansion_dwell,
            "lateral_variance_rate": variance_rate,
            "lateral_min": lateral_min,
            "lateral_max": lateral_max,
            "lateral_span": lateral_max - lateral_min,
            "lateral_p90_minus_p10": lateral_p90_span,
            "cheap_lidar_left_range": float(np.mean(left_ranges)),
            "cheap_lidar_right_range": float(np.mean(right_ranges)),
            "cheap_lidar_left_free_fraction": float(np.mean(~scan.hit[left])),
            "cheap_lidar_right_free_fraction": float(np.mean(~scan.hit[right])),
            "cheap_lidar_scan_change": scan_change,
            "cheap_lidar_free_angular_fraction": float(np.mean(~scan.hit)),
            "evaluation_only_gt_distance_to_junction": math.hypot(
                lidar.position.x - self.gt_center[0], lidar.position.y - self.gt_center[1]
            ),
            "evaluation_only_gt_distance_to_opening_boundary": opening_distance,
            "evaluation_only_inside_junction_opening_zone": inside_zone,
            "evaluation_only_phase": evaluation_phase,
            "evaluation_only_sph_phase": sph_phase,
            "evaluation_only_front_cohort_center_distance_to_opening_boundary": front_center_boundary_distance,
            "evaluation_only_observed_cohort_center_distance_to_opening_boundary": observed_center_boundary_distance,
            "evaluation_only_frontmost_progress_distance_to_opening_boundary": frontmost_boundary_distance,
            "evaluation_only_frontmost_boundary_distance": frontmost_boundary_distance,
            "evaluation_only_frontmost_signed_boundary_distance": frontmost_signed_boundary,
            "evaluation_only_front_cohort_signed_boundary_distance": front_center.y - self.opening_boundary[0][1],
            "evaluation_only_observed_cohort_signed_boundary_distance": observed_signed_boundary,
            "evaluation_only_front_cohort_crossed_fraction": front_crossed_fraction,
            "evaluation_only_observed_cohort_crossed_fraction": observed_crossed_fraction,
            "observed_cohort_robot_count": len(observed),
            "broad_observed_robot_count": len(observed),
            "front_local_robot_count": len(front_local),
            "front_local_reduced_radius": reduced_radius,
            "local_front_surface_robot_count": len(local_surface),
            "local_front_cohort_robot_count": len(local_cohort),
            "local_front_direction_unavailable_count": direction_unavailable,
            "local_front_cohort_jaccard_overlap": local_overlap,
            # Descriptive analysis markers only; these are not production
            # trigger thresholds and do not affect robot behavior.
            "lateral_ratio_gt_1_1": expansion_ratio > 1.1,
            "lateral_ratio_gt_1_28": expansion_ratio > 1.28,
            "lateral_dwell_positive": expansion_dwell > 0.0,
            "scan_change_gt_5": bool(scan_change != "" and scan_change > 5.0),
        }
        for cohort_name, values in cohort_features.items():
            prefix = cohort_name.lower()
            row.update({
                f"{prefix}_variance": values[0],
                f"{prefix}_baseline": values[1],
                f"{prefix}_variance_delta": values[2],
                f"{prefix}_expansion_ratio": values[3],
                f"{prefix}_expansion_dwell": values[4],
                f"{prefix}_variance_rate": values[5],
                f"{prefix}_lateral_span": values[7] - values[6],
                f"{prefix}_lateral_p90_minus_p10": values[8],
                f"{prefix}_ratio_gt_1_1": values[3] > 1.1,
                f"{prefix}_ratio_gt_1_28": values[3] > 1.28,
                f"{prefix}_dwell_positive": values[4] > 0.0,
                f"{prefix}_sustained_marker": values[4] > 0.0 and values[5] > 0.0,
            })
        for cohort_name, values in local_features.items():
            prefix = cohort_name.lower()
            row.update({
                f"{prefix}_variance": values[0],
                f"{prefix}_baseline": values[1],
                f"{prefix}_variance_delta": values[2],
                f"{prefix}_expansion_ratio": values[3],
                f"{prefix}_expansion_dwell": values[4],
                f"{prefix}_variance_rate": values[5],
                f"{prefix}_lateral_span": values[7] - values[6],
                f"{prefix}_lateral_p90_minus_p10": values[8],
                f"{prefix}_ratio_gt_1_1": values[3] > 1.1,
                f"{prefix}_ratio_gt_1_28": values[3] > 1.28,
                f"{prefix}_dwell_positive": values[4] > 0.0,
                f"{prefix}_sustained_marker": values[4] > 0.0 and values[5] > 0.0,
            })
        self.rows.append(row)

    def save(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        fields = list(self.rows[0]) if self.rows else ["timestamp"]
        with (OUTPUT / "lidar_front_trigger_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        if not self.rows:
            return
        self._save_phase_summary()
        self._save_event_summary()
        self._save_cohort_summary()
        self._save_cohort_events()
        self._save_cohort_plot()
        self._save_local_front_outputs()
        time = np.asarray([r["timestamp"] for r in self.rows])
        variance = np.asarray([r["lateral_variance"] for r in self.rows])
        scan_change = np.asarray([float(r["cheap_lidar_scan_change"] or 0.0) for r in self.rows])
        gt_distance = np.asarray([r["evaluation_only_gt_distance_to_junction"] for r in self.rows])
        boundary_distance = np.asarray([r["evaluation_only_gt_distance_to_opening_boundary"] for r in self.rows])
        figure, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        axes[0].plot(time, variance, label="lateral variance")
        axes[0].plot(time, [r["lateral_baseline"] for r in self.rows], label="adaptive baseline")
        axes[0].legend(); axes[0].set_ylabel("SPH spread")
        axes[1].plot(time, [r["lateral_expansion_ratio"] for r in self.rows], label="lateral ratio")
        axes[1].plot(time, [r["lateral_expansion_dwell"] for r in self.rows], label="lateral dwell")
        axes[1].legend(); axes[1].set_ylabel("ratio / dwell")
        axes[2].plot(time, [r["evaluation_only_front_cohort_center_distance_to_opening_boundary"] for r in self.rows], label="front cohort center")
        axes[2].plot(time, [r["evaluation_only_observed_cohort_center_distance_to_opening_boundary"] for r in self.rows], label="observed cohort center")
        axes[2].plot(time, [r["evaluation_only_frontmost_boundary_distance"] for r in self.rows], label="frontmost")
        axes[2].axhline(0.0, color="black", linestyle=":", label="boundary")
        axes[2].legend(); axes[2].set_ylabel("boundary distance")
        axes[3].plot(time, [r["evaluation_only_front_cohort_crossed_fraction"] for r in self.rows], label="front crossed fraction")
        axes[3].plot(time, [r["evaluation_only_observed_cohort_crossed_fraction"] for r in self.rows], label="observed crossed fraction")
        axes[3].set_ylim(-0.02, 1.02); axes[3].legend(); axes[3].set_ylabel("crossed fraction"); axes[3].set_xlabel("time [s]")
        event_summary = self._event_summary()
        event_styles = {
            "first_sustained_lateral_onset": ("tab:red", "sustained lateral onset"),
            "first_existing_expansion_ratio_onset": ("tab:orange", "ratio onset"),
            "first_positive_lateral_dwell": ("tab:purple", "positive dwell"),
            "frontmost_boundary_crossing": ("tab:blue", "frontmost crossing"),
            "front_cohort_center_crossing": ("tab:green", "front center crossing"),
            "observed_cohort_center_crossing": ("tab:brown", "observed center crossing"),
        }
        for key, (color, label) in event_styles.items():
            timestamp = event_summary.get(key)
            if timestamp is not None:
                for axis in axes:
                    axis.axvline(timestamp, color=color, linestyle="--", alpha=0.65, label=label)
        for axis in axes:
            for row in self.rows:
                if row["evaluation_only_sph_phase"] == "SPH_OPENING_APPROACH":
                    axis.axvspan(row["timestamp"] - 0.05, row["timestamp"] + 0.05, color="tab:green", alpha=0.03)
        figure.suptitle("Fixed hardware LiDAR/front-cohort diagnostic (GT overlays evaluation-only)")
        figure.tight_layout()
        figure.savefig(OUTPUT / "lidar_front_trigger_timeline.png", dpi=150)
        plt.close(figure)
        self._print_onset_summary()

    def _event_summary(self):
        """Compute retrospective event times; all markers are evaluation-only."""
        if not self.rows:
            return {}
        def first(predicate):
            row = next((item for item in self.rows if predicate(item)), None)
            return None if row is None else float(row["timestamp"])
        def crossing(field):
            previous = None
            for row in self.rows:
                current = float(row[field])
                if previous is not None and previous > 0.0 and current <= 0.0:
                    return float(row["timestamp"])
                previous = current
            return None
        # B/C reuse existing expansion and dwell definitions.  A requires the
        # same existing condition plus a positive measured variance rate, so
        # no new production threshold is introduced.
        events = {
            "first_sustained_lateral_onset": first(lambda r: r["lateral_dwell_positive"] and r["lateral_variance_rate"] > 0.0),
            "first_existing_expansion_ratio_onset": first(lambda r: r["lateral_ratio_gt_1_28"]),
            "first_positive_lateral_dwell": first(lambda r: r["lateral_dwell_positive"]),
            "frontmost_boundary_crossing": crossing("evaluation_only_frontmost_signed_boundary_distance"),
            "front_cohort_center_crossing": crossing("evaluation_only_front_cohort_signed_boundary_distance"),
            "observed_cohort_center_crossing": crossing("evaluation_only_observed_cohort_signed_boundary_distance"),
        }
        for name, crossing in (("frontmost", "frontmost_boundary_crossing"), ("front_center", "front_cohort_center_crossing"), ("observed_center", "observed_cohort_center_crossing")):
            onset = events["first_sustained_lateral_onset"]
            crossing_time = events[crossing]
            events[f"delta_t_{name}"] = None if onset is None or crossing_time is None else onset - crossing_time
        return events

    def _save_event_summary(self):
        """Persist retrospective onset/crossing timestamps and signed deltas."""
        events = self._event_summary()
        with (OUTPUT / "lidar_front_trigger_event_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event", "timestamp_or_delta_s"])
            writer.writerows((key, value if value is not None else "") for key, value in events.items())

    def _cohort_event(self, cohort_name, marker):
        prefix = cohort_name.lower()
        row = next((r for r in self.rows if bool(r[f"{prefix}_{marker}"])), None)
        return None if row is None else float(row["timestamp"])

    def _save_cohort_events(self):
        """Persist cohort onset times and deltas against GT entry references."""
        reference = self._event_summary()
        with (OUTPUT / "lidar_front_trigger_cohort_events.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["cohort_name", "ratio_onset", "positive_dwell_onset", "sustained_onset", "delta_t_frontmost", "delta_t_front_center"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for cohort in self.lateral_states:
                sustained = self._cohort_event(cohort, "sustained_marker")
                ratio = self._cohort_event(cohort, "ratio_gt_1_28")
                dwell = self._cohort_event(cohort, "dwell_positive")
                writer.writerow({
                    "cohort_name": cohort,
                    "ratio_onset": "" if ratio is None else ratio,
                    "positive_dwell_onset": "" if dwell is None else dwell,
                    "sustained_onset": "" if sustained is None else sustained,
                    "delta_t_frontmost": "" if sustained is None or reference.get("frontmost_boundary_crossing") is None else sustained - reference["frontmost_boundary_crossing"],
                    "delta_t_front_center": "" if sustained is None or reference.get("front_cohort_center_crossing") is None else sustained - reference["front_cohort_center_crossing"],
                })

    def _save_cohort_summary(self):
        """Write cohort event and phase statistics for this single run."""
        phases = ("SPH_CORRIDOR", "SPH_OPENING_APPROACH", "SPH_JUNCTION_REGION")
        fields = ["cohort_name", "mean_robot_count", "sustained_onset", "delta_t_frontmost", "delta_t_front_center"]
        for phase in phases:
            fields.extend([
                f"{phase.lower()}_variance_mean", f"{phase.lower()}_variance_std",
                f"{phase.lower()}_variance_rate_std", f"{phase.lower()}_ratio_mean",
                f"{phase.lower()}_ratio_max", f"{phase.lower()}_false_ratio_1_28_count",
                f"{phase.lower()}_positive_dwell_count",
            ])
        with (OUTPUT / "lidar_front_trigger_cohort_comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for cohort_name in self.lateral_states:
                prefix = cohort_name.lower()
                count_key = {
                    "BROAD_OBSERVED": "broad_observed_robot_count",
                    "FRONT_COHORT": "front_cohort_robot_count",
                    "FRONT_LOCAL_REDUCED": "front_local_robot_count",
                }[cohort_name]
                sustained = self._cohort_event(cohort_name, "sustained_marker")
                row = {
                    "cohort_name": cohort_name,
                    "mean_robot_count": float(np.mean([r[count_key] for r in self.rows])),
                    "sustained_onset": sustained if sustained is not None else "",
                    "delta_t_frontmost": "" if sustained is None or self._event_summary().get("frontmost_boundary_crossing") is None else sustained - self._event_summary()["frontmost_boundary_crossing"],
                    "delta_t_front_center": "" if sustained is None or self._event_summary().get("front_cohort_center_crossing") is None else sustained - self._event_summary()["front_cohort_center_crossing"],
                }
                for phase in phases:
                    rows = [r for r in self.rows if r["evaluation_only_sph_phase"] == phase]
                    values = [float(r[f"{prefix}_variance"]) for r in rows]
                    rates = [float(r[f"{prefix}_variance_rate"]) for r in rows]
                    ratios = [float(r[f"{prefix}_expansion_ratio"]) for r in rows]
                    key = phase.lower()
                    row.update({
                        f"{key}_variance_mean": float(np.mean(values)) if values else "",
                        f"{key}_variance_std": float(np.std(values)) if values else "",
                        f"{key}_variance_rate_std": float(np.std(rates)) if rates else "",
                        f"{key}_ratio_mean": float(np.mean(ratios)) if ratios else "",
                        f"{key}_ratio_max": max(ratios) if ratios else "",
                        f"{key}_false_ratio_1_28_count": sum(r[f"{prefix}_ratio_gt_1_28"] for r in rows),
                        f"{key}_positive_dwell_count": sum(r[f"{prefix}_dwell_positive"] for r in rows),
                    })
                writer.writerow(row)

    def _save_cohort_plot(self):
        """Plot the three retrospective local-cohort signals."""
        time = np.asarray([r["timestamp"] for r in self.rows])
        fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
        labels = {"BROAD_OBSERVED": "BROAD", "FRONT_COHORT": "FRONT", "FRONT_LOCAL_REDUCED": "FRONT_LOCAL"}
        for cohort, label in labels.items():
            prefix = cohort.lower()
            axes[0].plot(time, [r[f"{prefix}_variance"] for r in self.rows], label=label)
            axes[1].plot(time, [r[f"{prefix}_expansion_ratio"] for r in self.rows], label=label)
            axes[2].plot(time, [r[f"{prefix}_variance_rate"] for r in self.rows], label=label)
        axes[3].plot(time, [r["evaluation_only_frontmost_signed_boundary_distance"] for r in self.rows], label="frontmost GT")
        axes[3].plot(time, [r["evaluation_only_front_cohort_signed_boundary_distance"] for r in self.rows], label="front center GT")
        axes[3].plot(time, [r["evaluation_only_observed_cohort_signed_boundary_distance"] for r in self.rows], label="observed center GT")
        axes[3].axhline(0.0, color="black", linestyle=":")
        for cohort, label in labels.items():
            count_key = "observed_cohort_robot_count" if cohort == "BROAD_OBSERVED" else ("front_cohort_robot_count" if cohort == "FRONT_COHORT" else "front_local_robot_count")
            axes[4].plot(time, [r[count_key] for r in self.rows], label=label)
        event_times = [("frontmost crossing", self._event_summary().get("frontmost_boundary_crossing")),
                       ("front center crossing", self._event_summary().get("front_cohort_center_crossing"))]
        for cohort, label in labels.items():
            event_times.append((f"{label} sustained onset", self._cohort_event(cohort, "sustained_marker")))
        for label, timestamp in event_times:
            if timestamp is not None:
                for axis in axes:
                    axis.axvline(timestamp, linestyle="--", alpha=0.55, label=label)
        for axis, ylabel in zip(axes, ("variance", "ratio", "variance rate", "signed boundary distance", "robot count")):
            axis.set_ylabel(ylabel); axis.legend(loc="best")
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Retrospective lateral cohort comparison (GT overlays evaluation-only)")
        fig.tight_layout()
        fig.savefig(OUTPUT / "lidar_front_trigger_cohort_comparison.png", dpi=150)
        plt.close(fig)

    def _save_local_front_outputs(self):
        """Export local-topology timeline, memberships, events, and plot."""
        timeline_fields = [
            "timestamp", "front_cohort_robot_count", "local_front_surface_robot_count",
            "local_front_cohort_robot_count", "local_front_direction_unavailable_count",
            "local_front_cohort_jaccard_overlap", "local_front_surface_variance",
            "local_front_surface_baseline", "local_front_surface_expansion_ratio",
            "local_front_surface_expansion_dwell", "local_front_surface_variance_rate",
            "local_front_cohort_variance", "local_front_cohort_baseline",
            "local_front_cohort_expansion_ratio", "local_front_cohort_expansion_dwell",
            "local_front_cohort_variance_rate", "evaluation_only_sph_phase",
            "evaluation_only_frontmost_signed_boundary_distance",
            "evaluation_only_front_cohort_signed_boundary_distance",
        ]
        with (OUTPUT / "lidar_front_trigger_local_front_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=timeline_fields); writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in timeline_fields} for row in self.rows)
        membership_fields = ["timestamp", "robot_id", "speed", "direction_available", "sph_neighbor_count", "forward_neighbor_count", "is_local_front_surface", "is_local_front_cohort"]
        with (OUTPUT / "local_front_membership.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=membership_fields); writer.writeheader(); writer.writerows(self.local_membership_rows)
        reference = self._event_summary()
        with (OUTPUT / "local_front_event_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["cohort_name", "ratio_onset", "positive_dwell_onset", "sustained_onset", "delta_t_frontmost", "delta_t_quantile_front_center"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for cohort in ("FRONT_COHORT", "LOCAL_FRONT_SURFACE", "LOCAL_FRONT_COHORT"):
                sustained = self._cohort_event(cohort, "sustained_marker")
                ratio = self._cohort_event(cohort, "ratio_gt_1_28")
                dwell = self._cohort_event(cohort, "dwell_positive")
                writer.writerow({
                    "cohort_name": cohort,
                    "ratio_onset": "" if ratio is None else ratio,
                    "positive_dwell_onset": "" if dwell is None else dwell,
                    "sustained_onset": "" if sustained is None else sustained,
                    "delta_t_frontmost": "" if sustained is None or reference.get("frontmost_boundary_crossing") is None else sustained - reference["frontmost_boundary_crossing"],
                    "delta_t_quantile_front_center": "" if sustained is None or reference.get("front_cohort_center_crossing") is None else sustained - reference["front_cohort_center_crossing"],
                })
        phases = ("SPH_CORRIDOR", "SPH_OPENING_APPROACH", "SPH_JUNCTION_REGION")
        with (OUTPUT / "local_front_phase_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["cohort_name", "phase", "sample_count", "variance_mean", "variance_std", "ratio_mean", "ratio_max", "ratio_gt_1_1_count", "ratio_gt_1_28_count", "positive_dwell_count", "sustained_count"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for cohort in ("FRONT_COHORT", "LOCAL_FRONT_SURFACE", "LOCAL_FRONT_COHORT"):
                prefix = cohort.lower()
                for phase in phases:
                    rows = [r for r in self.rows if r["evaluation_only_sph_phase"] == phase]
                    values = [float(r[f"{prefix}_variance"]) for r in rows]; ratios = [float(r[f"{prefix}_expansion_ratio"]) for r in rows]
                    writer.writerow({"cohort_name": cohort, "phase": phase, "sample_count": len(rows), "variance_mean": float(np.mean(values)) if values else "", "variance_std": float(np.std(values)) if values else "", "ratio_mean": float(np.mean(ratios)) if ratios else "", "ratio_max": max(ratios) if ratios else "", "ratio_gt_1_1_count": sum(r[f"{prefix}_ratio_gt_1_1"] for r in rows), "ratio_gt_1_28_count": sum(r[f"{prefix}_ratio_gt_1_28"] for r in rows), "positive_dwell_count": sum(r[f"{prefix}_dwell_positive"] for r in rows), "sustained_count": sum(r[f"{prefix}_sustained_marker"] for r in rows)})
        time = np.asarray([r["timestamp"] for r in self.rows])
        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        for cohort, label in (("FRONT_COHORT", "QUANTILE_FRONT"), ("LOCAL_FRONT_SURFACE", "LOCAL_SURFACE"), ("LOCAL_FRONT_COHORT", "LOCAL_COHORT")):
            prefix = cohort.lower(); axes[0].plot(time, [r[f"{prefix}_robot_count"] if f"{prefix}_robot_count" in r else r["local_front_surface_robot_count"] if cohort == "LOCAL_FRONT_SURFACE" else r["local_front_cohort_robot_count"] if cohort == "LOCAL_FRONT_COHORT" else r["front_cohort_robot_count"] for r in self.rows], label=label)
            axes[1].plot(time, [r[f"{prefix}_expansion_ratio"] for r in self.rows], label=label)
            axes[2].plot(time, [r[f"{prefix}_variance"] for r in self.rows], label=label)
        axes[3].plot(time, [r["evaluation_only_frontmost_signed_boundary_distance"] for r in self.rows], label="frontmost GT")
        axes[3].plot(time, [r["evaluation_only_front_cohort_signed_boundary_distance"] for r in self.rows], label="quantile center GT")
        axes[4].plot(time, [r["local_front_cohort_jaccard_overlap"] for r in self.rows], label="local cohort Jaccard")
        for axis, label in zip(axes, ("robot count", "expansion ratio", "variance", "signed boundary distance", "membership overlap")):
            axis.set_ylabel(label); axis.legend(loc="best")
        axes[-1].set_xlabel("time [s]"); fig.suptitle("Local front topology shadow comparison (GT evaluation-only)"); fig.tight_layout(); fig.savefig(OUTPUT / "local_front_shadow_comparison.png", dpi=150); plt.close(fig)

    def _save_phase_summary(self):
        """Summarize actual cohort phases and descriptive false onsets."""
        phases = ("SPH_CORRIDOR", "SPH_OPENING_APPROACH", "SPH_BOUNDARY_CROSSING", "SPH_JUNCTION_REGION", "SPH_POST_BOUNDARY")
        fields = (
            "phase", "sample_count", "variance_mean", "variance_std",
            "variance_rate_mean", "ratio_mean", "ratio_max", "dwell_positive_fraction",
            "lateral_span_mean", "lateral_p90_minus_p10_mean",
            "ratio_gt_1_1_count", "ratio_gt_1_28_count", "dwell_positive_count",
        )
        with (OUTPUT / "lidar_front_trigger_phase_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for phase in phases:
                rows = [row for row in self.rows if row["evaluation_only_sph_phase"] == phase]
                variances = [float(row["lateral_variance"]) for row in rows]
                rates = [float(row["lateral_variance_rate"]) for row in rows]
                ratios = [float(row["lateral_expansion_ratio"]) for row in rows]
                spans = [float(row["lateral_span"]) for row in rows]
                robust_spans = [float(row["lateral_p90_minus_p10"]) for row in rows]
                writer.writerow({
                    "phase": phase,
                    "sample_count": len(rows),
                    "variance_mean": float(np.mean(variances)) if rows else "",
                    "variance_std": float(np.std(variances)) if rows else "",
                    "variance_rate_mean": float(np.mean(rates)) if rows else "",
                    "ratio_mean": float(np.mean(ratios)) if rows else "",
                    "ratio_max": max(ratios) if rows else "",
                    "dwell_positive_fraction": sum(row["lateral_dwell_positive"] for row in rows) / len(rows) if rows else "",
                    "lateral_span_mean": float(np.mean(spans)) if rows else "",
                    "lateral_p90_minus_p10_mean": float(np.mean(robust_spans)) if rows else "",
                    "ratio_gt_1_1_count": sum(row["lateral_ratio_gt_1_1"] for row in rows),
                    "ratio_gt_1_28_count": sum(row["lateral_ratio_gt_1_28"] for row in rows),
                    "dwell_positive_count": sum(row["lateral_dwell_positive"] for row in rows),
                })

    def _print_onset_summary(self):
        """Print first descriptive markers with evaluation-only geometry."""
        events = self._event_summary()
        print("retrospective_event_summary:")
        for key, value in events.items():
            print(f"  {key}={value if value is not None else 'not observed'}")
        for cohort in self.lateral_states:
            print(
                f"  {cohort}: ratio={self._cohort_event(cohort, 'ratio_gt_1_28')}, "
                f"dwell={self._cohort_event(cohort, 'dwell_positive')}, "
                f"sustained={self._cohort_event(cohort, 'sustained_marker')}"
            )
        markers = (
            ("lateral_ratio_gt_1_1", "ratio>1.1"),
            ("lateral_ratio_gt_1_28", "ratio>1.28"),
            ("lateral_dwell_positive", "dwell>0"),
            ("scan_change_gt_5", "scan_change>5"),
        )
        for field, label in markers:
            row = next((item for item in self.rows if item[field]), None)
            if row is None:
                print(f"{label}: not observed")
            else:
                message = (
                    f"{label}: t={row['timestamp']:.4f}, "
                    f"center_dist={row['evaluation_only_gt_distance_to_junction']:.3f}, "
                    f"boundary_dist={row['evaluation_only_gt_distance_to_opening_boundary']:.3f}, "
                    f"front_cohort_boundary={row['evaluation_only_front_cohort_center_distance_to_opening_boundary']:.3f}, "
                    f"observed_cohort_boundary={row['evaluation_only_observed_cohort_center_distance_to_opening_boundary']:.3f}, "
                    f"frontmost_boundary={row['evaluation_only_frontmost_progress_distance_to_opening_boundary']:.3f}, "
                    f"sph_phase={row['evaluation_only_sph_phase']}"
                )
                if field == "scan_change_gt_5":
                    message += (
                        f", left_free={row['cheap_lidar_left_free_fraction']}, "
                        f"right_free={row['cheap_lidar_right_free_fraction']}, "
                        f"total_free={row['cheap_lidar_free_angular_fraction']}"
                    )
                else:
                    message += (
                        f", variance={row['lateral_variance']:.3f}, "
                        f"baseline={row['lateral_baseline']:.3f}, "
                        f"ratio={row['lateral_expansion_ratio']:.3f}, "
                        f"dwell={row['lateral_expansion_dwell']:.3f}"
                    )
                print(message)

def run() -> None:
    """Run the unchanged simulator headlessly and export diagnostics."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
    os.environ.setdefault("SPH_DFS_HEADLESS_FAST", "1")
    os.environ.setdefault("SPH_DFS_MAX_FRAMES", "250")
    os.environ["SPH_DFS_ANCHOR_SHADOW"] = "1"
    os.environ["SPH_DFS_PROVISIONAL_CONFIRMATION"] = "0"
    from junction_detection.integration.anchor_pointcloud_junction_confirmation import simulate_polygon_lidar

    collector = DiagnosticCollector(simulate_polygon_lidar)
    target_name = str(TARGET)
    initial_offset_y = collector.initial_setup_offset_y

    def local_trace(frame, event, arg):
        if event == "return":
            local = frame.f_locals
            collector.sample(
                local["self"], local["robots"], local["timestamp"],
                local["dt"], frame.f_globals,
            )
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == target_name and frame.f_code.co_name == "initialize_simulation":
            return initialization_trace
        if (
            event == "call"
            and frame.f_code.co_filename == target_name
            and frame.f_code.co_name == "update"
            and "robots" in frame.f_code.co_varnames
            and "timestamp" in frame.f_code.co_varnames
        ):
            # We only need the method return snapshot; avoid tracing every
            # production line in this large simulator.
            frame.f_trace_lines = False
            return local_trace
        return None

    def initialization_trace(frame, event, arg):
        if event == "return" and isinstance(arg, tuple) and arg and isinstance(arg[0], list):
            robots = arg[0]
            # Evaluation-only setup: translate the initial swarm along the
            # known incoming-corridor axis without changing geometry, width,
            # forces, or production simulator code.
            for robot in robots:
                robot.position.y += initial_offset_y
                robot.previous_position.y += initial_offset_y
            return None
        return initialization_trace

    previous_trace = sys.gettrace()
    sys.settrace(global_trace)
    try:
        try:
            runpy.run_path(str(TARGET), run_name="__main__")
        except SystemExit:
            pass
    finally:
        sys.settrace(previous_trace)
        collector.save()
    in_front = sum(bool(r["lidar_is_in_front_cohort"]) for r in collector.rows)
    print(f"fixed_lidar_robot_id={collector.lidar_id} samples={len(collector.rows)} in_front_fraction={in_front / max(len(collector.rows), 1):.3f}")
    print(f"output={OUTPUT.resolve()}")


if __name__ == "__main__":
    run()
