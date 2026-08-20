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
OUTPUT = ROOT / "junction_detection/integration/output/lidar_front_trigger_diagnostics"


def _cross_geometry() -> tuple[list[tuple[int, int]], float, tuple[float, float]]:
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
    return points, float(width), center


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
        self.previous_variance: float | None = None
        self.diagnostic_baseline = 0.0
        self.diagnostic_dwell = 0.0
        self.front_maintenance_dwell = 0.0
        self.last_sample_timestamp: float | None = None
        self.sample_interval_s = 0.1
        self.polygon, self.max_range, self.gt_center = _cross_geometry()

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

    def _lateral_features(self, observed, forward, globals_dict, timestamp):
        """Recompute lateral spread independently of AnchorShadow lifecycle."""
        if not observed:
            return 0.0, self.diagnostic_baseline, 0.0, 1.0, self.diagnostic_dwell, 0.0
        center = sum((r.position for r in observed), type(observed[0].position)()) / len(observed)
        lateral = type(forward)(-forward.y, forward.x)
        values = [(r.position - center).dot(lateral) for r in observed]
        variance = float(sum(v * v for v in values) / len(values))
        min_delta = float(globals_dict.get("JUNCTION_LATERAL_EXPANSION_MIN", (4.5 * 0.70) ** 2))
        ratio_limit = float(globals_dict.get("JUNCTION_LATERAL_EXPANSION_RATIO", 1.28))
        alpha = float(globals_dict.get("JUNCTION_BASELINE_ALPHA", 0.035))
        if self.diagnostic_baseline <= 1e-12:
            self.diagnostic_baseline = max(variance, min_delta)
        delta = variance - self.diagnostic_baseline
        ratio = variance / max(self.diagnostic_baseline, 1e-12)
        expanding = delta >= min_delta and ratio >= ratio_limit
        dt = 0.0 if not self.rows else max(timestamp - self.rows[-1]["timestamp"], 1e-12)
        self.diagnostic_dwell = self.diagnostic_dwell + dt if expanding else 0.0
        if not expanding and variance < self.diagnostic_baseline:
            self.diagnostic_baseline += alpha * (variance - self.diagnostic_baseline)
        rate = 0.0 if self.previous_variance is None else (variance - self.previous_variance) / max(dt, 1e-12)
        self.previous_variance = variance
        return variance, self.diagnostic_baseline, delta, ratio, self.diagnostic_dwell, rate

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
        variance, baseline, variance_delta, expansion_ratio, expansion_dwell, variance_rate = self._lateral_features(
            observed, forward, globals_dict, timestamp
        )
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
            "cheap_lidar_left_range": float(np.mean(left_ranges)),
            "cheap_lidar_right_range": float(np.mean(right_ranges)),
            "cheap_lidar_left_free_fraction": float(np.mean(~scan.hit[left])),
            "cheap_lidar_right_free_fraction": float(np.mean(~scan.hit[right])),
            "cheap_lidar_scan_change": scan_change,
            "cheap_lidar_free_angular_fraction": float(np.mean(~scan.hit)),
            "evaluation_only_gt_distance_to_junction": math.hypot(
                lidar.position.x - self.gt_center[0], lidar.position.y - self.gt_center[1]
            ),
        }
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
        time = np.asarray([r["timestamp"] for r in self.rows])
        variance = np.asarray([r["lateral_variance"] for r in self.rows])
        scan_change = np.asarray([float(r["cheap_lidar_scan_change"] or 0.0) for r in self.rows])
        gt_distance = np.asarray([r["evaluation_only_gt_distance_to_junction"] for r in self.rows])
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(time, variance, label="lateral variance")
        axes[0].plot(time, [r["lateral_baseline"] for r in self.rows], label="adaptive baseline")
        axes[0].legend(); axes[0].set_ylabel("SPH spread")
        axes[1].plot(time, scan_change, color="tab:orange", label="cheap LiDAR scan change")
        axes[1].legend(); axes[1].set_ylabel("range change")
        axes[2].plot(time, gt_distance, color="tab:gray", label="GT distance (evaluation-only)")
        axes[2].legend(); axes[2].set_ylabel("distance"); axes[2].set_xlabel("time [s]")
        figure.suptitle("Fixed hardware LiDAR/front-cohort diagnostic")
        figure.tight_layout()
        figure.savefig(OUTPUT / "lidar_front_trigger_timeline.png", dpi=150)
        plt.close(figure)


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

    def local_trace(frame, event, arg):
        if event == "return":
            local = frame.f_locals
            collector.sample(
                local["self"], local["robots"], local["timestamp"],
                local["dt"], frame.f_globals,
            )
        return local_trace

    def global_trace(frame, event, arg):
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
