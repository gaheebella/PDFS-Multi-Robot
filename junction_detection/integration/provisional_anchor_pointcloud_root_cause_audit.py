"""Evaluation-only audit of the provisional Anchor Point Cloud miss.

This reproduces the saved physical map and Anchor pose, without changing the
simulator or detector.  Ground-truth branch directions are used only for
post-hoc labelling and plotting.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.anchor_pointcloud_junction_confirmation import (
    simulate_polygon_lidar,
)
from junction_detection.pointcloud.pointcloud_junction_detector_sensor_enhanced import (
    detect_openings,
)


OUTPUT = ROOT / "junction_detection/integration/output/provisional_anchor_confirmation"


def wrap_angle(angle_deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def build_saved_cross() -> tuple[list[tuple[int, int]], int]:
    """Recreate the unchanged production benchmark polygon for audit only."""
    center_x, center_y = 400, 350
    scale = 0.70
    corridor_width = round(120 * scale)
    half_width = corridor_width // 2
    normal_length = round(180 * scale)
    right_length = normal_length * 2
    base_length = round(normal_length * (2.0 / 3.0))
    points = [
        (center_x - half_width, center_y - half_width - normal_length),
        (center_x + half_width, center_y - half_width - normal_length),
        (center_x + half_width, center_y - half_width),
        (center_x + half_width + right_length, center_y - half_width),
        (center_x + half_width + right_length, center_y + half_width),
        (center_x + half_width, center_y + half_width),
        (center_x + half_width, center_y + half_width + base_length),
        (center_x - half_width, center_y + half_width + base_length),
        (center_x - half_width, center_y + half_width),
        (center_x - half_width - normal_length, center_y + half_width),
        (center_x - half_width - normal_length, center_y - half_width),
        (center_x - half_width, center_y - half_width),
    ]
    return points, corridor_width


def run_audit() -> None:
    """Generate raw visibility metrics, replay rows, and one diagnostic plot."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    polygon, max_range = build_saved_cross()
    anchor = (398.58074613651434, 403.8301685620054)
    yaw = -89.63298601520763
    scan = simulate_polygon_lidar(
        polygon_points=polygon,
        anchor_xy=anchor,
        anchor_reference_yaw_deg=yaw,
        max_range_world_units=max_range,
    )
    openings = detect_openings(scan.angles_deg, scan.ranges)

    # World branch headings are evaluation-only labels.  The detector never
    # receives these values.
    branch_world = {"UP": -90.0, "RIGHT": 0.0, "LEFT": 180.0}
    branch_local = {
        name: wrap_angle(world - yaw) for name, world in branch_world.items()
    }
    forward = [name for name, angle in branch_local.items() if abs(angle) < 100.0]
    opening_rows = []
    for index, opening in enumerate(openings):
        opening_rows.append(
            {
                "opening_id": index,
                "center_deg": opening["center_angle"],
                "start_deg": opening["start_angle"],
                "end_deg": opening["end_angle"],
                "width_deg": opening["width_deg"],
                "confidence": opening["confidence"],
            }
        )
    with (OUTPUT / "anchor_local_scan_missing_branch_replay.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "timestamp", "opening_count", "opening_centers_deg",
                "missing_gt_branch", "anchor_x", "anchor_y",
                "max_range", "max_range_sample_count",
            ),
        )
        writer.writeheader()
        max_samples = int(np.count_nonzero(~scan.hit))
        centers = [round(float(item["center_angle"]), 3) for item in openings]
        for step in range(10):
            writer.writerow(
                {
                    "timestamp": round(1.64 + 0.1 * step, 2),
                    "opening_count": len(openings),
                    "opening_centers_deg": centers,
                    "missing_gt_branch": "LEFT/RIGHT occluded; UP shares forward opening",
                    "anchor_x": anchor[0],
                    "anchor_y": anchor[1],
                    "max_range": scan.max_range,
                    "max_range_sample_count": max_samples,
                }
            )

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(scan.angles_deg, scan.ranges, color="black", linewidth=1.0, label="raw Anchor-local scan")
    axis.axhline(scan.max_range, color="tab:gray", linestyle=":", label=f"max range={scan.max_range:.1f}")
    for index, opening in enumerate(openings):
        start, end = opening["start_angle"], opening["end_angle"]
        if start <= end:
            axis.axvspan(start, end, color="tab:orange", alpha=0.20,
                         label="detected opening" if index == 0 else None)
        else:
            axis.axvspan(start, 180, color="tab:orange", alpha=0.20)
            axis.axvspan(-180, end, color="tab:orange", alpha=0.20)
        axis.axvline(opening["center_angle"], color="tab:orange", linestyle="--")
    colors = {"UP": "tab:blue", "LEFT": "tab:red", "RIGHT": "tab:green"}
    for name, angle in branch_local.items():
        axis.axvline(angle, color=colors[name], linewidth=1.5, linestyle="-.",
                     label=f"GT {name} (evaluation-only)")
    axis.set_xlim(-180, 180)
    axis.set_xlabel("Anchor-local angle [deg]")
    axis.set_ylabel("range [pygame world unit]")
    axis.set_title("Evaluation-only: LEFT/RIGHT occluded; UP is one forward opening")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "anchor_local_scan_missing_branch_audit.png", dpi=150)
    plt.close(figure)

    distance = math.hypot(anchor[0] - 400.0, anchor[1] - 350.0)
    print(f"anchor_to_junction_center={distance:.3f}")
    print(f"opening_count={len(openings)} centers={[row['center_deg'] for row in opening_rows]}")
    print(f"gt_branch_local_angles={branch_local}")
    print(f"max_range={scan.max_range:.3f} max_range_samples={int(np.count_nonzero(~scan.hit))}")
    print(f"replay_csv={OUTPUT / 'anchor_local_scan_missing_branch_replay.csv'}")
    print(f"plot={OUTPUT / 'anchor_local_scan_missing_branch_audit.png'}")


if __name__ == "__main__":
    run_audit()
