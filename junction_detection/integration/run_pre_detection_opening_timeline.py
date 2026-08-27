"""Export the frozen M1 LiDAR detector's pre-detection opening timeline.

This diagnostic is a read-only consumer of ``LidarProfileJunctionDetector``
output. It does not reproduce or alter thresholding, grouping, expected-range
calculation, corridor estimation, or the simulator's LiDAR scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdfs_mpl_cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.pointcloud.lidar_profile_junction_detector import (  # noqa: E402
    GeometryProfileConfig,
    LidarProfileJunctionDetector,
)
from pygame_simulator.pre_exploration_general_pipeline_simulator import (  # noqa: E402
    LIDAR_MAX_RANGE,
    SimulationRunner,
)


MAP_CASE = "M1_CROSS_BASELINE"
START_FRAME = 60
END_FRAME = 126
DEFAULT_OUTPUT = (
    ROOT
    / "junction_detection/integration/output/pre_detection_opening_timeline"
    / "pre_detection_opening_timeline.csv"
)
FIELDS = (
    "frame",
    "time",
    "open_candidate_count",
    "opening_group_count",
    "group_start_deg",
    "group_end_deg",
    "group_width_deg",
    "junction_detected",
)


def run_timeline() -> list[dict[str, Any]]:
    """Return one row per sampled LiDAR frame from frame 60 through 126.

    Group-valued columns are compact JSON arrays so both M1 side openings fit
    in one frame row without dropping or duplicating detector evidence.
    """
    detector = LidarProfileJunctionDetector(GeometryProfileConfig(LIDAR_MAX_RANGE))
    runner = SimulationRunner(
        MAP_CASE,
        "local_forward",
        profile_detector=detector,
        hold_on_profile_detection=False,
    )
    output: list[dict[str, Any]] = []
    for frame in range(END_FRAME + 1):
        sampled = runner.step(frame)
        if sampled is None or frame < START_FRAME:
            continue
        result = runner.last_profile_result
        groups = result["opening_groups"]
        output.append(
            {
                "frame": frame,
                "time": float(sampled["timestamp"]),
                "open_candidate_count": int(result["opening_candidate_count"]),
                "opening_group_count": int(result["opening_group_count"]),
                "group_start_deg": _group_values(groups, "start_angle_deg"),
                "group_end_deg": _group_values(groups, "end_angle_deg"),
                "group_width_deg": _group_values(groups, "angular_width_deg"),
                "junction_detected": bool(result["profile_junction_detected"]),
            }
        )
    return output


def _group_values(groups: list[dict[str, Any]], key: str) -> str:
    """Serialize unchanged detector group values as a compact JSON array."""
    return json.dumps([float(group[key]) for group in groups], separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _first_frame(rows: list[dict[str, Any]], key: str) -> int | None:
    return next((int(row["frame"]) for row in rows if row[key]), None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = run_timeline()
    write_csv(args.output, rows)
    print(
        f"output={args.output.resolve()} rows={len(rows)} "
        f"first_candidate_frame={_first_frame(rows, 'open_candidate_count')} "
        f"first_group_frame={_first_frame(rows, 'opening_group_count')} "
        f"first_detection_frame={_first_frame(rows, 'junction_detected')}"
    )


if __name__ == "__main__":
    main()
