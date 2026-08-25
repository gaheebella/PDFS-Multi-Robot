"""Moving LiDAR 55% GUI with an EXP-042 pre-corridor start alias.

Only the initial evaluation geometry/pose differs for
``M1_PRE_CORRIDOR_55PCT``.  Rendering, local-forward simulation, LiDAR
raycasting, and the frozen adaptive 55% detector are reused unchanged from
the existing read-only threshold visualizer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_detection.integration.run_lidar_local_corridor_estimation import (
    REAR_START_SHIFT,
    _rear_start,
)
from pygame_simulator import lidar_junction_detection_threshold_visualizer as _base


M0_CASE = "M0_STRAIGHT"
M1_BASELINE_CASE = "M1_CROSS_BASELINE"
M1_PRE_CORRIDOR_CASE = "M1_PRE_CORRIDOR_55PCT"
MAP_CASES = (M0_CASE, M1_BASELINE_CASE, M1_PRE_CORRIDOR_CASE)


class AdaptiveSession(_base.AdaptiveSession):
    """Apply the validated rear-start setup before the first physics step."""

    def __init__(self, map_case: str) -> None:
        runner_case = (
            M1_BASELINE_CASE
            if map_case == M1_PRE_CORRIDOR_CASE
            else map_case
        )
        super().__init__(runner_case)
        if map_case == M1_PRE_CORRIDOR_CASE:
            _rear_start(self.runner, REAR_START_SHIFT)
        self.map_case = map_case


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-case",
        choices=MAP_CASES,
        default=M1_PRE_CORRIDOR_CASE,
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="0: GUI until ESC; headless/validate default 240",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--pause-on-detect", action="store_true")
    parser.add_argument(
        "--show-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="headless M0/baseline/pre-corridor plus deterministic replay",
    )
    parser.add_argument("--output-dir", type=Path, default=_base.DEFAULT_OUTPUT)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if (args.headless or args.validate) and args.frames == 0:
        args.frames = 240
    return args


def main(argv: list[str] | None = None) -> None:
    # The reused GUI resolves these names from its module at runtime.  Replacing
    # only the session factory/case catalog keeps all detector and renderer
    # behavior in the existing implementation while adding this start alias.
    _base.AdaptiveSession = AdaptiveSession
    _base.MAP_CASES = MAP_CASES
    _base.parse_args = parse_args
    _base.EXPERIMENT_NAME = (
        "Moving LiDAR Adaptive 55% Junction Detection GUI — pre-corridor"
    )
    _base.main(argv)


if __name__ == "__main__":
    main()
