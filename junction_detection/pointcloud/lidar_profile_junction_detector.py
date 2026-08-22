"""Ideal geometry-based LiDAR range-profile Junction detector.

The detector is intentionally a first baseline: zero noise/dropout/occlusion,
constant corridor width, and a LiDAR near the corridor centerline. Runtime
evidence consists only of body-relative ray angles and measured ranges. The
known corridor width and sensor maximum range are configuration, not map pose
or Junction metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from junction_detection.pointcloud.pointcloud_junction_detector import (
    _circular_runs,
    _normalize_angles,
    _validate_circular_scan,
)


@dataclass(frozen=True)
class GeometryProfileConfig:
    """Known ideal-corridor and sensor parameters for the baseline."""

    corridor_width: float
    max_range: float
    min_beam_count: int = 2

    def __post_init__(self) -> None:
        if self.corridor_width <= 0.0 or self.max_range <= 0.0:
            raise ValueError("corridor_width and max_range must be positive")
        if self.min_beam_count < 1:
            raise ValueError("min_beam_count must be positive")


def expected_corridor_ranges(
    angles_deg: Sequence[float], corridor_width: float, max_range: float,
    lateral_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal centerline side-wall ranges and their valid-angle mask.

    For body-relative bearing ``theta``, a ray's lateral displacement at
    distance ``d`` is ``d * sin(theta)``.  Intersecting a side wall at half
    width ``h`` gives ``d = h / abs(sin(theta))``. Rays whose intersection is
    at or beyond sensor range are excluded from side-opening evidence; this
    removes ordinary forward/backward free corridors without a direction- or
    map-specific rule.
    """
    angles=np.asarray(angles_deg,dtype=float)
    if angles.ndim!=1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must be a finite 1D sequence")
    if corridor_width<=0.0 or max_range<=0.0:
        raise ValueError("corridor_width and max_range must be positive")
    half_width=0.5*float(corridor_width)
    if abs(lateral_offset)>=half_width:
        raise ValueError("lateral_offset must remain inside the corridor")
    signed_sine=np.sin(np.deg2rad(angles)); sine=np.abs(signed_sine)
    side_distance=np.where(signed_sine>=0.0,half_width-lateral_offset,half_width+lateral_offset)
    expected=np.full(angles.shape,float(max_range))
    non_singular=sine>np.finfo(float).eps*32.0
    expected[non_singular]=np.minimum(float(max_range),side_distance[non_singular]/sine[non_singular])
    # Equality at max range cannot distinguish a missing wall from saturation.
    valid=expected < float(max_range)-np.finfo(float).eps*max(1.0,float(max_range))*64.0
    return expected,valid


def _run_width(run: np.ndarray, steps: np.ndarray) -> float:
    """Return circular angular support width for a connected run."""
    return float(np.sum(steps[run]))


class LidarProfileJunctionDetector:
    """Detect additional side free-space groups in an ideal corridor scan."""

    RUNTIME_INPUTS=("angles_deg","ranges")

    def __init__(self, config: GeometryProfileConfig):
        self.config=config
        # Matches the existing wall/ray intersection tolerance scale. This is
        # numerical protection, not a noise or map-tuned robustness margin.
        self.numerical_margin=max(1.0e-8,np.finfo(float).eps*config.max_range*128.0)
        self.lateral_offset=None

    def _observable_centerline_offset(self, angles: np.ndarray, ranges: np.ndarray) -> float | None:
        """Estimate local offset when both lateral corridor walls are visible.

        The ideal formula uses zero offset. The clean simulator's discrete grid
        puts its closest front robot 1.4 units off center, so the two lateral
        beams locally recover that offset without a global pose. Their summed
        ranges must agree with the configured corridor width.
        """
        positive=int(np.argmin(np.abs(angles-90.0))); negative=int(np.argmin(np.abs(angles+90.0)))
        positive_range=float(ranges[positive]); negative_range=float(ranges[negative])
        width_error=abs((positive_range+negative_range)-self.config.corridor_width)
        if width_error>self.numerical_margin*16.0:
            return None
        return 0.5*(negative_range-positive_range)

    def detect(self, angles_deg: Sequence[float], ranges: Sequence[float]) -> dict[str,Any]:
        """Return opening groups and diagnostic arrays from one moving scan."""
        angles,measured,steps=_validate_circular_scan(angles_deg,ranges)
        if np.any(measured>self.config.max_range+self.numerical_margin):
            raise ValueError("range exceeds configured sensor max_range")
        observed_offset=self._observable_centerline_offset(angles,measured)
        if observed_offset is not None:
            self.lateral_offset=observed_offset
        elif self.lateral_offset is None:
            # Explicit simplifying assumption if startup does not expose both
            # lateral walls; no global pose is consulted.
            self.lateral_offset=0.0
        expected,valid=expected_corridor_ranges(angles,self.config.corridor_width,self.config.max_range,self.lateral_offset)
        delta=measured-expected
        candidates=valid&(delta>self.numerical_margin)
        groups=[]; confirmed=np.zeros(candidates.shape,dtype=bool)
        for run in _circular_runs(candidates):
            if len(run)<self.config.min_beam_count:
                continue
            width=_run_width(run,steps)
            start=float(_normalize_angles(angles[int(run[0])]-0.5*steps[(int(run[0])-1)%len(steps)]))
            end=float(_normalize_angles(angles[int(run[-1])]+0.5*steps[int(run[-1])]))
            center=float(_normalize_angles(start+0.5*width))
            confirmed[run]=True
            groups.append({
                "group_id":len(groups),"start_angle_deg":start,"end_angle_deg":end,
                "center_angle_deg":center,"angular_width_deg":width,"beam_count":int(len(run)),
                "mean_range":float(np.mean(measured[run])),"max_range":float(np.max(measured[run])),
                "mean_delta_range":float(np.mean(delta[run])),"max_delta_range":float(np.max(delta[run])),
            })
        groups.sort(key=lambda item:item["center_angle_deg"])
        for group_id,group in enumerate(groups): group["group_id"]=group_id
        detected=bool(groups)
        valid_abs_delta=np.abs(delta[valid])
        return {
            "profile_detector_state":"LIDAR_PROFILE_JUNCTION_DETECTED" if detected else "LIDAR_PROFILE_CLEAR",
            "profile_junction_detected":detected,"opening_group_count":len(groups),"opening_groups":groups,
            "expected_ranges":expected,"delta_ranges":delta,"valid_angle_mask":valid,
            "open_candidate_mask":candidates,"confirmed_opening_mask":confirmed,
            "corridor_width":self.config.corridor_width,"profile_max_range":self.config.max_range,
            "profile_lateral_offset":float(self.lateral_offset),
            "profile_numerical_margin":self.numerical_margin,
            "profile_max_abs_valid_delta":float(np.max(valid_abs_delta)) if len(valid_abs_delta) else 0.0,
            "measured_profile_min":float(np.min(measured)),"measured_profile_mean":float(np.mean(measured)),"measured_profile_max":float(np.max(measured)),
            "expected_profile_min":float(np.min(expected)),"expected_profile_mean":float(np.mean(expected)),"expected_profile_max":float(np.max(expected)),
        }
