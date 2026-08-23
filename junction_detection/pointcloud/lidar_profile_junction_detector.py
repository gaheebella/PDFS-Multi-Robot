"""Ideal geometry-based LiDAR range-profile Junction detector.

The detector is intentionally a first baseline: zero noise/dropout/occlusion
and straight corridor walls.  The corridor axis, width, and lateral offset are
all estimated in the body-local LiDAR frame.
Runtime evidence consists only of body-relative ray angles and measured
ranges. Corridor width and lateral offset are learned from local side-wall
returns; only the sensor maximum range is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from junction_detection.pointcloud.pointcloud_junction_detector import (
    _circular_runs,
    _normalize_angles,
    _validate_circular_scan,
)


@dataclass(frozen=True)
class GeometryProfileConfig:
    """Sensor and minimal grouping parameters for the ideal baseline."""

    max_range: float
    min_beam_count: int = 2
    initialization_scan_count: int = 2

    def __post_init__(self) -> None:
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if self.min_beam_count < 1:
            raise ValueError("min_beam_count must be positive")
        if self.initialization_scan_count < 1:
            raise ValueError("initialization_scan_count must be positive")


def expected_corridor_ranges_from_walls(
    angles_deg: Sequence[float], left_wall_distance: float,
    right_wall_distance: float, max_range: float,
    corridor_orientation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return expected ranges from local perpendicular side-wall distances.

    Body-relative bearing ``theta`` is converted to corridor-relative bearing
    ``alpha = theta - psi_hat``.  A ray's lateral displacement at distance
    ``d`` is ``d * sin(alpha)``.  Intersecting a side wall at half
    distance ``d_side`` gives ``d = d_side / abs(sin(alpha))``. Positive
    corridor-relative bearings use the left distance and negative bearings use
    the right distance.
    Rays whose intersection is at or beyond sensor range are excluded from
    side-opening evidence.
    """
    angles=np.asarray(angles_deg,dtype=float)
    if angles.ndim!=1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must be a finite 1D sequence")
    if left_wall_distance<=0.0 or right_wall_distance<=0.0 or max_range<=0.0:
        raise ValueError("wall distances and max_range must be positive")
    corridor_angles=_normalize_angles(angles-float(corridor_orientation_deg))
    signed_sine=np.sin(np.deg2rad(corridor_angles)); sine=np.abs(signed_sine)
    side_distance=np.where(signed_sine>=0.0,float(left_wall_distance),float(right_wall_distance))
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
        self.initial_widths=[]
        self.initial_orientations=[]
        self.stable_corridor_width=None
        self.stable_offset=None
        self.stable_corridor_orientation_deg=None
        self.model_update_count=0
        self.model_frozen=False

    @staticmethod
    def _normalize_axial_deg(angle_deg: float) -> float:
        """Normalize a 180-degree-periodic line direction to [-90, 90)."""
        return float((angle_deg+90.0)%180.0-90.0)

    @classmethod
    def _axial_difference_deg(cls, first: float, second: float) -> float:
        """Return the smallest unsigned separation of two axial directions."""
        return abs(cls._normalize_axial_deg(first-second))

    @classmethod
    def _axial_mean_deg(cls, angles_deg: Sequence[float]) -> float:
        """Average 180-degree-periodic directions with doubled angles."""
        values=np.deg2rad(np.asarray(angles_deg,dtype=float)*2.0)
        return cls._normalize_axial_deg(0.5*math.degrees(math.atan2(float(np.mean(np.sin(values))),float(np.mean(np.cos(values))))))

    @classmethod
    def _fit_wall_line(cls, points: np.ndarray) -> tuple[float,float]:
        """Fit one ideal side wall and return axial angle and perpendicular distance."""
        centroid=np.mean(points,axis=0)
        centered=points-centroid
        covariance=centered.T@centered
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        direction=eigenvectors[:,int(np.argmax(eigenvalues))]
        angle=cls._normalize_axial_deg(math.degrees(math.atan2(float(direction[1]),float(direction[0]))))
        # The 2-D cross product magnitude is origin-to-line distance because
        # the PCA direction is unit length.
        distance=abs(float(direction[0]*centroid[1]-direction[1]*centroid[0]))
        return angle,distance

    def _side_wall_observation(self, angles: np.ndarray, ranges: np.ndarray) -> dict[str,float|bool]:
        """Estimate two parallel side walls solely from body-local scan points."""
        radians=np.deg2rad(angles)
        points=np.column_stack((ranges*np.cos(radians),ranges*np.sin(radians)))
        hit=ranges<self.config.max_range-self.numerical_margin
        # Broad lateral sectors retain both walls for |psi| <= 15 degrees and
        # avoid relying on either exact body +/-90 degree ray.
        left_mask=hit&(angles>=45.0)&(angles<=135.0)
        right_mask=hit&(angles>=-135.0)&(angles<=-45.0)
        valid=np.count_nonzero(left_mask)>=2 and np.count_nonzero(right_mask)>=2
        if not valid:
            nan=float("nan")
            return {"left":nan,"right":nan,"width":nan,"offset":nan,"left_orientation":nan,"right_orientation":nan,"orientation":nan,"parallel_error":nan,"valid":False}
        left_orientation,left=self._fit_wall_line(points[left_mask])
        right_orientation,right=self._fit_wall_line(points[right_mask])
        parallel_error=self._axial_difference_deg(left_orientation,right_orientation)
        # This tolerance rejects non-parallel Junction contamination while
        # remaining only a floating-point guard in the requested noiseless run.
        parallel_valid=parallel_error<=1.0e-5
        orientation=self._axial_mean_deg((left_orientation,right_orientation))
        width=left+right if parallel_valid else float("nan")
        # Positive offset means closer to body-left wall: left shrinks and
        # right grows, hence (right-left)/2.
        offset=0.5*(right-left) if parallel_valid else float("nan")
        return {"left":left,"right":right,"width":width,"offset":offset,"left_orientation":left_orientation,"right_orientation":right_orientation,"orientation":orientation,"parallel_error":parallel_error,"valid":parallel_valid}

    @staticmethod
    def _stable_side_distances(width: float, offset: float) -> tuple[float,float]:
        """Convert width/offset model to left/right perpendicular distances."""
        return 0.5*width-offset,0.5*width+offset

    def detect(self, angles_deg: Sequence[float], ranges: Sequence[float]) -> dict[str,Any]:
        """Return opening groups and diagnostic arrays from one moving scan."""
        angles,measured,steps=_validate_circular_scan(angles_deg,ranges)
        if np.any(measured>self.config.max_range+self.numerical_margin):
            raise ValueError("range exceeds configured sensor max_range")
        observation=self._side_wall_observation(angles,measured)
        left=float(observation["left"]); right=float(observation["right"])
        width_observation=float(observation["width"]); offset_observation=float(observation["offset"])
        orientation_observation=float(observation["orientation"])
        side_walls_valid=bool(observation["valid"])
        initialized=self.stable_corridor_width is not None
        width_consistent=initialized and side_walls_valid and abs(width_observation-self.stable_corridor_width)<=self.numerical_margin*16.0
        orientation_consistent=initialized and side_walls_valid and self._axial_difference_deg(orientation_observation,self.stable_corridor_orientation_deg)<=1.0e-5
        if initialized:
            # Current offset may compensate local lateral motion only when both
            # walls still reproduce the frozen stable width. Width itself is
            # never changed before detection.
            offset_for_detection=offset_observation if width_consistent and orientation_consistent else self.stable_offset
            expected_left,expected_right=self._stable_side_distances(self.stable_corridor_width,offset_for_detection)
            expected,valid=expected_corridor_ranges_from_walls(angles,expected_left,expected_right,self.config.max_range,self.stable_corridor_orientation_deg)
        elif side_walls_valid:
            expected,valid=expected_corridor_ranges_from_walls(angles,left,right,self.config.max_range,orientation_observation)
        else:
            expected=np.full(angles.shape,self.config.max_range); valid=np.zeros(angles.shape,dtype=bool)
        delta=measured-expected
        candidates=valid&(delta>self.numerical_margin)
        groups=[]; confirmed=np.zeros(candidates.shape,dtype=bool)
        for run in _circular_runs(candidates):
            if len(run)<self.config.min_beam_count:
                continue
            width=_run_width(run,steps)
            start_body=float(_normalize_angles(angles[int(run[0])]-0.5*steps[(int(run[0])-1)%len(steps)]))
            end_body=float(_normalize_angles(angles[int(run[-1])]+0.5*steps[int(run[-1])]))
            center_body=float(_normalize_angles(start_body+0.5*width))
            orientation_for_group=(self.stable_corridor_orientation_deg if initialized else orientation_observation)
            start=float(_normalize_angles(start_body-orientation_for_group))
            end=float(_normalize_angles(end_body-orientation_for_group))
            center=float(_normalize_angles(center_body-orientation_for_group))
            confirmed[run]=True
            groups.append({
                "group_id":len(groups),"start_angle_deg":start,"end_angle_deg":end,
                "center_angle_deg":center,"angular_width_deg":width,"beam_count":int(len(run)),
                "start_body_angle_deg":start_body,"end_body_angle_deg":end_body,
                "center_body_angle_deg":center_body,
                "mean_range":float(np.mean(measured[run])),"max_range":float(np.max(measured[run])),
                "mean_delta_range":float(np.mean(delta[run])),"max_delta_range":float(np.max(delta[run])),
            })
        groups.sort(key=lambda item:item["center_angle_deg"])
        for group_id,group in enumerate(groups): group["group_id"]=group_id
        detected=bool(groups)
        candidate_count=int(np.count_nonzero(candidates))
        opening_evidence=candidate_count>0
        update_enabled=False; model_just_initialized=False
        # DETECT FIRST, UPDATE SECOND: any current opening evidence freezes the
        # existing model before this scan can affect its width.
        if initialized and opening_evidence:
            self.model_frozen=True
        elif not self.model_frozen and side_walls_valid and not opening_evidence:
            if not initialized:
                self.initial_widths.append(width_observation)
                self.initial_orientations.append(orientation_observation)
                if len(self.initial_widths)>=self.config.initialization_scan_count:
                    self.stable_corridor_width=float(np.median(self.initial_widths))
                    self.stable_offset=float(offset_observation)
                    self.stable_corridor_orientation_deg=self._axial_mean_deg(self.initial_orientations)
                    self.model_update_count+=1; model_just_initialized=True; update_enabled=True
            elif width_consistent and orientation_consistent:
                count=self.model_update_count
                self.stable_corridor_width=(self.stable_corridor_width*count+width_observation)/(count+1)
                self.stable_offset=float(offset_observation)
                self.stable_corridor_orientation_deg=self._axial_mean_deg((self.stable_corridor_orientation_deg,orientation_observation))
                self.model_update_count+=1; update_enabled=True
        initialized=self.stable_corridor_width is not None
        stable_left,stable_right=(self._stable_side_distances(self.stable_corridor_width,self.stable_offset) if initialized else (float("nan"),float("nan")))
        valid_abs_delta=np.abs(delta[valid])
        return {
            "profile_detector_state":"LIDAR_PROFILE_JUNCTION_DETECTED" if detected else "LIDAR_PROFILE_CLEAR",
            "profile_junction_detected":detected,"opening_group_count":len(groups),"opening_groups":groups,
            "expected_ranges":expected,"delta_ranges":delta,"valid_angle_mask":valid,
            "open_candidate_mask":candidates,"confirmed_opening_mask":confirmed,
            "left_wall_range":left,"right_wall_range":right,"width_observation":width_observation,"offset_observation":offset_observation,
            "estimated_corridor_width":float(self.stable_corridor_width) if initialized else float("nan"),
            "estimated_offset":float(self.stable_offset) if initialized else float("nan"),
            "current_corridor_orientation_deg":orientation_observation,
            "stable_corridor_orientation_deg":float(self.stable_corridor_orientation_deg) if initialized else float("nan"),
            "left_wall_orientation_deg":float(observation["left_orientation"]),
            "right_wall_orientation_deg":float(observation["right_orientation"]),
            "parallel_error_deg":float(observation["parallel_error"]),
            "orientation_initialized":initialized,"orientation_frozen":self.model_frozen,
            "stable_left_wall":stable_left,"stable_right_wall":stable_right,"side_walls_valid":side_walls_valid,
            "corridor_model_initialized":initialized,"corridor_model_just_initialized":model_just_initialized,
            "corridor_model_update_enabled":update_enabled,"corridor_model_frozen":self.model_frozen,"corridor_model_update_count":self.model_update_count,
            "opening_candidate_count":candidate_count,"profile_max_range":self.config.max_range,
            "profile_numerical_margin":self.numerical_margin,
            "profile_max_abs_valid_delta":float(np.max(valid_abs_delta)) if len(valid_abs_delta) else 0.0,
            "measured_profile_min":float(np.min(measured)),"measured_profile_mean":float(np.mean(measured)),"measured_profile_max":float(np.max(measured)),
            "expected_profile_min":float(np.min(expected)),"expected_profile_mean":float(np.mean(expected)),"expected_profile_max":float(np.max(expected)),
        }
