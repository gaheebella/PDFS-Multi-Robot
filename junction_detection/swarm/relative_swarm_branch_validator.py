"""Validate Branch candidates from Anchor-relative swarm behavior.

Algorithm-visible information is deliberately restricted to local relative
measurements and robot-to-robot neighbor IDs.  No global position, map, wall
geometry, known Branch count, or known Branch direction appears in the public
data model or validator interface.

The module does not use LiDAR opening thresholds.  It identifies robots whose
Anchor-relative range has a statistically significant positive temporal slope,
forms connected components in their neighbor graph, and estimates one Branch
direction per multi-robot progressing cohort using circular statistics.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite, log, sqrt
from typing import Hashable, Iterable, Literal, Optional, Sequence

import numpy as np


RobotId = Hashable
MotionState = Literal["progressing", "non_progressing", "uncertain"]


@dataclass(frozen=True)
class RobotObservation:
    """One robot's algorithm-visible state at one local timestamp.

    ``anchor_range_m`` and ``anchor_bearing_deg`` are relative to the stationary
    Anchor. There are intentionally no global x/y or global heading fields.
    """

    timestamp: float
    robot_id: RobotId
    anchor_range_m: float
    anchor_bearing_deg: float
    neighbor_ids: tuple[RobotId, ...] = ()
    radial_velocity_mps: Optional[float] = None
    speed_mps: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate finite local measurements and canonicalize neighbor IDs."""
        if not isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        try:
            hash(self.robot_id)
        except TypeError as error:
            raise ValueError("robot_id must be hashable") from error
        if not isfinite(float(self.anchor_range_m)) or self.anchor_range_m < 0.0:
            raise ValueError("anchor_range_m must be finite and non-negative")
        if not isfinite(float(self.anchor_bearing_deg)):
            raise ValueError("anchor_bearing_deg must be finite")
        if self.radial_velocity_mps is not None and not isfinite(
            float(self.radial_velocity_mps)
        ):
            raise ValueError("radial_velocity_mps must be finite when supplied")
        if self.speed_mps is not None:
            if not isfinite(float(self.speed_mps)) or self.speed_mps < 0.0:
                raise ValueError("speed_mps must be finite and non-negative")
        neighbors = tuple(dict.fromkeys(self.neighbor_ids))
        if self.robot_id in neighbors:
            raise ValueError("a robot cannot list itself as a neighbor")
        for neighbor_id in neighbors:
            try:
                hash(neighbor_id)
            except TypeError as error:
                raise ValueError("neighbor IDs must be hashable") from error
        object.__setattr__(self, "neighbor_ids", neighbors)
        object.__setattr__(
            self,
            "anchor_bearing_deg",
            _normalize_angle_deg(float(self.anchor_bearing_deg)),
        )


@dataclass(frozen=True)
class RobotTrend:
    """Temporal radial-motion statistics for one robot."""

    robot_id: RobotId
    state: MotionState
    observation_count: int
    window_start_s: float
    window_end_s: float
    radial_slope_mps: float
    slope_standard_error_mps: float
    slope_test_statistic: float
    slope_ci_low_mps: float
    slope_ci_high_mps: float
    residual_rmse_m: float
    latest_range_m: float
    latest_bearing_deg: float


@dataclass(frozen=True)
class BranchCandidate:
    """One connected progressing cohort and its observed statistics."""

    cohort_id: int
    member_robot_ids: tuple[RobotId, ...]
    robot_count: int
    estimated_direction_deg: float
    circular_bearing_spread_deg: float
    mean_resultant_length: float
    mean_radial_slope_mps: float
    min_slope_ci_low_mps: float
    range_min_m: float
    range_max_m: float


@dataclass(frozen=True)
class ValidationResult:
    """Branch candidates plus explicit radial-trend diagnostics."""

    timestamp: float
    progressing_robot_ids: tuple[RobotId, ...]
    non_progressing_robot_ids: tuple[RobotId, ...]
    uncertain_robot_ids: tuple[RobotId, ...]
    progressing_components: tuple[tuple[RobotId, ...], ...]
    rejected_progressing_components: tuple[tuple[RobotId, ...], ...]
    branch_candidates: tuple[BranchCandidate, ...]
    trends: tuple[RobotTrend, ...]
    unavailable_neighbor_references: tuple[tuple[RobotId, RobotId], ...]


def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def circular_mean_and_spread_deg(
    angles_deg: Sequence[float],
) -> tuple[float, float, float]:
    """Return circular mean, circular standard deviation, and resultant length."""
    values = np.asarray(angles_deg, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("angles_deg must be a non-empty finite 1D sequence")
    radians = np.deg2rad(values)
    mean_cos = float(np.mean(np.cos(radians)))
    mean_sin = float(np.mean(np.sin(radians)))
    resultant = hypot(mean_cos, mean_sin)
    mean_angle = _normalize_angle_deg(degrees(atan2(mean_sin, mean_cos)))
    if resultant <= np.finfo(float).eps:
        spread_deg = 180.0
    else:
        spread_deg = min(180.0, degrees(sqrt(max(0.0, -2.0 * log(resultant)))))
    return mean_angle, spread_deg, min(1.0, resultant)


class RelativeSwarmBranchValidator:
    """Stateful localization-free radial-progress cohort validator.

    Parameters are explicit because temporal inference cannot be defined without
    an observation horizon, sample support, and significance level. Defaults are
    experimental validation settings, not claimed physical constants.

    A robot is ``progressing`` when the lower confidence bound of its OLS radial
    slope is above zero. It is ``non_progressing`` when that interval contains
    zero, which means positive progress was not statistically confirmed—not that
    the robot was proven to be physically stopped at a wall. A significantly
    negative slope and insufficient history remain ``uncertain``.

    ``confidence_multiplier=1.96`` is a normal-approximation multiplier used for
    synthetic validation. It does not claim exact 95% coverage for small samples;
    Student-t, bootstrap, or empirical calibration should be evaluated on real
    SPH/robot data. OLS is retained because it handles irregular timestamps and
    provides interpretable slope uncertainty without SciPy or a velocity cutoff.

    All defaults are experimental synthetic-validation settings, not claimed
    physical constants.
    """

    def __init__(
        self,
        *,
        temporal_window_s: float = 4.0,
        minimum_observations: int = 5,
        confidence_multiplier: float = 1.96,
        minimum_cohort_size: int = 2,
    ) -> None:
        if not isfinite(float(temporal_window_s)) or temporal_window_s <= 0.0:
            raise ValueError("temporal_window_s must be finite and positive")
        if minimum_observations < 3:
            raise ValueError("minimum_observations must be at least 3 for slope uncertainty")
        if not isfinite(float(confidence_multiplier)) or confidence_multiplier <= 0.0:
            raise ValueError("confidence_multiplier must be finite and positive")
        if minimum_cohort_size < 2:
            raise ValueError("minimum_cohort_size must be at least 2")
        self.temporal_window_s = float(temporal_window_s)
        self.minimum_observations = int(minimum_observations)
        self.confidence_multiplier = float(confidence_multiplier)
        self.minimum_cohort_size = int(minimum_cohort_size)
        self._history: dict[RobotId, deque[RobotObservation]] = defaultdict(deque)
        self._last_frame_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Clear all temporal histories."""
        self._history.clear()
        self._last_frame_timestamp = None

    def update(self, observations: Sequence[RobotObservation]) -> ValidationResult:
        """Consume one timestamp frame and return current cohort validation.

        All observations in a frame must share one timestamp and unique robot
        IDs. Robots missing from a frame retain history but cannot join its graph.
        Neighbor references to missing robots are ignored and reported.
        """
        frame = tuple(observations)
        if not frame:
            raise ValueError("an observation frame cannot be empty")
        timestamp = float(frame[0].timestamp)
        if any(not np.isclose(obs.timestamp, timestamp) for obs in frame):
            raise ValueError("all observations in one frame must share a timestamp")
        if self._last_frame_timestamp is not None and timestamp <= self._last_frame_timestamp:
            raise ValueError("frame timestamps must be strictly increasing")
        frame_by_id: dict[RobotId, RobotObservation] = {}
        for observation in frame:
            if observation.robot_id in frame_by_id:
                raise ValueError(f"duplicate robot_id in frame: {observation.robot_id!r}")
            frame_by_id[observation.robot_id] = observation

        cutoff = timestamp - self.temporal_window_s
        for observation in frame:
            history = self._history[observation.robot_id]
            if history and observation.timestamp <= history[-1].timestamp:
                raise ValueError(
                    f"non-increasing timestamp for robot {observation.robot_id!r}"
                )
            history.append(observation)
            while history and history[0].timestamp < cutoff:
                history.popleft()
        self._last_frame_timestamp = timestamp

        trends = tuple(
            self._estimate_trend(robot_id, frame_by_id[robot_id])
            for robot_id in _sorted_robot_ids(frame_by_id)
        )
        trends_by_id = {trend.robot_id: trend for trend in trends}
        progressing = tuple(
            trend.robot_id for trend in trends if trend.state == "progressing"
        )
        non_progressing = tuple(
            trend.robot_id for trend in trends if trend.state == "non_progressing"
        )
        uncertain = tuple(
            trend.robot_id for trend in trends if trend.state == "uncertain"
        )

        graph, unavailable = _build_progressing_graph(frame_by_id, set(progressing))
        components = _connected_components(graph)
        accepted = tuple(
            component
            for component in components
            if len(component) >= self.minimum_cohort_size
        )
        rejected = tuple(
            component
            for component in components
            if len(component) < self.minimum_cohort_size
        )
        candidates = tuple(
            self._make_candidate(index, component, frame_by_id, trends_by_id)
            for index, component in enumerate(accepted)
        )
        return ValidationResult(
            timestamp=timestamp,
            progressing_robot_ids=progressing,
            non_progressing_robot_ids=non_progressing,
            uncertain_robot_ids=uncertain,
            progressing_components=components,
            rejected_progressing_components=rejected,
            branch_candidates=candidates,
            trends=trends,
            unavailable_neighbor_references=unavailable,
        )

    def _estimate_trend(
        self, robot_id: RobotId, latest: RobotObservation
    ) -> RobotTrend:
        """Fit range = intercept + slope*time and classify its confidence interval."""
        history = tuple(self._history[robot_id])
        count = len(history)
        if count < self.minimum_observations:
            return _insufficient_trend(robot_id, latest, history)

        timestamps = np.asarray([obs.timestamp for obs in history], dtype=float)
        ranges = np.asarray([obs.anchor_range_m for obs in history], dtype=float)
        centered_time = timestamps - float(np.mean(timestamps))
        sum_squared_time = float(np.dot(centered_time, centered_time))
        numerical_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(timestamps))))
        if sum_squared_time <= numerical_floor:
            return _insufficient_trend(robot_id, latest, history)

        centered_range = ranges - float(np.mean(ranges))
        slope = float(np.dot(centered_time, centered_range) / sum_squared_time)
        intercept = float(np.mean(ranges) - slope * np.mean(timestamps))
        residuals = ranges - (intercept + slope * timestamps)
        residual_sum_squares = float(np.dot(residuals, residuals))
        residual_variance = residual_sum_squares / (count - 2)
        slope_standard_error = sqrt(max(0.0, residual_variance / sum_squared_time))
        residual_rmse = sqrt(max(0.0, residual_sum_squares / count))
        slope_floor = np.finfo(float).eps * max(1.0, abs(slope)) * 32.0

        if slope_standard_error <= slope_floor:
            if slope > slope_floor:
                state: MotionState = "progressing"
                test_statistic = float("inf")
            elif slope < -slope_floor:
                state = "uncertain"
                test_statistic = float("-inf")
            else:
                state = "non_progressing"
                test_statistic = 0.0
            ci_low = slope
            ci_high = slope
        else:
            test_statistic = slope / slope_standard_error
            ci_low = slope - self.confidence_multiplier * slope_standard_error
            ci_high = slope + self.confidence_multiplier * slope_standard_error
            if ci_low > 0.0:
                state = "progressing"
            elif ci_low <= 0.0 <= ci_high:
                state = "non_progressing"
            else:
                state = "uncertain"

        return RobotTrend(
            robot_id=robot_id,
            state=state,
            observation_count=count,
            window_start_s=float(timestamps[0]),
            window_end_s=float(timestamps[-1]),
            radial_slope_mps=slope,
            slope_standard_error_mps=slope_standard_error,
            slope_test_statistic=test_statistic,
            slope_ci_low_mps=ci_low,
            slope_ci_high_mps=ci_high,
            residual_rmse_m=residual_rmse,
            latest_range_m=float(latest.anchor_range_m),
            latest_bearing_deg=float(latest.anchor_bearing_deg),
        )

    @staticmethod
    def _make_candidate(
        cohort_id: int,
        component: tuple[RobotId, ...],
        frame_by_id: dict[RobotId, RobotObservation],
        trends_by_id: dict[RobotId, RobotTrend],
    ) -> BranchCandidate:
        """Summarize one connected cohort without arbitrary confidence weights."""
        bearings = [frame_by_id[robot_id].anchor_bearing_deg for robot_id in component]
        ranges = [frame_by_id[robot_id].anchor_range_m for robot_id in component]
        slopes = [trends_by_id[robot_id].radial_slope_mps for robot_id in component]
        lower_bounds = [trends_by_id[robot_id].slope_ci_low_mps for robot_id in component]
        direction, spread, resultant = circular_mean_and_spread_deg(bearings)
        return BranchCandidate(
            cohort_id=cohort_id,
            member_robot_ids=component,
            robot_count=len(component),
            estimated_direction_deg=direction,
            circular_bearing_spread_deg=spread,
            mean_resultant_length=resultant,
            mean_radial_slope_mps=float(np.mean(slopes)),
            min_slope_ci_low_mps=float(np.min(lower_bounds)),
            range_min_m=float(np.min(ranges)),
            range_max_m=float(np.max(ranges)),
        )


def _insufficient_trend(
    robot_id: RobotId,
    latest: RobotObservation,
    history: Sequence[RobotObservation],
) -> RobotTrend:
    """Create an uncertain trend before enough temporal evidence exists."""
    start = float(history[0].timestamp) if history else float(latest.timestamp)
    return RobotTrend(
        robot_id=robot_id,
        state="uncertain",
        observation_count=len(history),
        window_start_s=start,
        window_end_s=float(latest.timestamp),
        radial_slope_mps=float("nan"),
        slope_standard_error_mps=float("nan"),
        slope_test_statistic=float("nan"),
        slope_ci_low_mps=float("nan"),
        slope_ci_high_mps=float("nan"),
        residual_rmse_m=float("nan"),
        latest_range_m=float(latest.anchor_range_m),
        latest_bearing_deg=float(latest.anchor_bearing_deg),
    )


def _sorted_robot_ids(values: Iterable[RobotId]) -> tuple[RobotId, ...]:
    """Sort heterogeneous hashable IDs deterministically by type and repr."""
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _build_progressing_graph(
    frame_by_id: dict[RobotId, RobotObservation],
    progressing_ids: set[RobotId],
) -> tuple[dict[RobotId, set[RobotId]], tuple[tuple[RobotId, RobotId], ...]]:
    """Build the undirected induced graph of currently progressing robots."""
    graph = {robot_id: set() for robot_id in progressing_ids}
    unavailable: set[tuple[RobotId, RobotId]] = set()
    for robot_id, observation in frame_by_id.items():
        for neighbor_id in observation.neighbor_ids:
            if neighbor_id not in frame_by_id:
                unavailable.add((robot_id, neighbor_id))
                continue
            if robot_id in progressing_ids and neighbor_id in progressing_ids:
                # Symmetrize a reported local link; either endpoint may be the
                # one that successfully communicated the neighbor observation.
                graph[robot_id].add(neighbor_id)
                graph[neighbor_id].add(robot_id)
    return graph, tuple(
        sorted(unavailable, key=lambda pair: (repr(pair[0]), repr(pair[1])))
    )


def _connected_components(
    graph: dict[RobotId, set[RobotId]],
) -> tuple[tuple[RobotId, ...], ...]:
    """Compute connected components without a graph-library dependency."""
    unseen = set(graph)
    components: list[tuple[RobotId, ...]] = []
    while unseen:
        start = _sorted_robot_ids(unseen)[0]
        queue = deque([start])
        unseen.remove(start)
        component: set[RobotId] = set()
        while queue:
            robot_id = queue.popleft()
            component.add(robot_id)
            for neighbor_id in _sorted_robot_ids(graph[robot_id]):
                if neighbor_id in unseen:
                    unseen.remove(neighbor_id)
                    queue.append(neighbor_id)
        components.append(_sorted_robot_ids(component))
    components.sort(key=lambda component: (repr(component[0]), len(component)))
    return tuple(components)
