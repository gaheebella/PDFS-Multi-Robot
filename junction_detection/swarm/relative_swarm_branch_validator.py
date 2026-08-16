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
from math import atan2, degrees, exp, hypot, isfinite, lgamma, log, log1p, sqrt
from typing import Hashable, Iterable, Literal, Optional, Sequence

import numpy as np


RobotId = Hashable
MotionState = Literal[
    "progressing", "non_progressing", "returning", "insufficient"
]
NeighborEdgePolicy = Literal["union", "reciprocal"]


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
    confidence_critical_value: float
    residual_rmse_m: float
    latest_range_m: float
    latest_bearing_deg: float
    representative_bearing_deg: float


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
    returning_robot_ids: tuple[RobotId, ...]
    insufficient_robot_ids: tuple[RobotId, ...]
    progressing_components: tuple[tuple[RobotId, ...], ...]
    rejected_progressing_components: tuple[tuple[RobotId, ...], ...]
    bearing_subcohorts: tuple[tuple[RobotId, ...], ...]
    rejected_bearing_subcohorts: tuple[tuple[RobotId, ...], ...]
    branch_candidates: tuple[BranchCandidate, ...]
    trends: tuple[RobotTrend, ...]
    unavailable_neighbor_references: tuple[tuple[RobotId, RobotId], ...]
    rejected_neighbor_edges: tuple["RejectedNeighborEdge", ...]
    robot_diagnostics: tuple["RobotDiagnostic", ...]


@dataclass(frozen=True)
class RejectedNeighborEdge:
    """A reported local link that was unavailable or rejected by graph policy."""

    first_robot_id: RobotId
    second_robot_id: RobotId
    reason: str


@dataclass(frozen=True)
class RobotDiagnostic:
    """Per-robot explanation of temporal state and cohort assignment."""

    robot_id: RobotId
    observation_count: int
    motion_state: MotionState
    radial_slope_mps: float
    slope_standard_error_mps: float
    slope_ci_low_mps: float
    slope_ci_high_mps: float
    residual_rmse_m: float
    latest_range_m: float
    representative_bearing_deg: float
    neighbor_ids: tuple[RobotId, ...]
    graph_component_id: Optional[int]
    final_cohort_id: Optional[int]
    excluded: bool
    exclusion_reason: Optional[str]


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


def circular_distance_deg(first_deg: float, second_deg: float) -> float:
    """Return the smallest absolute separation between two circular angles."""
    return abs(_normalize_angle_deg(float(first_deg) - float(second_deg)))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction (Numerical Recipes)."""
    maximum_iterations = 200
    tolerance = 3.0e-14
    tiny = np.finfo(float).tiny / tolerance
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= tolerance:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return the regularized incomplete beta for positive a/b and x in [0, 1]."""
    if a <= 0.0 or b <= 0.0 or not 0.0 <= x <= 1.0:
        raise ValueError("regularized incomplete beta received an invalid argument")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = exp(
        lgamma(a + b)
        - lgamma(a)
        - lgamma(b)
        + a * log(x)
        + b * log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """Evaluate the Student-t CDF without requiring SciPy."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail_twice = _regularized_incomplete_beta(
        0.5 * degrees_of_freedom, 0.5, x
    )
    return 1.0 - 0.5 * tail_twice if value > 0.0 else 0.5 * tail_twice


def student_t_critical_value(confidence_level: float, degrees_of_freedom: int) -> float:
    """Return the positive two-sided Student-t critical value.

    SciPy is intentionally not a dependency of this project environment. The
    inverse CDF is therefore found by monotone bisection over a Student-t CDF
    evaluated through the regularized incomplete beta function. This is slower
    than a library quantile but deterministic and accurate for the small OLS
    sample sizes used here. Finite double precision remains the numerical limit
    for confidence levels extremely close to 0 or 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    target = 0.5 * (1.0 + confidence_level)
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < target:
        upper *= 2.0
        if upper > 1.0e8:
            raise ArithmeticError("Student-t quantile bracketing failed")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _student_t_cdf(midpoint, degrees_of_freedom) < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


class RelativeSwarmBranchValidator:
    """Stateful localization-free radial-progress cohort validator.

    Parameters are explicit because temporal inference cannot be defined without
    an observation horizon, sample support, and significance level. Defaults are
    experimental validation settings, not claimed physical constants.

    A robot is ``progressing`` when the lower confidence bound of its OLS radial
    slope is above zero. It is ``non_progressing`` when that interval contains
    zero, which means positive progress was not statistically confirmed—not that
    the robot was proven to be physically stopped at a wall. A significantly
    negative slope is ``returning`` and unavailable regression evidence is
    ``insufficient``. Confidence intervals use a two-sided Student-t critical
    value with ``df=n-2`` for each robot's actual window sample count.

    The progressing graph uses reciprocal links by default so one erroneous
    one-way report cannot merge cohorts. Within each connected component, graph
    edges whose representative-bearing separation exceeds the explicit
    ``maximum_neighbor_bearing_gap_deg`` setting are removed. Connected
    sub-cohorts are then candidates. This graph-constrained circular split lets
    a broad, gradually curving chain remain connected while preventing a single
    angularly inconsistent bridge from merging two direction groups.

    All defaults are experimental synthetic-validation settings, not claimed
    physical constants.
    """

    def __init__(
        self,
        *,
        temporal_window_s: float = 4.0,
        minimum_observations: int = 5,
        confidence_level: float = 0.95,
        minimum_cohort_size: int = 2,
        neighbor_edge_policy: NeighborEdgePolicy = "reciprocal",
        maximum_neighbor_bearing_gap_deg: float = 20.0,
    ) -> None:
        if not isfinite(float(temporal_window_s)) or temporal_window_s <= 0.0:
            raise ValueError("temporal_window_s must be finite and positive")
        if minimum_observations < 3:
            raise ValueError("minimum_observations must be at least 3 for slope uncertainty")
        if not isfinite(float(confidence_level)) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if minimum_cohort_size < 2:
            raise ValueError("minimum_cohort_size must be at least 2")
        if neighbor_edge_policy not in ("union", "reciprocal"):
            raise ValueError("neighbor_edge_policy must be 'union' or 'reciprocal'")
        if (
            not isfinite(float(maximum_neighbor_bearing_gap_deg))
            or not 0.0 < maximum_neighbor_bearing_gap_deg <= 180.0
        ):
            raise ValueError(
                "maximum_neighbor_bearing_gap_deg must lie in (0, 180]"
            )
        self.temporal_window_s = float(temporal_window_s)
        self.minimum_observations = int(minimum_observations)
        self.confidence_level = float(confidence_level)
        self.minimum_cohort_size = int(minimum_cohort_size)
        self.neighbor_edge_policy = neighbor_edge_policy
        self.maximum_neighbor_bearing_gap_deg = float(
            maximum_neighbor_bearing_gap_deg
        )
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
        returning = tuple(
            trend.robot_id for trend in trends if trend.state == "returning"
        )
        insufficient = tuple(
            trend.robot_id for trend in trends if trend.state == "insufficient"
        )

        graph, unavailable, rejected_edges = _build_progressing_graph(
            frame_by_id,
            set(progressing),
            self.neighbor_edge_policy,
        )
        components = _connected_components(graph)
        accepted_components = tuple(
            component
            for component in components
            if len(component) >= self.minimum_cohort_size
        )
        rejected = tuple(
            component
            for component in components
            if len(component) < self.minimum_cohort_size
        )
        representative_bearings = {
            trend.robot_id: trend.representative_bearing_deg for trend in trends
        }
        subcohorts: list[tuple[RobotId, ...]] = []
        rejected_subcohorts: list[tuple[RobotId, ...]] = []
        bearing_rejected_edges: list[RejectedNeighborEdge] = []
        for component in accepted_components:
            component_subcohorts, component_rejected_edges = (
                _bearing_compatible_subcohorts(
                    graph,
                    component,
                    representative_bearings,
                    self.maximum_neighbor_bearing_gap_deg,
                )
            )
            bearing_rejected_edges.extend(component_rejected_edges)
            for subcohort in component_subcohorts:
                if len(subcohort) >= self.minimum_cohort_size:
                    subcohorts.append(subcohort)
                else:
                    rejected_subcohorts.append(subcohort)
        subcohorts.sort(key=lambda cohort: (repr(cohort[0]), len(cohort)))
        rejected_subcohorts.sort(
            key=lambda cohort: (repr(cohort[0]), len(cohort))
        )
        candidates = tuple(
            self._make_candidate(index, subcohort, frame_by_id, trends_by_id)
            for index, subcohort in enumerate(subcohorts)
        )
        all_rejected_edges = tuple(
            sorted(
                (*rejected_edges, *bearing_rejected_edges),
                key=lambda edge: (
                    repr(edge.first_robot_id),
                    repr(edge.second_robot_id),
                    edge.reason,
                ),
            )
        )
        diagnostics = _make_robot_diagnostics(
            frame_by_id=frame_by_id,
            trends=trends,
            components=components,
            rejected_components=rejected,
            rejected_subcohorts=tuple(rejected_subcohorts),
            candidates=candidates,
            rejected_edges=all_rejected_edges,
        )
        return ValidationResult(
            timestamp=timestamp,
            progressing_robot_ids=progressing,
            non_progressing_robot_ids=non_progressing,
            returning_robot_ids=returning,
            insufficient_robot_ids=insufficient,
            progressing_components=components,
            rejected_progressing_components=rejected,
            bearing_subcohorts=tuple(subcohorts),
            rejected_bearing_subcohorts=tuple(rejected_subcohorts),
            branch_candidates=candidates,
            trends=trends,
            unavailable_neighbor_references=unavailable,
            rejected_neighbor_edges=all_rejected_edges,
            robot_diagnostics=diagnostics,
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
        critical_value = student_t_critical_value(self.confidence_level, count - 2)
        representative_bearing, _, _ = circular_mean_and_spread_deg(
            [observation.anchor_bearing_deg for observation in history]
        )

        if slope_standard_error <= slope_floor:
            if slope > slope_floor:
                state: MotionState = "progressing"
                test_statistic = float("inf")
            elif slope < -slope_floor:
                state = "returning"
                test_statistic = float("-inf")
            else:
                state = "non_progressing"
                test_statistic = 0.0
            ci_low = slope
            ci_high = slope
        else:
            test_statistic = slope / slope_standard_error
            ci_low = slope - critical_value * slope_standard_error
            ci_high = slope + critical_value * slope_standard_error
            if ci_low > 0.0:
                state = "progressing"
            elif ci_low <= 0.0 <= ci_high:
                state = "non_progressing"
            else:
                state = "returning"

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
            confidence_critical_value=critical_value,
            residual_rmse_m=residual_rmse,
            latest_range_m=float(latest.anchor_range_m),
            latest_bearing_deg=float(latest.anchor_bearing_deg),
            representative_bearing_deg=representative_bearing,
        )

    @staticmethod
    def _make_candidate(
        cohort_id: int,
        component: tuple[RobotId, ...],
        frame_by_id: dict[RobotId, RobotObservation],
        trends_by_id: dict[RobotId, RobotTrend],
    ) -> BranchCandidate:
        """Summarize one connected cohort without arbitrary confidence weights."""
        bearings = [
            trends_by_id[robot_id].representative_bearing_deg
            for robot_id in component
        ]
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
    """Create an insufficient trend before enough temporal evidence exists."""
    start = float(history[0].timestamp) if history else float(latest.timestamp)
    representative_bearing, _, _ = circular_mean_and_spread_deg(
        [observation.anchor_bearing_deg for observation in history] or [
            latest.anchor_bearing_deg
        ]
    )
    return RobotTrend(
        robot_id=robot_id,
        state="insufficient",
        observation_count=len(history),
        window_start_s=start,
        window_end_s=float(latest.timestamp),
        radial_slope_mps=float("nan"),
        slope_standard_error_mps=float("nan"),
        slope_test_statistic=float("nan"),
        slope_ci_low_mps=float("nan"),
        slope_ci_high_mps=float("nan"),
        confidence_critical_value=float("nan"),
        residual_rmse_m=float("nan"),
        latest_range_m=float(latest.anchor_range_m),
        latest_bearing_deg=float(latest.anchor_bearing_deg),
        representative_bearing_deg=representative_bearing,
    )


def _sorted_robot_ids(values: Iterable[RobotId]) -> tuple[RobotId, ...]:
    """Sort heterogeneous hashable IDs deterministically by type and repr."""
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _build_progressing_graph(
    frame_by_id: dict[RobotId, RobotObservation],
    progressing_ids: set[RobotId],
    edge_policy: NeighborEdgePolicy,
) -> tuple[
    dict[RobotId, set[RobotId]],
    tuple[tuple[RobotId, RobotId], ...],
    tuple[RejectedNeighborEdge, ...],
]:
    """Build the undirected progressing graph under an explicit edge policy."""
    graph = {robot_id: set() for robot_id in progressing_ids}
    unavailable: set[tuple[RobotId, RobotId]] = set()
    reported_links: set[tuple[RobotId, RobotId]] = set()
    rejected: list[RejectedNeighborEdge] = []
    for robot_id, observation in frame_by_id.items():
        for neighbor_id in observation.neighbor_ids:
            if neighbor_id not in frame_by_id:
                unavailable.add((robot_id, neighbor_id))
                rejected.append(
                    RejectedNeighborEdge(
                        robot_id, neighbor_id, "unavailable_neighbor_reference"
                    )
                )
                continue
            reported_links.add((robot_id, neighbor_id))

    progressing_pairs: set[frozenset[RobotId]] = set()
    for first, second in reported_links:
        if first in progressing_ids and second in progressing_ids:
            progressing_pairs.add(frozenset((first, second)))
    for pair in progressing_pairs:
        first, second = _sorted_robot_ids(pair)
        reciprocal = (
            (first, second) in reported_links and (second, first) in reported_links
        )
        if edge_policy == "reciprocal" and not reciprocal:
            rejected.append(
                RejectedNeighborEdge(
                    first, second, "non_reciprocal_neighbor_link"
                )
            )
            continue
        graph[first].add(second)
        graph[second].add(first)
    unavailable_sorted = tuple(
        sorted(unavailable, key=lambda pair: (repr(pair[0]), repr(pair[1])))
    )
    rejected_sorted = tuple(
        sorted(
            rejected,
            key=lambda edge: (
                repr(edge.first_robot_id),
                repr(edge.second_robot_id),
                edge.reason,
            ),
        )
    )
    return graph, unavailable_sorted, rejected_sorted


def _bearing_compatible_subcohorts(
    graph: dict[RobotId, set[RobotId]],
    component: tuple[RobotId, ...],
    representative_bearings: dict[RobotId, float],
    maximum_gap_deg: float,
) -> tuple[
    tuple[tuple[RobotId, ...], ...], tuple[RejectedNeighborEdge, ...]
]:
    """Split a component using both local edges and circular bearing coherence.

    Only existing neighbor edges are considered. An edge is retained when its
    endpoints' temporal representative bearings differ by at most the configured
    circular gap. Thus a wide branch represented by a gradual neighbor chain can
    remain one cohort even when its end-to-end angular span exceeds the limit.
    """
    members = set(component)
    compatible_graph = {robot_id: set() for robot_id in component}
    rejected: list[RejectedNeighborEdge] = []
    visited_edges: set[frozenset[RobotId]] = set()
    for first in component:
        for second in graph[first]:
            if second not in members:
                continue
            pair = frozenset((first, second))
            if pair in visited_edges:
                continue
            visited_edges.add(pair)
            ordered_first, ordered_second = _sorted_robot_ids(pair)
            separation = circular_distance_deg(
                representative_bearings[ordered_first],
                representative_bearings[ordered_second],
            )
            if separation <= maximum_gap_deg:
                compatible_graph[ordered_first].add(ordered_second)
                compatible_graph[ordered_second].add(ordered_first)
            else:
                rejected.append(
                    RejectedNeighborEdge(
                        ordered_first,
                        ordered_second,
                        "bearing_incompatible_neighbor_link",
                    )
                )
    return _connected_components(compatible_graph), tuple(rejected)


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


def _make_robot_diagnostics(
    *,
    frame_by_id: dict[RobotId, RobotObservation],
    trends: tuple[RobotTrend, ...],
    components: tuple[tuple[RobotId, ...], ...],
    rejected_components: tuple[tuple[RobotId, ...], ...],
    rejected_subcohorts: tuple[tuple[RobotId, ...], ...],
    candidates: tuple[BranchCandidate, ...],
    rejected_edges: tuple[RejectedNeighborEdge, ...],
) -> tuple[RobotDiagnostic, ...]:
    """Explain every current robot's state, graph membership, and exclusion."""
    component_by_robot = {
        robot_id: component_id
        for component_id, component in enumerate(components)
        for robot_id in component
    }
    rejected_component_by_robot = {
        robot_id: component
        for component in rejected_components
        for robot_id in component
    }
    rejected_subcohort_ids = {
        robot_id for cohort in rejected_subcohorts for robot_id in cohort
    }
    cohort_by_robot = {
        robot_id: candidate.cohort_id
        for candidate in candidates
        for robot_id in candidate.member_robot_ids
    }
    non_reciprocal_ids = {
        robot_id
        for edge in rejected_edges
        if edge.reason == "non_reciprocal_neighbor_link"
        for robot_id in (edge.first_robot_id, edge.second_robot_id)
    }
    diagnostics: list[RobotDiagnostic] = []
    for trend in trends:
        robot_id = trend.robot_id
        cohort_id = cohort_by_robot.get(robot_id)
        if trend.state == "insufficient":
            reason: Optional[str] = "insufficient_observations"
        elif trend.state == "returning":
            reason = "returning"
        elif trend.state == "non_progressing":
            reason = "not_progressing"
        elif cohort_id is not None:
            reason = None
        elif robot_id in rejected_subcohort_ids:
            reason = "bearing_subcluster_rejected"
        elif robot_id in rejected_component_by_robot:
            component = rejected_component_by_robot[robot_id]
            if robot_id in non_reciprocal_ids:
                reason = "non_reciprocal_neighbor_link"
            elif len(component) == 1:
                reason = "isolated_progressing_robot"
            else:
                reason = "below_minimum_cohort_size"
        else:
            reason = "below_minimum_cohort_size"
        diagnostics.append(
            RobotDiagnostic(
                robot_id=robot_id,
                observation_count=trend.observation_count,
                motion_state=trend.state,
                radial_slope_mps=trend.radial_slope_mps,
                slope_standard_error_mps=trend.slope_standard_error_mps,
                slope_ci_low_mps=trend.slope_ci_low_mps,
                slope_ci_high_mps=trend.slope_ci_high_mps,
                residual_rmse_m=trend.residual_rmse_m,
                latest_range_m=trend.latest_range_m,
                representative_bearing_deg=trend.representative_bearing_deg,
                neighbor_ids=frame_by_id[robot_id].neighbor_ids,
                graph_component_id=component_by_robot.get(robot_id),
                final_cohort_id=cohort_id,
                excluded=cohort_id is None,
                exclusion_reason=reason,
            )
        )
    return tuple(diagnostics)
