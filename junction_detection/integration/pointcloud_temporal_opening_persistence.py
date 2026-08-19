"""Anchor-local temporal validation for single-scan opening candidates.

This module deliberately sits after an opening detector.  It receives only
local angular intervals, timestamps, and declared scan metadata; no map,
geometry, pose, branch identity, or ground truth is used at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


def _wrap(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _ccw_width(start: float, end: float) -> float:
    return (float(end) - float(start)) % 360.0


def _interval_parts(start: float, end: float) -> list[tuple[float, float]]:
    s, width = _wrap(start), _ccw_width(start, end)
    e = s + width
    if width >= 360.0 - 1e-9:
        return [(-180.0, 180.0)]
    if e <= 180.0:
        return [(s, e)]
    return [(s, 180.0), (-180.0, e - 360.0)]


def circular_interval_iou(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Return IoU for two local angular intervals, including -180/180 wrap."""
    a_parts, b_parts = _interval_parts(a["start_angle"], a["end_angle"]), _interval_parts(b["start_angle"], b["end_angle"])
    intersection = sum(max(0.0, min(x2, y2) - max(x1, y1)) for x1, x2 in a_parts for y1, y2 in b_parts)
    area_a, area_b = _ccw_width(a["start_angle"], a["end_angle"]), _ccw_width(b["start_angle"], b["end_angle"])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def circular_angle_distance(a: float, b: float) -> float:
    """Return the shortest absolute angular distance in degrees."""
    return abs(_wrap(float(a) - float(b)))


@dataclass(frozen=True)
class TemporalPersistenceConfig:
    """Observable temporal policy; thresholds are configurable, not frame magic."""

    min_persistence_fraction: float = 0.50
    min_persistence_time_sec: float = 0.20
    association_iou: float = 0.20
    association_center_deg: float = 12.0
    max_boundary_std_deg: float = 10.0
    max_track_gap_sec: float = 0.25


@dataclass
class OpeningTrack:
    """Local runtime bookkeeping for one interval hypothesis."""

    track_id: int
    first_seen_time: float
    last_seen_time: float
    observation_count: int = 1
    valid_scan_count: int = 1
    starts: list[float] = field(default_factory=list)
    ends: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    intervals: list[dict[str, float]] = field(default_factory=list)
    accepted: bool = False
    status: str = "PROVISIONAL"
    rejection_reason: str = ""

    @property
    def persistence_duration_sec(self) -> float:
        return max(0.0, self.last_seen_time - self.first_seen_time)

    @property
    def persistence_fraction(self) -> float:
        return self.observation_count / max(self.valid_scan_count, 1)

    @property
    def boundary_std_deg(self) -> float:
        import numpy as np
        # Unwrap each boundary around its first observation so -180/180 does
        # not look like a 360-degree physical jump.
        deviations: list[float] = []
        for series in (self.starts, self.ends):
            if series:
                reference = series[0]
                unwrapped = [reference + _wrap(value - reference) for value in series]
                deviations.append(float(np.std(unwrapped)))
        return max(deviations, default=0.0)

    def latest(self) -> dict[str, float]:
        return dict(self.intervals[-1])

    def summary(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "track_id": self.track_id,
            "first_seen_time": self.first_seen_time,
            "last_seen_time": self.last_seen_time,
            "observation_count": self.observation_count,
            "valid_scan_count": self.valid_scan_count,
            "persistence_duration_sec": self.persistence_duration_sec,
            "persistence_fraction": self.persistence_fraction,
            "latest_start_angle": latest["start_angle"],
            "latest_end_angle": latest["end_angle"],
            "latest_center_angle": latest["center_angle"],
            "latest_width_deg": latest["width_deg"],
            "boundary_std_deg": self.boundary_std_deg,
            "accepted": self.accepted,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
        }


class TemporalOpeningPersistence:
    """Associate circular intervals and accept persistent local openings."""

    def __init__(self, config: TemporalPersistenceConfig = TemporalPersistenceConfig()):
        self.config = config
        self.tracks: list[OpeningTrack] = []
        self._next_id = 0

    def _associate(self, candidate: Mapping[str, float]) -> OpeningTrack | None:
        available = [t for t in self.tracks if t.last_seen_time >= self._time - self.config.max_track_gap_sec]
        scored = [(circular_interval_iou(candidate, t.latest()), t) for t in available]
        scored = [(score, t) for score, t in scored if score >= self.config.association_iou or circular_angle_distance(candidate["center_angle"], t.latest()["center_angle"]) <= self.config.association_center_deg]
        return max(scored, key=lambda item: item[0])[1] if scored else None

    def update(self, timestamp_sec: float, candidates: Sequence[Mapping[str, float]], valid_scan: bool = True) -> list[dict[str, float]]:
        """Add one scan and return currently temporally accepted intervals."""
        if timestamp_sec < 0.0:
            raise ValueError("timestamp_sec must be non-negative")
        self._time = float(timestamp_sec)
        if valid_scan:
            for track in self.tracks:
                track.valid_scan_count += 1
        used: set[int] = set()
        for raw in candidates:
            candidate = {key: float(raw[key]) for key in ("start_angle", "end_angle", "center_angle", "width_deg")}
            track = self._associate(candidate)
            if track is None or track.track_id in used:
                track = OpeningTrack(self._next_id, self._time, self._time)
                self._next_id += 1
                self.tracks.append(track)
            used.add(track.track_id)
            if track.observation_count == 1 and not track.intervals:
                track.starts.append(candidate["start_angle"])
                track.ends.append(candidate["end_angle"])
                track.widths.append(candidate["width_deg"])
            else:
                track.observation_count += 1
                track.last_seen_time = self._time
                track.starts.append(candidate["start_angle"])
                track.ends.append(candidate["end_angle"])
                track.widths.append(candidate["width_deg"])
            track.intervals.append(candidate)
            track.status = "PROVISIONAL"
            track.rejection_reason = ""
        for track in self.tracks:
            if track.track_id not in used and track.last_seen_time < self._time:
                if track.persistence_fraction >= self.config.min_persistence_fraction and track.persistence_duration_sec >= self.config.min_persistence_time_sec and track.boundary_std_deg <= self.config.max_boundary_std_deg:
                    track.accepted, track.status = True, "ACCEPTED"
                else:
                    track.status = "TRANSIENT"
                    track.rejection_reason = "insufficient_temporal_evidence"
        accepted = []
        for track in self.tracks:
            if track.accepted and track.track_id in used:
                accepted.append(track.latest())
        return accepted

    def finalize(self) -> list[dict[str, Any]]:
        """Close tracks and return summaries for audit/output."""
        for track in self.tracks:
            if track.persistence_fraction >= self.config.min_persistence_fraction and track.persistence_duration_sec >= self.config.min_persistence_time_sec and track.boundary_std_deg <= self.config.max_boundary_std_deg:
                track.accepted, track.status = True, "ACCEPTED"
            elif track.status != "ACCEPTED":
                track.status, track.rejection_reason = "REJECTED", "insufficient_temporal_evidence"
        return [track.summary() for track in self.tracks]


def run_synthetic_sanity() -> dict[str, Any]:
    """Check circular association and transient rejection without GT inputs."""
    layer = TemporalOpeningPersistence(TemporalPersistenceConfig(min_persistence_time_sec=0.1))
    wrap = {"start_angle": 170.0, "end_angle": -170.0, "center_angle": 180.0, "width_deg": 20.0}
    layer.update(0.0, [wrap]); layer.update(0.1, [dict(wrap, start_angle=171.0, end_angle=-171.0, center_angle=-179.0)])
    tracks = layer.finalize()
    return {"wrap_track_count": len(tracks), "wrap_accepted": tracks[0]["accepted"], "pass": len(tracks) == 1 and tracks[0]["accepted"]}
