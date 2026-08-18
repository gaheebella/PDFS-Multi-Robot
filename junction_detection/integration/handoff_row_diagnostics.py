"""Downstream-only diagnostics for the existing local handoff-row search.

The simulator supplies values already computed by its controller.  This
module stores and plots them; it does not return a decision, threshold, force,
role assignment, or fallback to the simulator.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous dictionaries with stable first-seen columns."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class HandoffRowDiagnostics:
    """Accumulate observational handoff state without producing control data."""

    timeline_rows: list[dict[str, Any]] = field(default_factory=list)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    eligibility_rows: list[dict[str, Any]] = field(default_factory=list)
    occupancy_rows: list[dict[str, Any]] = field(default_factory=list)
    comm_rows: list[dict[str, Any]] = field(default_factory=list)
    _last_state_signature: tuple[Any, ...] | None = None
    _next_event_id: int = 0
    _recorded_outcomes: set[tuple[str, str]] = field(default_factory=set)

    def needs_outcome(self, branch: str, result: str) -> bool:
        """Return whether this Branch/result pair still needs one snapshot."""
        return (branch, result) not in self._recorded_outcomes

    def record_state(self, row: Mapping[str, Any]) -> None:
        """Record only phase/Branch/subphase changes from the existing state."""
        signature = (
            row.get("phase"),
            row.get("current_branch"),
            row.get("target_branch"),
            row.get("subphase"),
            row.get("blocking_reason"),
        )
        if signature == self._last_state_signature:
            return
        self._last_state_signature = signature
        self.timeline_rows.append({"event_type": "STATE_CHANGE", **dict(row)})

    def record_handoff(
        self,
        *,
        event: Mapping[str, Any],
        robots: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]],
        components: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record one unmodified call to the controller's row resolver."""
        event_id = self._next_event_id
        self._next_event_id += 1
        event_row = {"event_id": event_id, "event_type": "HANDOFF_ATTEMPT", **dict(event)}
        self._recorded_outcomes.add(
            (str(event_row["current_branch"]), str(event_row["result"]))
        )
        total = int(event_row["robot_count"])
        frontier_count = int(event_row["frontier_count"])
        if len(robots) != total:
            raise AssertionError("handoff robot snapshot does not match robot_count")
        included = sum(bool(row["candidate_included"]) for row in robots)
        if included != frontier_count:
            raise AssertionError("included candidate count does not match frontier_count")
        for row in rows:
            if int(row["safe_slot_count"]) + int(row["unsafe_slot_count"]) != frontier_count:
                raise AssertionError("row occupancy does not match frontier_count")

        self.timeline_rows.append(event_row)
        self.candidate_rows.append({
            **event_row,
            "included_candidate_count": included,
            "excluded_robot_count": total - included,
            "attempted_row_count": len(rows),
            "candidate_depths": json.dumps([row["candidate_depth"] for row in rows]),
            "common_row_depths": json.dumps([
                row["candidate_depth"] for row in rows if row["all_slots_walkable"]
            ]),
        })
        self.eligibility_rows.extend(
            {"event_id": event_id, **dict(row)} for row in robots
        )
        self.occupancy_rows.extend(
            {"event_id": event_id, **dict(row)} for row in rows
        )
        self.comm_rows.extend(
            {"event_id": event_id, **dict(row)} for row in components
        )

    def _validate(self) -> None:
        """Recheck every stored count before artifacts are written."""
        eligibility_by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.eligibility_rows:
            eligibility_by_event[int(row["event_id"])].append(row)
        occupancy_by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.occupancy_rows:
            occupancy_by_event[int(row["event_id"])].append(row)
        for event in self.candidate_rows:
            event_id = int(event["event_id"])
            robot_rows = eligibility_by_event[event_id]
            if len(robot_rows) != int(event["robot_count"]):
                raise AssertionError("saved robot eligibility total mismatch")
            included = sum(bool(row["candidate_included"]) for row in robot_rows)
            if included != int(event["frontier_count"]):
                raise AssertionError("saved frontier eligibility total mismatch")
            for row in occupancy_by_event[event_id]:
                if int(row["safe_slot_count"]) + int(row["unsafe_slot_count"]) != included:
                    raise AssertionError("saved row slot total mismatch")

    def save(self, output_dir: str | Path) -> None:
        """Validate, then write all requested CSV and PNG diagnostics."""
        self._validate()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(directory / "handoff_event_timeline.csv", self.timeline_rows)
        _write_rows(directory / "handoff_candidate_snapshot.csv", self.candidate_rows)
        _write_rows(directory / "handoff_robot_eligibility.csv", self.eligibility_rows)
        _write_rows(directory / "handoff_row_occupancy.csv", self.occupancy_rows)
        _write_rows(directory / "handoff_comm_diagnostics.csv", self.comm_rows)
        comparison = self._comparison_rows()
        _write_rows(directory / "successful_vs_failed_handoff.csv", comparison)
        _write_rows(directory / "handoff_diagnostic_summary.csv", self._summary_rows(comparison))
        self._save_plots(directory, comparison)

    def _comparison_rows(self) -> list[dict[str, Any]]:
        """Return the first resolver outcome per Branch in execution order."""
        first_by_branch: dict[str, dict[str, Any]] = {}
        for row in self.candidate_rows:
            first_by_branch.setdefault(str(row["current_branch"]), row)
        return list(first_by_branch.values())

    def _summary_rows(
        self, comparison: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create transparent aggregates without classifying root cause."""
        eligibility_by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.eligibility_rows:
            eligibility_by_event[int(row["event_id"])].append(row)
        summary: list[dict[str, Any]] = []
        for row in comparison:
            event_id = int(row["event_id"])
            exclusions = Counter(
                item["eligibility_reason"]
                for item in eligibility_by_event[event_id]
                if not item["candidate_included"]
            )
            summary.append({
                "branch": row["current_branch"],
                "event_id": event_id,
                "frame": row["frame"],
                "timestamp": row["timestamp"],
                "result": row["result"],
                "blocking_reason": row["blocking_reason"],
                "frontier_count": row["frontier_count"],
                "contacted_depth": row["contacted_depth"],
                "resolved_depth": row["resolved_depth"],
                "attempted_row_count": row["attempted_row_count"],
                "walkable_common_row_count": len(json.loads(row["common_row_depths"])),
                "role_counts": row.get("role_counts", "{}"),
                "connected_robot_count": row.get("connected_robot_count", ""),
                "component_count": row.get("component_count", ""),
                "candidate_exclusion_counts": json.dumps(exclusions, sort_keys=True),
            })
        return summary

    def _save_plots(
        self,
        directory: Path,
        comparison: Sequence[Mapping[str, Any]],
    ) -> None:
        """Render compact diagnostic plots from already-recorded values."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        attempts = [row for row in self.timeline_rows if row["event_type"] == "HANDOFF_ATTEMPT"]
        if attempts:
            figure, axis = plt.subplots(figsize=(9, 4))
            y = [1 if row["result"] == "SUCCESS" else 0 for row in attempts]
            axis.scatter([row["timestamp"] for row in attempts], y, c=["tab:green" if value else "tab:red" for value in y])
            for row, value in zip(attempts, y):
                axis.annotate(str(row["current_branch"]), (row["timestamp"], value))
            axis.set(xlabel="simulation time [s]", yticks=[0, 1], yticklabels=["failed", "success"], title="Local handoff resolver outcomes")
            axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(directory / "handoff_timeline.png", dpi=160)
            plt.close(figure)

        if comparison:
            event_ids = {int(row["event_id"]) for row in comparison}
            rows = [row for row in self.occupancy_rows if int(row["event_id"]) in event_ids]
            figure, axis = plt.subplots(figsize=(9, 5))
            branch_by_event = {int(row["event_id"]): str(row["current_branch"]) for row in comparison}
            for event_id in sorted(event_ids):
                selected = [row for row in rows if int(row["event_id"]) == event_id]
                axis.plot([row["candidate_depth"] for row in selected], [row["safe_slot_count"] for row in selected], marker=".", label=branch_by_event[event_id])
            axis.set(xlabel="candidate local axial depth", ylabel="walkable retained slots", title="Successful vs failed row occupancy")
            axis.legend()
            axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(directory / "successful_vs_failed_row_occupancy.png", dpi=160)
            plt.close(figure)

        failed = next((row for row in comparison if row["result"] == "FAILED"), None)
        if failed is not None:
            event_id = int(failed["event_id"])
            robot_rows = [row for row in self.eligibility_rows if int(row["event_id"]) == event_id and row["candidate_included"]]
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter([row["local_axial"] for row in robot_rows], [row["local_lateral"] for row in robot_rows], label="frontier robot")
            axis.scatter([failed["contacted_depth"]] * len(robot_rows), [row["slot_lateral"] for row in robot_rows], marker="x", label="contacted-depth slots")
            axis.set(xlabel="local axial", ylabel="local lateral", title=f"{failed['current_branch']} failed handoff local frame")
            axis.legend()
            axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(directory / "right_handoff_local_frame.png", dpi=160)
            plt.close(figure)

            all_rows = [row for row in self.eligibility_rows if int(row["event_id"]) == event_id]
            counts = Counter(row["eligibility_reason"] for row in all_rows)
            figure, axis = plt.subplots(figsize=(9, 4))
            axis.bar(list(counts), list(counts.values()))
            axis.set(ylabel="robot count", title="Candidate inclusion/exclusion at failed handoff")
            axis.tick_params(axis="x", rotation=25)
            figure.tight_layout()
            figure.savefig(directory / "candidate_exclusion_breakdown.png", dpi=160)
            plt.close(figure)


def run_synthetic_test() -> None:
    """Exercise count invariants without depending on the simulator."""
    diagnostics = HandoffRowDiagnostics()
    diagnostics.record_handoff(
        event={
            "frame": 1, "timestamp": 0.1, "phase": "EXPLORE_BRANCH",
            "subphase": "READY_CONTACT_STALL", "current_branch": "TEST",
            "target_branch": "", "result": "FAILED",
            "blocking_reason": "NO_COMMON_LOCAL_HANDOFF_ROW",
            "robot_count": 2, "frontier_count": 1,
            "contacted_depth": 5.0, "resolved_depth": "",
        },
        robots=(
            {"robot_id": 1, "candidate_included": True, "eligibility_reason": "INCLUDED", "local_axial": 4.9, "local_lateral": 0.0, "slot_lateral": 0.0},
            {"robot_id": 2, "candidate_included": False, "eligibility_reason": "ROLE_NOT_FRONTIER_SHEPHERD", "local_axial": 1.0, "local_lateral": 1.0, "slot_lateral": ""},
        ),
        rows=({"candidate_depth": 5.0, "safe_slot_count": 0, "unsafe_slot_count": 1, "all_slots_walkable": False},),
        components=({"component_id": 0, "robot_count": 2},),
    )
    diagnostics._validate()


if __name__ == "__main__":
    run_synthetic_test()
    print("handoff_row_diagnostics synthetic test: PASS")
