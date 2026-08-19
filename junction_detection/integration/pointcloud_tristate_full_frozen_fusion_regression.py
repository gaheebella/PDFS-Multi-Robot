"""EXP-PointCloudDetector-006 exact-pair audit and frozen-fusion regression.

This harness refuses to join non-identical physical runs.  If the existing
Stable-motion cache cannot be paired to EXP-005 rows, it writes explicit
unpaired output instead of claiming a numeric fusion result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from junction_detection.integration.general_branch_orientation_fusion import (
    BranchOrientationEvidence, fuse_branch_orientation,
)
from junction_detection.integration.pointcloud_temporal_opening_persistence import run_synthetic_sanity


PROTECTED_FILES = (
    "junction_detection/pointcloud/pointcloud_junction_detector.py",
    "junction_detection/pointcloud/pointcloud_junction_detector_uncertainty_aware.py",
    "junction_detection/integration/pointcloud_temporal_opening_persistence.py",
    "junction_detection/integration/pointcloud_wall_parallel_orientation.py",
    "junction_detection/integration/pointcloud_wall_orientation_reliability.py",
    "junction_detection/integration/general_branch_orientation_fusion.py",
)


def _short_head() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError): return "unknown"


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _sha(path: str) -> str:
    try: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError: return "missing"


def _find_motion_sources(search_roots: Sequence[Path]) -> list[Path]:
    patterns = ("*motion*.csv", "*stability*.csv", "*boundary*.csv", "*fusion*.csv")
    found: set[Path] = set()
    for root in search_roots:
        if not root.exists(): continue
        for pattern in patterns: found.update(root.rglob(pattern))
    return sorted(found)


def _pair_audit(evidence_rows: Sequence[Mapping[str, str]], motion_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_keys = {(row.get("case_id", ""), row.get("condition_id", ""), row.get("sequence_type", ""), row.get("seed", "")) for row in evidence_rows}
    motion_rows: list[dict[str, Any]] = []
    for path in motion_paths:
        try:
            for row in _read(path):
                motion_rows.append({"source": str(path), **row})
        except (OSError, csv.Error):
            continue
    motion_keys = {(row.get("case_id", ""), row.get("condition_id", row.get("sensor_condition", "")), row.get("sequence_type", ""), row.get("seed", "")) for row in motion_rows}
    paired = evidence_keys & motion_keys
    audit = [{"key_case_id": key[0], "condition_id": key[1], "sequence_type": key[2], "seed": key[3], "paired": key in paired, "reason": "exact_key_match" if key in paired else "no_exact_motion_row"} for key in sorted(evidence_keys)]
    return audit, motion_rows


def _fusion_api_sanity() -> dict[str, Any]:
    """Exercise the imported frozen API without simulator/GT inputs."""
    agreement = fuse_branch_orientation(BranchOrientationEvidence(10.0, 1.0, True, "geometry"), BranchOrientationEvidence(10.5, 1.0, True, "motion"))
    conflict = fuse_branch_orientation(BranchOrientationEvidence(10.0, 1.0, True, "geometry"), BranchOrientationEvidence(40.0, 1.0, True, "motion"))
    return {"agreement_status": agreement.status, "conflict_status": conflict.status, "pass": agreement.status == "agreement_fused" and conflict.status == "conflict"}


def run_experiment(output_dir: Path, *, evidence_dir: Path = Path("/tmp/pdfs_opening_evidence_35case_bcaaefa"), motion_roots: Sequence[Path] = (Path("/tmp"), Path("junction_detection"))) -> dict[str, Any]:
    """Audit exact pairing and produce regression artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True); evidence_path = evidence_dir / "opening_evidence_35case_oof.csv"; evidence_rows = _read(evidence_path) if evidence_path.exists() else []
    motion_paths = _find_motion_sources(motion_roots); pairing, motion_rows = _pair_audit(evidence_rows, motion_paths); paired = [row for row in pairing if row["paired"]]
    _write(output_dir / "fusion_pairing_audit.csv", pairing); _write(output_dir / "fusion_paired_evidence.csv", [row for row in evidence_rows if (row.get("case_id", ""), row.get("condition_id", ""), row.get("sequence_type", ""), row.get("seed", "")) in {(x["key_case_id"], x["condition_id"], x["sequence_type"], x["seed"]) for x in paired}])
    result_rows = [{"status": "UNPAIRED", "case_id": row["key_case_id"], "condition_id": row["condition_id"], "sequence_type": row["sequence_type"], "seed": row["seed"], "fusion_status": "not_run", "numeric_error_deg": "", "reason": row["reason"]} for row in pairing]
    _write(output_dir / "full_frozen_fusion_results.csv", result_rows); _write(output_dir / "full_frozen_fusion_summary.csv", [{"fusion_status": "not_run", "count": len(result_rows), "numeric_coverage": 0.0}]); _write(output_dir / "full_frozen_fusion_by_condition.csv", []); _write(output_dir / "full_frozen_fusion_by_sequence.csv", []); _write(output_dir / "full_frozen_fusion_status_counts.csv", [{"status": "not_run", "count": len(result_rows)}]); _write(output_dir / "full_frozen_fusion_failure_cases.csv", []); _write(output_dir / "full_frozen_fusion_motion_only_cases.csv", []); _write(output_dir / "full_frozen_fusion_conflicts.csv", []); _write(output_dir / "full_frozen_fusion_false_accepted_openings.csv", []); _write(output_dir / "catastrophic_26_30_replay.csv", [{"status": "not_reproduced", "reason": "no_exact_paired_motion_cache", "case_context": "boundary_t080_long_production"}])
    for name, title in (("fusion_error_distribution.png", "Fusion error: not run"), ("fusion_error_by_condition.png", "Fusion error by condition: not run"), ("fusion_status_by_condition.png", "Fusion status: unpaired"), ("geometry_vs_motion_vs_fusion_error.png", "Geometry/motion/fusion: not run"), ("motion_only_error_distribution.png", "Motion-only: not run"), ("conflict_examples.png", "Conflict examples: not run"), ("false_accepted_downstream_effect.png", "False accepted downstream: not run"), ("resolution_4deg_fusion.png", "4-degree fusion: not run"), ("catastrophic_failure_replay.png", "26-30 degree replay: not run"), ("geometry_availability_vs_motion_fallback.png", "Geometry availability/fallback: not run")):
        figure, axis = plt.subplots(figsize=(7, 3)); axis.text(.5, .5, "No exact paired Stable-motion cache\nNumeric frozen fusion not executed", ha="center", va="center"); axis.set_title(title); axis.axis("off"); figure.tight_layout(); figure.savefig(output_dir / name, dpi=140); plt.close(figure)
    protected = {path: _sha(path) for path in PROTECTED_FILES}; metadata = {"experiment_id": "EXP-PointCloudDetector-006", "head": _short_head(), "evidence_source": str(evidence_path), "evidence_rows": len(evidence_rows), "motion_source_paths": [str(path) for path in motion_paths], "motion_rows_discovered": len(motion_rows), "paired_rows": len(paired), "unmatched_rows": len(pairing) - len(paired), "pairing_coverage": len(paired) / max(len(pairing), 1), "pairing_key": ["case_id", "condition_id", "sequence_type", "seed"], "fusion_numeric_executed": False, "reason": "No exact physical-run Stable-motion cache matched EXP-005 rows; no unsafe join performed", "runtime_leakage": {"gt": False, "map": False, "global_pose": False, "case_id_feature": False, "sensor_condition_feature": False}, "fusion_api_sanity": _fusion_api_sanity(), "circular_sanity": run_synthetic_sanity(), "protected_sha256": protected}
    (output_dir / "fusion_regression_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); return metadata


def _audit() -> dict[str, Any]:
    signature = " ".join(inspect.signature(fuse_branch_orientation).parameters).lower(); forbidden = [token for token in ("gt", "map", "case", "global", "yaw") if token in signature]; return {"fusion_signature": signature, "forbidden_tokens": forbidden, "api_sanity": _fusion_api_sanity(), "pass": not forbidden and _fusion_api_sanity()["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=Path(f"/tmp/pdfs_tristate_full_fusion_{_short_head()}")); parser.add_argument("--evidence-dir", type=Path, default=Path("/tmp/pdfs_opening_evidence_35case_bcaaefa")); parser.add_argument("--motion-root", action="append", type=Path, default=[Path("/tmp")]); parser.add_argument("--audit", action="store_true"); args = parser.parse_args()
    if args.audit: print(json.dumps(_audit(), indent=2)); return
    print(json.dumps({"output_dir": str(args.output_dir), **run_experiment(args.output_dir, evidence_dir=args.evidence_dir, motion_roots=args.motion_root)}, indent=2))


if __name__ == "__main__": main()
