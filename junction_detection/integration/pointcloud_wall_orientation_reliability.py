"""Grouped-OOF calibration of wall-tangent angular reliability.

The input is the already-generated sensor-robustness CSV.  GT angular error is
used only as the calibration/evaluation target.  Runtime prediction consumes
only fields observable in one ``WallEstimate`` plus its selected wall span.

The selected model is an interpretable linear 0.90 quantile regression trained
with pinball loss.  It uses estimate mode, fitted point support, observed span,
and left/right disagreement (including structural missingness).  TLS residual
is evaluated as an ablation because the source benchmark showed that a two-
point catastrophic fit can still have zero residual.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from junction_detection.integration.pointcloud_wall_parallel_orientation import (
    WallEstimate,
)


QUANTILE = 0.90
MODEL_VERSION = "wall_linear_quantile_v1"
MODES = (
    "two_wall_parallel",
    "one_wall_dominant_span",
    "one_wall_observed",
)
FORBIDDEN_RUNTIME_TERMS = (
    "gt",
    "actual_error",
    "case_id",
    "topology",
    "anchor",
    "branch_angle",
    "degradation",
    "noise",
    "dropout",
    "visibility",
    "occlusion",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Observable feature subset for an ablation model."""

    name: str
    point_count: bool = False
    wall_span: bool = False
    disagreement: bool = False
    residual: bool = False


@dataclass(frozen=True)
class WallReliability:
    """Runtime wall-orientation availability and predicted 90% error bound."""

    predicted_p90_error_deg: float | None
    available: bool
    model_version: str
    estimate_mode: str
    reason: str


SELECTED_SPEC = FeatureSpec(
    "candidate_mode_support_span_disagreement",
    point_count=True,
    wall_span=True,
    disagreement=True,
    residual=False,
)

ABLATION_SPECS = (
    FeatureSpec("mode_plus_point_count", point_count=True),
    FeatureSpec("mode_plus_span", wall_span=True),
    FeatureSpec("mode_plus_disagreement", disagreement=True),
    FeatureSpec("mode_plus_residual", residual=True),
    SELECTED_SPEC,
    FeatureSpec(
        "candidate_full_including_residual",
        point_count=True,
        wall_span=True,
        disagreement=True,
        residual=True,
    ),
)


def _short_head() -> str:
    """Return current short Git identity without repository mutation."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _load_available_rows(path: Path) -> list[dict[str, str]]:
    """Load only rows for which the existing estimator returned a tangent."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("wall_estimator_available") == "True"
        ]
    required = {
        "case_id",
        "branch_id",
        "estimate_mode",
        "selected_wall_point_count",
        "selected_wall_span_m",
        "left_right_raw_disagreement_deg",
        "line_fit_residual_m",
        "wall_tangent_error_deg",
    }
    missing = required - set(rows[0] if rows else ())
    if not rows or missing:
        raise ValueError(f"robustness CSV has no usable rows or misses {sorted(missing)}")
    unknown_modes = {row["estimate_mode"] for row in rows} - set(MODES)
    if unknown_modes:
        raise ValueError(f"unsupported available estimate modes: {sorted(unknown_modes)}")
    return rows


def _group_folds(
    rows: Sequence[Mapping[str, str]], folds: int, seed: int
) -> np.ndarray:
    """Assign whole geometry cases to balanced deterministic folds."""
    groups: dict[str, int] = {}
    for row in rows:
        case_id = str(row["case_id"])
        groups[case_id] = groups.get(case_id, 0) + 1
    if folds < 2 or len(groups) < folds:
        raise ValueError("grouped CV requires at least two folds and enough cases")
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    # Stable largest-first greedy allocation balances unequal 3/4/5-way cases.
    order_index = {case_id: index for index, case_id in enumerate(shuffled)}
    ordered = sorted(shuffled, key=lambda case_id: (-groups[case_id], order_index[case_id]))
    loads = [0] * folds
    assignment: dict[str, int] = {}
    for case_id in ordered:
        fold = min(range(folds), key=lambda value: (loads[value], value))
        assignment[case_id] = fold
        loads[fold] += groups[case_id]
    result = np.asarray([assignment[str(row["case_id"])] for row in rows], dtype=int)
    # Leakage audit: every case has exactly one test-fold identity.
    for case_id in groups:
        case_folds = set(result[index] for index, row in enumerate(rows) if row["case_id"] == case_id)
        if len(case_folds) != 1:
            raise AssertionError(f"group leakage for {case_id}: {case_folds}")
    return result


def _observable_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    mode = np.asarray([str(row["estimate_mode"]) for row in rows])
    point_count = np.asarray([float(row["selected_wall_point_count"]) for row in rows])
    wall_span = np.log1p([float(row["selected_wall_span_m"]) for row in rows])
    missing = np.asarray(
        [row.get("left_right_raw_disagreement_deg", "") in ("", None) for row in rows],
        dtype=float,
    )
    disagreement = np.asarray([
        np.nan
        if is_missing
        else math.log1p(float(row["left_right_raw_disagreement_deg"]))
        for row, is_missing in zip(rows, missing)
    ])
    residual = np.asarray([float(row["line_fit_residual_m"]) for row in rows])
    return {
        "mode": mode,
        "point_count": point_count,
        "wall_span_log1p": wall_span,
        "disagreement_log1p": disagreement,
        "disagreement_missing": missing,
        "residual": residual,
    }


def _fit_preprocessing(
    rows: Sequence[Mapping[str, Any]], spec: FeatureSpec
) -> dict[str, dict[str, float]]:
    arrays = _observable_arrays(rows)
    selected: list[str] = []
    if spec.point_count:
        selected.append("point_count")
    if spec.wall_span:
        selected.append("wall_span_log1p")
    if spec.disagreement:
        selected.append("disagreement_log1p")
    if spec.residual:
        selected.append("residual")
    preprocessing: dict[str, dict[str, float]] = {}
    for name in selected:
        values = arrays[name]
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(values), values, median)
        mean = float(np.mean(filled))
        scale = float(np.std(filled))
        numerical_floor = np.finfo(float).eps * max(1.0, abs(mean))
        if scale <= numerical_floor:
            scale = 1.0
        preprocessing[name] = {"median": median, "mean": mean, "scale": scale}
    return preprocessing


def _design_matrix(
    rows: Sequence[Mapping[str, Any]],
    spec: FeatureSpec,
    preprocessing: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, list[str]]:
    """Create a fixed, interpretable design matrix from runtime observables."""
    arrays = _observable_arrays(rows)
    columns = [np.ones(len(rows), dtype=float)]
    names = ["intercept"]
    for mode in MODES[1:]:
        columns.append((arrays["mode"] == mode).astype(float))
        names.append(f"mode={mode}")
    selected: list[str] = []
    if spec.point_count:
        selected.append("point_count")
    if spec.wall_span:
        selected.append("wall_span_log1p")
    if spec.disagreement:
        selected.append("disagreement_log1p")
    if spec.residual:
        selected.append("residual")
    for name in selected:
        values = arrays[name]
        parameters = preprocessing[name]
        filled = np.where(np.isfinite(values), values, parameters["median"])
        columns.append((filled - parameters["mean"]) / parameters["scale"])
        names.append(f"z({name})")
    # Missing disagreement is structural: it is exactly the one-wall-observed
    # mode in the source data.  The mode dummy already represents that fact;
    # adding a duplicate indicator would make both coefficients unidentified.
    return np.column_stack(columns), names


def _fit_linear_quantile(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    quantile: float = QUANTILE,
    iterations: int = 15_000,
    learning_rate: float = 0.03,
) -> np.ndarray:
    """Fit linear quantile regression by deterministic Adam pinball descent."""
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    coefficients[0] = float(np.quantile(target, quantile))
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    for iteration in range(1, iterations + 1):
        prediction = matrix @ coefficients
        gradient_factor = (
            -quantile * (target > prediction)
            + (1.0 - quantile) * (target < prediction)
        )
        gradient = gradient_factor @ matrix / len(target)
        # Numerical ridge only; the intercept is never regularized.
        gradient += 1.0e-6 * np.r_[0.0, coefficients[1:]]
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        coefficients -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1.0e-8
        )
    return coefficients


def _predict_linear(
    rows: Sequence[Mapping[str, Any]],
    spec: FeatureSpec,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    matrix, _ = _design_matrix(rows, spec, calibration["preprocessing"])
    coefficients = np.asarray(calibration["coefficients"], dtype=float)
    return np.maximum(0.0, matrix @ coefficients)


def _fit_calibration(
    rows: Sequence[Mapping[str, Any]], spec: FeatureSpec
) -> dict[str, Any]:
    preprocessing = _fit_preprocessing(rows, spec)
    matrix, feature_names = _design_matrix(rows, spec, preprocessing)
    target = np.asarray([float(row["wall_tangent_error_deg"]) for row in rows])
    coefficients = _fit_linear_quantile(matrix, target)
    return {
        "model_version": MODEL_VERSION,
        "model_name": spec.name,
        "quantile": QUANTILE,
        "feature_spec": asdict(spec),
        "feature_names": feature_names,
        "coefficients": coefficients.tolist(),
        "preprocessing": preprocessing,
        "modes": list(MODES),
    }


def _empirical_quantile_predictions(
    training_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    mode_only: bool,
) -> np.ndarray:
    target = np.asarray([float(row["wall_tangent_error_deg"]) for row in training_rows])
    if not mode_only:
        return np.full(len(test_rows), float(np.quantile(target, QUANTILE)))
    bounds = {}
    for mode in MODES:
        errors = [
            float(row["wall_tangent_error_deg"])
            for row in training_rows if row["estimate_mode"] == mode
        ]
        bounds[mode] = float(np.quantile(errors, QUANTILE)) if errors else float(np.quantile(target, QUANTILE))
    return np.asarray([bounds[str(row["estimate_mode"])] for row in test_rows])


def _oof_predictions(
    rows: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    *,
    spec: FeatureSpec | None = None,
    baseline: str | None = None,
) -> np.ndarray:
    prediction = np.zeros(len(rows), dtype=float)
    for fold in sorted(set(fold_ids)):
        train_indices = np.flatnonzero(fold_ids != fold)
        test_indices = np.flatnonzero(fold_ids == fold)
        training = [rows[index] for index in train_indices]
        testing = [rows[index] for index in test_indices]
        train_cases = {row["case_id"] for row in training}
        test_cases = {row["case_id"] for row in testing}
        if train_cases & test_cases:
            raise AssertionError(f"case leakage in fold {fold}")
        if baseline is not None:
            fold_prediction = _empirical_quantile_predictions(
                training, testing, mode_only=baseline == "mode_only"
            )
        elif spec is not None:
            calibration = _fit_calibration(training, spec)
            fold_prediction = _predict_linear(testing, spec, calibration)
        else:
            raise ValueError("spec or baseline is required")
        prediction[test_indices] = fold_prediction
    return prediction


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, implemented without SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank, second_rank = _rankdata(first), _rankdata(second)
    if np.std(first_rank) <= np.finfo(float).eps or np.std(second_rank) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _prediction_metrics(
    name: str, target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    residual = target - prediction
    pinball = np.where(residual >= 0.0, QUANTILE * residual, (QUANTILE - 1.0) * residual)
    return {
        "model": name,
        "nominal_quantile": QUANTILE,
        "empirical_coverage": float(np.mean(target <= prediction)),
        "mean_predicted_bound_deg": float(np.mean(prediction)),
        "median_predicted_bound_deg": float(np.median(prediction)),
        "p90_predicted_bound_deg": float(np.percentile(prediction, 90)),
        "max_predicted_bound_deg": float(np.max(prediction)),
        "spearman_predicted_vs_actual": _spearman(prediction, target),
        "mean_pinball_loss": float(np.mean(pinball)),
    }


def estimate_wall_reliability(
    wall_estimate: WallEstimate,
    observed_wall_span_m: float | None,
    calibration: Mapping[str, Any],
) -> WallReliability:
    """Predict a runtime 90% angular error bound without GT or sensor labels."""
    if wall_estimate.tangent_deg is None:
        return WallReliability(
            None, False, str(calibration.get("model_version", MODEL_VERSION)),
            wall_estimate.estimate_mode, "orientation_unavailable",
        )
    if observed_wall_span_m is None or not math.isfinite(observed_wall_span_m) or observed_wall_span_m < 0.0:
        return WallReliability(
            None, False, str(calibration.get("model_version", MODEL_VERSION)),
            wall_estimate.estimate_mode, "wall_span_unavailable",
        )
    row = {
        "estimate_mode": wall_estimate.estimate_mode,
        "selected_wall_point_count": wall_estimate.fitted_point_count,
        "selected_wall_span_m": observed_wall_span_m,
        "left_right_raw_disagreement_deg": ""
        if wall_estimate.wall_disagreement_deg is None
        else wall_estimate.wall_disagreement_deg,
        "line_fit_residual_m": 0.0
        if wall_estimate.line_fit_residual_m is None
        else wall_estimate.line_fit_residual_m,
    }
    spec = FeatureSpec(**calibration["feature_spec"])
    prediction = float(_predict_linear([row], spec, calibration)[0])
    return WallReliability(
        prediction, True, str(calibration["model_version"]),
        wall_estimate.estimate_mode, "available",
    )


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _by_mode(
    rows: Sequence[Mapping[str, Any]], target: np.ndarray, prediction: np.ndarray
) -> list[dict[str, Any]]:
    result = []
    for mode in MODES:
        indices = np.asarray([row["estimate_mode"] == mode for row in rows])
        actual, predicted = target[indices], prediction[indices]
        result.append({
            "estimate_mode": mode,
            "count": len(actual),
            "actual_mean_error_deg": float(np.mean(actual)),
            "actual_p90_error_deg": float(np.percentile(actual, 90)),
            "actual_max_error_deg": float(np.max(actual)),
            "predicted_mean_bound_deg": float(np.mean(predicted)),
            "predicted_p90_bound_deg": float(np.percentile(predicted, 90)),
            "empirical_coverage": float(np.mean(actual <= predicted)),
        })
    return result


def _save_plots(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = {MODES[0]: "tab:green", MODES[1]: "tab:blue", MODES[2]: "tab:red"}
    for mode in MODES:
        mask = np.asarray([row["estimate_mode"] == mode for row in rows])
        axis.scatter(prediction[mask], target[mask], s=15, alpha=0.5, label=mode, color=colors[mode])
    limit = max(float(np.max(target)), float(np.max(prediction)), 1.0)
    axis.plot([0.0, limit], [0.0, limit], "--", color="0.5", label="actual = bound")
    axis.set(xlabel="predicted P90 error bound [deg]", ylabel="actual error [deg]", title="Grouped-OOF uncertainty vs actual error")
    axis.legend(fontsize=8)
    figure.savefig(directory / "predicted_uncertainty_vs_actual_error.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    predicted_mode, actual_mode = [], []
    for mode in MODES:
        mask = np.asarray([row["estimate_mode"] == mode for row in rows])
        x_value = float(np.mean(prediction[mask]))
        y_value = float(np.percentile(target[mask], 90))
        coverage = float(np.mean(target[mask] <= prediction[mask]))
        predicted_mode.append(x_value)
        actual_mode.append(y_value)
        axis.scatter(x_value, y_value, s=65, color=colors[mode], label=mode)
        axis.annotate(f"coverage={coverage:.2f}", (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    limit = max(predicted_mode + actual_mode + [1.0])
    axis.plot([0.0, limit], [0.0, limit], "--", color="0.5", label="ideal P90")
    axis.set(xlabel="mean predicted P90 bound [deg]", ylabel="empirical P90 actual error [deg]", title="Grouped-OOF calibration by observable estimate mode")
    axis.legend()
    figure.savefig(directory / "reliability_calibration.png", dpi=160)
    plt.close(figure)

    point_count = np.asarray([float(row["selected_wall_point_count"]) for row in rows])
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    scatter = axis.scatter(point_count, prediction, c=target, cmap="plasma", s=18, alpha=0.55)
    axis.set(xlabel="fitted point count", ylabel="predicted P90 error bound [deg]", title="Uncertainty vs point support (color: actual error)")
    figure.colorbar(scatter, ax=axis, label="actual error [deg]")
    figure.savefig(directory / "uncertainty_vs_point_support.png", dpi=160)
    plt.close(figure)


def run_calibration(
    input_csv: Path, output_dir: Path, *, seed: int, folds: int
) -> dict[str, Any]:
    """Calibrate, grouped-CV evaluate, and save a reusable runtime artifact."""
    rows = _load_available_rows(input_csv)
    target = np.asarray([float(row["wall_tangent_error_deg"]) for row in rows])
    fold_ids = _group_folds(rows, folds, seed)

    predictions: dict[str, np.ndarray] = {
        "global_p90": _oof_predictions(rows, fold_ids, baseline="global"),
        "mode_only_p90": _oof_predictions(rows, fold_ids, baseline="mode_only"),
    }
    for spec in ABLATION_SPECS:
        predictions[spec.name] = _oof_predictions(rows, fold_ids, spec=spec)
    selected = predictions[SELECTED_SPEC.name]
    # Deterministic replay covers group allocation, preprocessing, fitting, and prediction.
    replay = _oof_predictions(rows, fold_ids, spec=SELECTED_SPEC)
    if not np.array_equal(selected, replay):
        raise AssertionError("deterministic reliability calibration replay mismatch")

    calibration = _fit_calibration(rows, SELECTED_SPEC)
    replay_calibration = _fit_calibration(rows, SELECTED_SPEC)
    if json.dumps(calibration, sort_keys=True) != json.dumps(replay_calibration, sort_keys=True):
        raise AssertionError("full-data calibration replay mismatch")
    calibration.update({
        "source_csv": str(input_csv),
        "source_csv_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "source_head": _short_head(),
        "training_row_count": len(rows),
        "training_case_count": len({row["case_id"] for row in rows}),
        "grouped_cv_folds": folds,
        "grouped_cv_seed": seed,
        "target_used_only_for_calibration": "wall_tangent_error_deg",
        "runtime_forbidden_features": list(FORBIDDEN_RUNTIME_TERMS),
    })

    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({
            "case_id": row["case_id"],
            "branch_id": row["branch_id"],
            "estimate_mode": row["estimate_mode"],
            "fitted_point_count": row["selected_wall_point_count"],
            "selected_wall_span_m": row["selected_wall_span_m"],
            "left_right_disagreement_deg": row["left_right_raw_disagreement_deg"],
            "disagreement_missing": row["left_right_raw_disagreement_deg"] == "",
            "line_fit_residual_m": row["line_fit_residual_m"],
            "actual_error_deg": target[index],
            "predicted_p90_error_deg": selected[index],
            "covered": target[index] <= selected[index],
            "fold": int(fold_ids[index]),
        })
    ablation = [
        _prediction_metrics(name, target, prediction)
        for name, prediction in predictions.items()
    ]
    selected_metrics = next(row for row in ablation if row["model"] == SELECTED_SPEC.name)
    mode_rows = _by_mode(rows, target, selected)
    worst_index = int(np.argmax(target))
    nominal_standard_error = math.sqrt(QUANTILE * (1.0 - QUANTILE) / len(target))
    mode_spearman = next(row["spearman_predicted_vs_actual"] for row in ablation if row["model"] == "mode_only_p90")
    classification = (
        "A"
        if selected_metrics["empirical_coverage"] >= QUANTILE - 1.96 * nominal_standard_error
        and selected_metrics["spearman_predicted_vs_actual"] > mode_spearman
        and selected[worst_index] > float(np.median(selected))
        else "B"
        if selected_metrics["spearman_predicted_vs_actual"] > 0.0
        else "C"
    )
    summary = [
        {"metric": "classification", "value": classification},
        {"metric": "available_row_count", "value": len(rows)},
        {"metric": "geometry_group_count", "value": len({row["case_id"] for row in rows})},
        {"metric": "grouped_cv_folds", "value": folds},
        {"metric": "group_leakage_case_count", "value": 0},
        {"metric": "nominal_coverage", "value": QUANTILE},
        *({"metric": key, "value": value} for key, value in selected_metrics.items() if key != "model"),
        {"metric": "worst_actual_error_deg", "value": target[worst_index]},
        {"metric": "worst_predicted_p90_error_deg", "value": selected[worst_index]},
        {"metric": "worst_case_id", "value": rows[worst_index]["case_id"]},
        {"metric": "worst_branch_id", "value": rows[worst_index]["branch_id"]},
        {"metric": "worst_fitted_point_count", "value": rows[worst_index]["selected_wall_point_count"]},
        {"metric": "deterministic_replay", "value": True},
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "wall_reliability_oof_predictions.csv", oof_rows)
    _write_rows(output_dir / "wall_reliability_summary.csv", summary)
    _write_rows(output_dir / "wall_reliability_by_mode.csv", mode_rows)
    _write_rows(output_dir / "wall_reliability_ablation.csv", ablation)
    (output_dir / "wall_reliability_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_plots(output_dir, rows, target, selected)
    result = {
        "classification": classification,
        "row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "folds": folds,
        "selected_model": SELECTED_SPEC.name,
        "empirical_coverage": selected_metrics["empirical_coverage"],
        "spearman": selected_metrics["spearman_predicted_vs_actual"],
        "worst_actual_error_deg": float(target[worst_index]),
        "worst_predicted_p90_error_deg": float(selected[worst_index]),
        "output_dir": str(output_dir),
    }
    return result


def _audit_runtime_api() -> None:
    """Assert that runtime API parameter names contain no forbidden GT fields."""
    parameters = set(inspect.signature(estimate_wall_reliability).parameters)
    joined = " ".join(parameters).lower()
    forbidden = [term for term in FORBIDDEN_RUNTIME_TERMS if term in joined]
    if forbidden:
        raise AssertionError(f"runtime API exposes forbidden fields: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "/tmp/pdfs_wall_sensor_robustness_3ff9e0b/"
            "wall_sensor_robustness_results.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"/tmp/pdfs_wall_reliability_{_short_head()}"),
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    _audit_runtime_api()
    result = run_calibration(
        args.input_csv, args.output_dir, seed=args.seed, folds=args.folds
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
