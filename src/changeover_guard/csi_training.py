"""Tank-initialized CSI training with whole-recipe holdout and explicit provenance.

The overhead photo supplies the fixed tank-camera color endpoints. Only CSI
late-vial proxies train the camera transfer, and all rows of a held-out recipe
composition stay outside its fit. No pre-fill measurements are read. This is
exploratory validation against appearance proxies, not chemical ground truth.
"""

import json

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

from changeover_guard.contracts import TANKS, Recipe
from changeover_guard.diagnostic_baselines import (
    ML,
    RGB,
    tank_optical_prediction,
    trusted_rows,
)
from changeover_guard.optical_model import fractions
from changeover_guard.vial_color import color_errors, encoded_rgb, linear_rgb

RAW_METHOD = "A_tank_camera"
TRAINED_METHOD = "A_trained_tank_camera"
EMPIRICAL_METHOD = "B_vial_camera"
METHODS = (RAW_METHOD, TRAINED_METHOD, EMPIRICAL_METHOD)
SCORE_COLUMNS = [
    "vial_key",
    "vial",
    "recipe_id",
    "session_id",
    "run_id",
    "vial_in_run",
    "fold_id",
    "partition",
    "method",
    "trusted",
    "evaluation_role",
    "previous_recipe_id",
    "reference_available",
    "reference_scope",
    "reference_reason",
    "reference_runs",
    "reference_vial_keys",
    "delta_e",
    "threshold",
    "below_threshold",
    "target_r",
    "target_g",
    "target_b",
    "previous_target_r",
    "previous_target_g",
    "previous_target_b",
    "previous_reference_reason",
    "composition_in_training_hull",
    "previous_composition_in_training_hull",
    "paired_comparison",
]


def _recipe(row) -> Recipe:
    recipe = Recipe(
        int(row.recipe_id),
        {
            tank: float(row[column])
            for tank, column in zip(TANKS, ML)
            if row[column] > 0
        },
    )
    recipe.validate()
    if not recipe.ml:
        raise ValueError("CSI recipe must contain positive quantities")
    return recipe


def _validate(data):
    required = {
        "vial_key",
        "vial",
        "recipe_id",
        "session_id",
        "run_id",
        "vial_in_run",
        "usable",
        "quality_flags",
        "source_clock_verified",
        *RGB,
        *ML,
    }
    if missing := required - set(data):
        raise ValueError(
            f"CSI training measurements missing columns: {sorted(missing)}"
        )
    if data.vial_key.duplicated().any():
        raise ValueError("CSI vial identities must be unique")
    if data.session_id.astype(str).str.startswith("10_29_26").any():
        raise ValueError(
            "October 29 vials are not CSI training/test data; only their tank photograph initializes colors"
        )
    if (
        data[["session_id", "run_id", "recipe_id", "vial_in_run", "vial"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "CSI training identities and sequence positions cannot be missing"
        )
    quantities = data[ML].to_numpy(float)
    if not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("CSI recipe quantities must be finite and nonnegative")
    for _, rows in data.groupby("recipe_id"):
        if len(rows[ML].drop_duplicates()) != 1:
            raise ValueError(
                "one CSI recipe ID must have one composition across all sessions"
            )
        _recipe(rows.iloc[0])
    for _, rows in data.groupby("run_id"):
        if rows.recipe_id.nunique() != 1 or rows.session_id.nunique() != 1:
            raise ValueError("a run must belong to one CSI recipe and session")
        positions = rows.sort_values("vial").vial_in_run.to_numpy(float)
        if not np.array_equal(positions, np.arange(1, len(rows) + 1)):
            raise ValueError("CSI vial positions must be contiguous within each run")


def _proxy_context(data):
    previous, late_keys, first_keys = {}, set(), set()
    for _, session in data.groupby("session_id", sort=False):
        prior = None
        for run_id, run in session.groupby("run_id", sort=False):
            previous[run_id] = prior
            run = run.sort_values("vial")
            if len(run) >= 3:
                late_keys.update(run.tail(2).vial_key)
            if prior is not None:
                first_keys.add(run.vial_key.iloc[0])
            prior = run.iloc[0]
    return previous, late_keys, first_keys


def _raw_target(row, tank):
    if tank.get("status") != "measured":
        return None, tank.get("reason", "Tank initialization image is unavailable")
    try:
        colors = {tank_id: tank["tanks"][tank_id]["rgb"] for tank_id in TANKS}
        return tank_optical_prediction(_recipe(row), colors), None
    except (ValueError, KeyError) as exc:
        return None, f"Tank initialization rejected: {exc}"


def _fit_transfer(data, raw_targets, late_keys, ridge_alpha):
    fit = data[data.trusted & data.vial_key.isin(late_keys)].copy()
    fit = fit[fit.vial_key.isin(raw_targets)]
    identity = {
        "fit_vial_keys": fit.vial_key.tolist(),
        "fit_recipe_ids": sorted(int(value) for value in fit.recipe_id.unique()),
        "fit_run_ids": sorted(fit.run_id.unique().tolist()),
        "fit_session_ids": sorted(fit.session_id.unique().tolist()),
    }
    identity["fit_fraction_groups"] = [
        list(key) for key in sorted(fit._fraction_key.unique())
    ]
    if fit._fraction_key.nunique() < 4:
        return {
            "status": "insufficient_data",
            "reason": "Need at least four distinct training fraction groups with usable late-vial proxies",
            **identity,
        }
    # Aggregate vials within runs, then runs within each fractional feature
    # group. Different recipe IDs/total mL with the same fractions are replicate
    # observations of one modeled feature, never extra independent fit rows.
    sample_rows, target_rows = [], []
    for _, group in fit.groupby("_fraction_key", sort=True):
        sample_rows.append(group.iloc[0])
        target_rows.append(
            group.groupby("run_id")[RGB].median().median().to_numpy(float)
        )
    samples = pd.DataFrame(sample_rows)
    target_rgb = np.asarray(target_rows)
    source = np.asarray([raw_targets[row.vial_key] for _, row in samples.iterrows()])
    x = np.column_stack([np.ones(len(source)), linear_rgb(source)])
    f = fractions(samples[ML].to_numpy(float))
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(f)), f[:, 1:]])) < 3:
        return {
            "status": "insufficient_data",
            "reason": "Training recipes do not independently vary both dye fractions",
            **identity,
        }
    if np.linalg.matrix_rank(x) < 4 or np.linalg.cond(x) > 1e8:
        return {
            "status": "insufficient_data",
            "reason": "Tank-derived RGB features cannot identify a full affine camera transfer",
            **identity,
        }
    y = linear_rgb(target_rgb)
    prior = np.vstack([np.zeros(3), np.eye(3)])
    coefficients = np.linalg.solve(
        x.T @ x + ridge_alpha * np.eye(4), x.T @ y + ridge_alpha * prior
    )
    predicted_linear = x @ coefficients
    in_gamut = np.isfinite(predicted_linear).all(axis=1) & (
        (predicted_linear >= 0) & (predicted_linear <= 1)
    ).all(axis=1)
    training_errors = (
        color_errors(
            encoded_rgb(predicted_linear[in_gamut]),
            target_rgb[in_gamut],
        )
        if in_gamut.any()
        else np.array([])
    )
    return {
        "status": "fitted",
        "reason": None,
        **identity,
        "kind": "tank_initialized_affine_linear_rgb",
        "coefficients": coefficients.tolist(),
        "coefficient_rows": ["intercept", "linear_R", "linear_G", "linear_B"],
        "ridge_alpha": float(ridge_alpha),
        "ridge_prior": "identity transfer; fixed before test evaluation",
        "training_fractions": f.tolist(),
        "feature_condition_number": float(np.linalg.cond(x)),
        "training_metrics": {
            "partition": "training_fit",
            "recipe_groups": len(samples),
            "fraction_groups": len(samples),
            "recipe_ids": int(fit.recipe_id.nunique()),
            "out_of_gamut_recipe_groups": int((~in_gamut).sum()),
            "median_delta_e": float(np.median(training_errors))
            if len(training_errors)
            else None,
            "p95_delta_e": float(np.percentile(training_errors, 95))
            if len(training_errors)
            else None,
            "scope": "in-sample training appearance proxies; not validation accuracy",
        },
        "scope": "CSI late-vial appearance proxies only; camera accuracy and chemical cleanliness unproven",
    }


def _predict_transfer(raw, model):
    if raw is None or model["status"] != "fitted":
        return None, "No identifiable train-only camera-transfer fit"
    linear = np.r_[1.0, linear_rgb(raw)] @ np.asarray(model["coefficients"])
    if not np.isfinite(linear).all() or (linear < 0).any() or (linear > 1).any():
        return None, "Train-only camera-transfer prediction is outside the RGB gamut"
    return encoded_rgb(linear), None


def _in_hull(row, model):
    if model["status"] != "fitted":
        return None
    try:
        hull = Delaunay(np.asarray(model["training_fractions"])[:, 1:])
        f = fractions([row[ML].to_numpy(float)])[0]
        return bool(hull.find_simplex(f[1:], tol=1e-8) >= 0)
    except QhullError:
        return False


def _metrics(scores, threshold):
    records = []
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "partition",
                "scope",
                "eligible_vials",
                "scored_vials",
                "status",
            ]
        )
    for method in METHODS:
        eligible = scores[scores.method.eq(method) & scores.trusted]
        for scope, rows in [
            ("available_coverage", eligible),
            ("common_holdout", eligible[eligible.paired_comparison]),
        ]:
            clean = rows[rows.evaluation_role.eq("late_run_proxy")].delta_e.dropna()
            first = rows[
                rows.evaluation_role.eq("first_changeover_proxy")
            ].delta_e.dropna()
            proxy_groups_present = len(clean) >= 2 and len(first) >= 2
            records.append(
                {
                    "method": method,
                    "partition": "test",
                    "scope": scope,
                    "eligible_vials": len(eligible),
                    "scored_vials": int(rows.delta_e.notna().sum()),
                    "late_proxy_n": len(clean),
                    "first_proxy_n": len(first),
                    "test_late_median_delta_e": float(clean.median())
                    if len(clean)
                    else None,
                    "test_late_p95_delta_e": float(np.percentile(clean, 95))
                    if len(clean)
                    else None,
                    "test_first_median_delta_e": float(first.median())
                    if len(first)
                    else None,
                    "proxy_separation_delta_e": float(first.median() - clean.median())
                    if proxy_groups_present
                    else None,
                    "balanced_proxy_accuracy": float(
                        ((first >= threshold).mean() + (clean < threshold).mean()) / 2
                    )
                    if proxy_groups_present
                    else None,
                    "status": "exploratory_proxy_evaluation"
                    if proxy_groups_present
                    else "insufficient_proxy_groups",
                }
            )
    return pd.DataFrame(records)


def train_test_csi(
    measurements: pd.DataFrame, tank: dict, threshold=2.2, *, ridge_alpha=1e-3
) -> dict:
    """Initialize from tank colors, fit on CSI training recipes, score held-out CSI recipes.

    All measurements sharing the same normalized tank fractions, across every
    session and total mL, are held out together. Fraction keys are rounded to 12
    decimal places. Total-volume/path-length effects are not independently
    modeled. Late-vial training proxies are selected by position, not low error.
    The fixed ridge model is declared before evaluating the held-out outcomes.
    A final all-CSI fit is exported separately and never used for test scores.
    """
    if (
        not np.isfinite(threshold)
        or threshold <= 0
        or not np.isfinite(ridge_alpha)
        or ridge_alpha <= 0
    ):
        raise ValueError("threshold and ridge_alpha must be finite and positive")
    if measurements.empty:
        scores = pd.DataFrame(columns=SCORE_COLUMNS)
        return {
            "scores": scores,
            "metrics": _metrics(scores, threshold),
            "folds": [],
            "training_metrics": pd.DataFrame(),
            "tails": [],
            "final_model": {
                "status": "insufficient_data",
                "reason": "CSI measurements unavailable",
            },
            "selection": {
                "primary_method": None,
                "status": "missing_measurements",
                "threshold": threshold,
                "reason": "CSI measurements unavailable; no October 29 vial fallback is permitted",
                "production_validated": False,
            },
        }
    _validate(measurements)
    data = measurements.sort_values(["session_id", "vial"]).copy()
    data["trusted"] = trusted_rows(data)
    data["_fraction_key"] = [
        tuple(np.round(row, 12)) for row in fractions(data[ML].to_numpy(float))
    ]
    previous, late_keys, first_keys = _proxy_context(data)
    raw_targets, raw_reasons = {}, {}
    for _, row in data.iterrows():
        raw, reason = _raw_target(row, tank)
        if raw is not None:
            raw_targets[row.vial_key] = raw
        raw_reasons[row.vial_key] = reason
    folds, models_by_recipe = [], {}
    for composition, held in data.groupby("_fraction_key", sort=True):
        tested_ids = sorted(int(value) for value in held.recipe_id.unique())
        test_mask = data.recipe_id.isin(tested_ids)
        train, test = data[~test_mask], data[test_mask]
        fold_id = "recipes_" + "_".join(str(value) for value in tested_ids)
        model = _fit_transfer(train, raw_targets, late_keys, ridge_alpha)
        fold = {
            "fold_id": fold_id,
            "held_out_recipe_ids": tested_ids,
            "held_out_fractions": list(composition),
            "held_out_compositions_ml": held[ML]
            .drop_duplicates()
            .to_numpy(float)
            .tolist(),
            "train_partition_vial_keys": train.vial_key.tolist(),
            "train_vial_keys": model["fit_vial_keys"],
            "test_vial_keys": test.vial_key.tolist(),
            "train_recipe_ids": sorted(
                int(value) for value in train.recipe_id.unique()
            ),
            "train_session_ids": sorted(train.session_id.unique().tolist()),
            "test_session_ids": sorted(test.session_id.unique().tolist()),
            "model": model,
        }
        assert not set(fold["train_partition_vial_keys"]) & set(fold["test_vial_keys"])
        folds.append(fold)
        for recipe_id in tested_ids:
            models_by_recipe[recipe_id] = fold
    score_rows = []
    for _, row in data.iterrows():
        fold = models_by_recipe[int(row.recipe_id)]
        model = fold["model"]
        raw = raw_targets.get(row.vial_key)
        trained, trained_reason = _predict_transfer(raw, model)
        prior = previous[row.run_id]
        prior_raw, prior_raw_reason = (
            _raw_target(prior, tank)
            if prior is not None
            else (None, "No preceding recipe in this recording")
        )
        prior_trained, prior_trained_reason = _predict_transfer(prior_raw, model)
        role = (
            "first_changeover_proxy"
            if row.vial_key in first_keys
            else ("late_run_proxy" if row.vial_key in late_keys else "followup")
        )
        for method, target, prior_target in [
            (RAW_METHOD, raw, prior_raw),
            (TRAINED_METHOD, trained, prior_trained),
            (EMPIRICAL_METHOD, None, None),
        ]:
            if method == RAW_METHOD:
                scope, reason = (
                    "tank_camera_unmapped",
                    raw_reasons[row.vial_key]
                    or "Uncalibrated tank-camera color candidate",
                )
                prior_reason = (
                    prior_raw_reason
                    or "Previous recipe predicted from the same fixed tank initialization"
                )
            elif method == TRAINED_METHOD:
                scope = "csi_train_only_camera_transfer"
                reason = (
                    trained_reason
                    or "Camera transfer fitted only to other CSI recipe compositions; exploratory"
                )
                prior_reason = (
                    prior_trained_reason
                    or "Previous recipe predicted using this test row's identical fold coefficients"
                )
            else:
                scope, reason = (
                    "held_out_recipe_unavailable",
                    "All vials of the test recipe are held out; no train-only empirical reference for that recipe",
                )
                prior_reason = "No same-fold empirical endpoint"
            available = target is not None
            delta_e = (
                float(color_errors(row[RGB].to_numpy(float), target))
                if available and row.trusted
                else np.nan
            )
            record = {
                "vial_key": row.vial_key,
                "vial": int(row.vial),
                "recipe_id": int(row.recipe_id),
                "session_id": row.session_id,
                "run_id": row.run_id,
                "vial_in_run": int(row.vial_in_run),
                "fold_id": fold["fold_id"],
                "partition": "test",
                "method": method,
                "trusted": bool(row.trusted),
                "evaluation_role": role,
                "previous_recipe_id": None if prior is None else int(prior.recipe_id),
                "reference_available": available,
                "reference_scope": scope,
                "reference_reason": reason,
                "reference_runs": json.dumps(model["fit_run_ids"])
                if method == TRAINED_METHOD
                else "[]",
                "reference_vial_keys": json.dumps(model["fit_vial_keys"])
                if method == TRAINED_METHOD
                else "[]",
                "delta_e": delta_e,
                "threshold": threshold,
                "below_threshold": bool(delta_e < threshold)
                if np.isfinite(delta_e)
                else None,
                "previous_reference_reason": prior_reason,
                "composition_in_training_hull": _in_hull(row, model)
                if method == TRAINED_METHOD
                else None,
                "previous_composition_in_training_hull": _in_hull(prior, model)
                if method == TRAINED_METHOD and prior is not None
                else None,
                **{
                    f"target_{channel}": float(target[index]) if available else np.nan
                    for index, channel in enumerate(RGB)
                },
                **{
                    f"previous_target_{channel}": float(prior_target[index])
                    if prior_target is not None
                    else np.nan
                    for index, channel in enumerate(RGB)
                },
            }
            score_rows.append(record)
    scores = pd.DataFrame(score_rows)
    a_keys = set(scores[scores.method.eq(RAW_METHOD) & scores.delta_e.notna()].vial_key)
    trained_keys = set(
        scores[scores.method.eq(TRAINED_METHOD) & scores.delta_e.notna()].vial_key
    )
    scores["paired_comparison"] = scores.method.isin(
        [RAW_METHOD, TRAINED_METHOD]
    ) & scores.vial_key.isin(a_keys & trained_keys)
    final_model = _fit_transfer(data, raw_targets, late_keys, ridge_alpha)
    final_model["evaluation_usage"] = (
        "All-CSI refit is not used for held-out test predictions"
    )
    training_metrics = pd.DataFrame(
        [
            {"fold_id": fold["fold_id"], **fold["model"]["training_metrics"]}
            for fold in folds
            if fold["model"]["status"] == "fitted"
        ]
    )
    selection = {
        "primary_method": TRAINED_METHOD if trained_keys else None,
        "status": "exploratory_trained" if trained_keys else "insufficient_data",
        "threshold": float(threshold),
        "quality_eligible_vials": int(data.trusted.sum()),
        "common_scored_vials": len(a_keys & trained_keys),
        "trained_scored_vials": len(trained_keys),
        "fitted_folds": sum(fold["model"]["status"] == "fitted" for fold in folds),
        "total_folds": len(folds),
        "holdout": "all vials sharing normalized tank fractions, across recipe IDs, total mL and sessions",
        "initialization": "fixed pure tank colors from the supplied overhead photograph",
        "training_source": "CSI late-vial appearance proxies only; no October 29 vial training/test data",
        "reason": "The tank-initialized ridge model is fixed a priori; held-out outcomes do not tune or select its coefficients"
        if trained_keys
        else (
            "Tank photo missing or training recipe coverage cannot identify the camera-transfer model"
        ),
        "production_validated": False,
        "prefill_subtraction_applied": False,
        "scope": "leave-one-fraction-group-out CSI appearance prediction; no independent total-mL effect",
        "limitations": [
            "Late vials are surrogate clean examples, not certified chemical controls",
            "Single-session recipe holdout does not establish new-session robustness",
            "Total volume/path-length effects are unestimated",
            "Out-of-gamut learned predictions are withheld; extrapolation is flagged",
            "No production confidence intervals or chemical purity claims",
        ],
    }
    return {
        "scores": scores,
        "metrics": _metrics(scores, threshold),
        "folds": folds,
        "training_metrics": training_metrics,
        "selection": selection,
        "final_model": final_model,
        "tails": [],
    }
