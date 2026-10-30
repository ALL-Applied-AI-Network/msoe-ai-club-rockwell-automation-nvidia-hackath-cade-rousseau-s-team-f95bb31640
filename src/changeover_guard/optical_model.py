"""Small constrained optical-density models with grouped validation.

These models use corrected colors from the vial camera. Top-view tank Lab is
provenance/covariate information, not a substitute for vial-path calibration.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, nnls
from scipy.spatial import Delaunay

from changeover_guard.contracts import Recipe
from changeover_guard.transition_analysis import (
    ML_COLUMNS,
    RGB_COLUMNS,
    validate_clean_controls,
)
from changeover_guard.vial_color import (
    VialCalibration,
    camera_evidence_ready,
    color_errors,
    encoded_rgb,
    lab_values,
    linear_rgb,
    rgb_array,
)


def fractions(quantities) -> np.ndarray:
    values = np.asarray(quantities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("quantities must be finite rows of T100,T200,T400 mL")
    if (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError("quantities must be nonnegative with positive total volume")
    return values / values.sum(axis=1, keepdims=True)


def optical_density(rgb, blank_rgb) -> np.ndarray:
    sample, blank = linear_rgb(rgb), linear_rgb(blank_rgb)
    if np.any(blank <= 0):
        raise ValueError("measured blank must have positive channels")
    # The floor prevents log(0), but clipped observations are rejected by the
    # measurement stage and must not be interpreted as precise absorbance.
    return -np.log(np.maximum(sample, 1e-6) / blank)


@dataclass
class CleanOpticalModel:
    calibration_id: str
    camera_id: str
    blank_rgb: list
    absorbance: list
    training_fractions: list
    validation: dict
    baseline_id: str | None = None
    version: int = 1

    def predict(self, recipe: Recipe, calibration_id: str | None = None) -> dict:
        recipe.validate()
        if calibration_id is not None and calibration_id != self.calibration_id:
            raise ValueError("optical model belongs to a different camera calibration")
        f = fractions([[recipe.ml.get(tank, 0) for tank in ("T100", "T200", "T400")]])[
            0
        ]
        od = f @ np.asarray(self.absorbance)
        rgb = encoded_rgb(linear_rgb(self.blank_rgb) * np.exp(-od))
        # Only interpolation within the measured recipe simplex is supported.
        hull = Delaunay(np.asarray(self.training_fractions)[:, 1:])
        inside = bool(hull.find_simplex(f[1:], tol=1e-8) >= 0)
        return {
            "rgb": rgb.tolist(),
            "lab": lab_values(rgb).tolist(),
            "optical_density": od.tolist(),
            "in_domain": inside,
            "camera_id": self.camera_id,
            "calibration_id": self.calibration_id,
            "baseline_id": self.baseline_id,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        model = cls(**data)
        absorb = np.asarray(model.absorbance, dtype=float)
        fs = np.asarray(model.training_fractions, dtype=float)
        if (
            model.version != 1
            or absorb.shape != (3, 3)
            or not np.isfinite(absorb).all()
            or (absorb < 0).any()
        ):
            raise ValueError("invalid constrained optical-model artifact")
        rgb_array(model.blank_rgb)
        if (
            fs.ndim != 2
            or fs.shape[1] != 3
            or not np.isfinite(fs).all()
            or not np.allclose(fs.sum(axis=1), 1)
        ):
            raise ValueError("invalid optical-model training domain")
        Delaunay(fs[:, 1:])
        return model


def _fit_coefficients(rows: pd.DataFrame, blank_rgb) -> np.ndarray:
    design = fractions(rows[ML_COLUMNS].to_numpy())[:, 1:]
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError(
            "clean controls do not independently vary T200 and T400 fractions"
        )
    observed = optical_density(rows[RGB_COLUMNS].to_numpy(), blank_rgb)
    result = np.zeros((3, 3))  # T100 is the measured clear blank, K100 = 0
    for channel in range(3):
        result[1:, channel] = nnls(design, observed[:, channel])[0]
    return result


def fit_clean_optical(
    controls: pd.DataFrame, calibration: VialCalibration, baseline_id=None
) -> CleanOpticalModel:
    controls = validate_clean_controls(controls, calibration)
    # Equal weight per independently collected clean run, rather than per frame.
    rows = controls.groupby(
        ["source_session_id", "source_run_id", "recipe_id"], as_index=False
    )[ML_COLUMNS + RGB_COLUMNS].median()
    blank_rows = rows[rows.t200_ml.eq(0) & rows.t400_ml.eq(0)]
    if len(blank_rows) < 2:
        raise ValueError(
            "need measured T100 blank controls from at least two independent runs"
        )
    blank = blank_rows[RGB_COLUMNS].median().to_numpy()
    coef = _fit_coefficients(rows, blank)
    domain = fractions(rows[ML_COLUMNS].to_numpy())
    if (
        np.linalg.matrix_rank(np.column_stack([np.ones(len(domain)), domain[:, 1:]]))
        < 3
    ):
        raise ValueError("clean controls do not cover a two-dimensional recipe domain")
    for recipe_id, group in rows.groupby("recipe_id"):
        if len(group[ML_COLUMNS].drop_duplicates()) != 1:
            raise ValueError(f"clean recipe {recipe_id} has inconsistent compositions")
    folds = []
    blank_ids = set(blank_rows.recipe_id)
    for recipe_id, held in rows.groupby("recipe_id"):
        if recipe_id in blank_ids:
            continue  # the independently measured blank is a fixed reference
        train = rows[rows.recipe_id != recipe_id]
        try:
            fitted = _fit_coefficients(train, blank)
        except ValueError:
            continue
        predicted = encoded_rgb(
            linear_rgb(blank) * np.exp(-fractions(held[ML_COLUMNS].to_numpy()) @ fitted)
        )
        errors = color_errors(predicted, held[RGB_COLUMNS].to_numpy())
        # A recipe, rather than each correlated capture, is one validation unit.
        folds.append(
            {
                "recipe_id": int(recipe_id),
                "delta_e": float(np.median(errors)),
                "worst_run_delta_e": float(np.max(errors)),
                "held_out_runs": len(held),
            }
        )
    errors = [fold["worst_run_delta_e"] for fold in folds]
    predicted_train = encoded_rgb(linear_rgb(blank) * np.exp(-domain @ coef))
    validation = {
        "scope": "leave_one_recipe_out_with_independent_fixed_blank",
        "folds": folds,
        "independent_recipe_folds": len(folds),
        "control_runs": len(rows),
        "control_sessions": int(rows.source_session_id.nunique()),
        "median_delta_e": None if not errors else float(np.median(errors)),
        "p95_delta_e": None if not errors else float(np.percentile(errors, 95)),
        "training_median_delta_e": float(
            np.median(color_errors(predicted_train, rows[RGB_COLUMNS].to_numpy()))
        ),
        "uncertainty_delta_e": None,
        "limitation": "RGB absorbance is an effective camera/path proxy. Held-out errors are not chemical-purity measurements.",
    }
    return CleanOpticalModel(
        calibration.calibration_id,
        calibration.camera_id,
        blank.tolist(),
        coef.tolist(),
        np.unique(domain, axis=0).tolist(),
        validation,
        baseline_id,
    )


def average_decay(volume_start, volume_end, tau_ml):
    """Integrate a washout state over each collected vial, rather than at its end."""
    start, end = (
        np.asarray(volume_start, dtype=float),
        np.asarray(volume_end, dtype=float),
    )
    if (
        not np.isfinite(tau_ml)
        or tau_ml <= 0
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
    ):
        raise ValueError("tau and volume intervals must be finite")
    if (start < 0).any() or (end <= start).any():
        raise ValueError("volume intervals require 0 <= start < end")
    return (
        tau_ml
        / (end - start)
        * np.exp(-start / tau_ml)
        * -np.expm1(-(end - start) / tau_ml)
    )


def _decay_fit(groups, bounds) -> tuple[float, dict]:
    def objective(log_tau):
        tau = np.exp(log_tau)
        losses = []
        for rows in groups.values():
            g = average_decay(rows["start"], rows["end"], tau)
            amplitude = g @ rows["residual"] / (g @ g)
            losses.append(np.mean((rows["residual"] - g[:, None] * amplitude) ** 2))
        return float(np.mean(losses))

    fitted = minimize_scalar(objective, bounds=np.log(bounds), method="bounded")
    tau = float(np.exp(fitted.x))
    amplitudes = {}
    for name, rows in groups.items():
        g = average_decay(rows["start"], rows["end"], tau)
        amplitudes[name] = (g @ rows["residual"] / (g @ g)).tolist()
    return tau, amplitudes


def fit_carryover_decay(
    curves: pd.DataFrame,
    support: dict,
    blank_rgb,
    calibration: VialCalibration | None = None,
    repeatability: dict | None = None,
    seed=7,
    threshold=2.2,
) -> dict:
    """Select one shared tau only when grouped validation beats simple baselines.

    Validation leaves out whole collection sessions. A held-out transition's
    first vial initializes its residual vector; only subsequent vials are scored.
    Thus this is a recovery forecast after observing the first vial, not an
    unobserved first-vial prediction for arbitrary recipe pairs.
    """
    result = {
        "kind": "shared_single_exponential_od",
        "status": "insufficient_data",
        "tau_ml": None,
        "tau_interval_ml": None,
        "transition_amplitudes_od": {},
        "validation": [],
        "scope": "conditional_recovery_after_first_observed_vial",
        "reason": support.get("reason"),
        "limitation": "No transition-specific, adsorption or multi-exponential parameters are inferred. Flush/loading effects remain unestimated.",
    }
    if not support.get("shared_decay_supported"):
        return result
    groups = {}
    for name, rows in curves.groupby("transition_id", sort=False):
        if name not in support["qualified_transitions"]:
            continue
        good = rows[rows.usable & rows.delta_e_to_clean.notna()]
        # A failed first sample cannot initialize an unseen transition.
        if good.empty or int(good.vial_in_run.iloc[0]) != 1:
            continue
        target = good[[f"target_{c}" for c in RGB_COLUMNS]].iloc[0].to_numpy(float)
        residual = optical_density(
            good[RGB_COLUMNS].to_numpy(), blank_rgb
        ) - optical_density(target, blank_rgb)
        groups[name] = {
            "start": good.volume_start_ml.to_numpy(),
            "end": good.cumulative_ml.to_numpy(),
            "residual": residual,
            "session": good.session_id.iloc[0],
        }
    sessions = sorted({r["session"] for r in groups.values()})
    if len(groups) < 5 or len(sessions) < 3:
        result["reason"] = (
            "Need five transitions across three sessions with usable first vials."
        )
        return result
    widths = np.concatenate([rows["end"] - rows["start"] for rows in groups.values()])
    # Below 1/4 vial the tau/amplitude product is effectively unresolved; above
    # 20 observed spans the curve is effectively constant. Reject either edge.
    bounds = (
        float(widths.min() / 4),
        float(max(r["end"].max() for r in groups.values()) * 20),
    )
    tau, amplitudes = _decay_fit(groups, bounds)
    folds = []
    for session in sessions:
        train = {k: r for k, r in groups.items() if r["session"] != session}
        held = {k: r for k, r in groups.items() if r["session"] == session}
        fitted_tau, _ = _decay_fit(train, bounds)
        losses, zero, constant = [], [], []
        for rows in held.values():
            g = average_decay(rows["start"], rows["end"], fitted_tau)
            predicted = g[1:, None] * (rows["residual"][0] / g[0])
            losses.append(float(np.mean((predicted - rows["residual"][1:]) ** 2)))
            zero.append(float(np.mean(rows["residual"][1:] ** 2)))
            constant.append(
                float(np.mean((rows["residual"][0] - rows["residual"][1:]) ** 2))
            )
        folds.append(
            {
                "session_id": session,
                "tau_ml": fitted_tau,
                "od_mse": float(np.mean(losses)),
                "no_carryover_od_mse": float(np.mean(zero)),
                "constant_od_mse": float(np.mean(constant)),
            }
        )
    loss = np.mean([f["od_mse"] for f in folds])
    baseline = min(
        np.mean([f["no_carryover_od_mse"] for f in folds]),
        np.mean([f["constant_od_mse"] for f in folds]),
    )
    identified = tau > bounds[0] * 1.05 and tau < bounds[1] / 1.05
    selected = identified and loss < 0.9 * baseline
    result.update(
        {
            "status": "fitted_exploratory" if selected else "unsupported",
            "validation": folds,
            "tau_search_bounds_ml": list(bounds),
            "reason": "Improves both held-out-session baselines by at least 10%."
            if selected
            else "Single exponential is unidentifiable or does not improve both grouped baselines by 10%.",
        }
    )
    if selected:
        result.update({"tau_ml": tau, "transition_amplitudes_od": amplitudes})
        if (
            calibration is not None
            and camera_evidence_ready(calibration, repeatability, threshold)
            and len(sessions) >= 5
        ):
            rng = np.random.default_rng(seed)
            estimates = []
            for _ in range(200):
                selected_sessions = rng.choice(sessions, len(sessions), replace=True)
                sample = {
                    f"{i}/{k}": r
                    for i, s in enumerate(selected_sessions)
                    for k, r in groups.items()
                    if r["session"] == s
                }
                estimate, _ = _decay_fit(sample, bounds)
                estimates.append(estimate)
            result["tau_interval_ml"] = np.percentile(estimates, [2.5, 97.5]).tolist()
            result["interval_method"] = (
                "percentile bootstrap of whole collection sessions, 200 resamples"
            )
    return result


def quality_decision(
    observed_rgb,
    recipe: Recipe,
    model: CleanOpticalModel,
    calibration: VialCalibration,
    repeatability: dict,
    threshold=2.2,
) -> dict:
    """Production decisions always compare to the clean prediction, never residue."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    predicted = model.predict(recipe, calibration.calibration_id)
    error = float(color_errors(observed_rgb, predicted["rgb"]))
    validation = model.validation
    reasons = []
    if not camera_evidence_ready(calibration, repeatability, threshold):
        reasons.append(
            "Camera repeatability/accuracy has not been proven for this calibration and threshold."
        )
    if (
        validation.get("independent_recipe_folds", 0) < 5
        or validation.get("control_sessions", 0) < 3
        or validation.get("p95_delta_e") is None
        or validation["p95_delta_e"] >= threshold / 3
    ):
        reasons.append(
            "Independent clean-model validation has not met the accuracy budget."
        )
    if not predicted["in_domain"]:
        reasons.append("Recipe lies outside the measured clean calibration domain.")
    return {
        "status": "PENDING" if reasons else ("PASS" if error < threshold else "FAIL"),
        "delta_e_to_clean": error,
        "expected_clean_rgb": predicted["rgb"],
        "expected_clean_lab": predicted["lab"],
        "prediction_error_p95_delta_e": validation.get("p95_delta_e")
        if not reasons
        else None,
        "measurement_worst_pair_delta_e": repeatability.get("worst_pair_delta_e")
        if not reasons
        else None,
        "reasons": reasons,
        "calibration_id": calibration.calibration_id,
        "baseline_id": model.baseline_id,
    }
