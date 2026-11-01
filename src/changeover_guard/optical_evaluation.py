"""Vial-disjoint evaluation, locked references, and retrospective hypotheses.

Recipe labels describe requested recipes. They do not independently establish
carryover, chemical composition, actual delivered volume, or production safety.
"""

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000

from changeover_guard.optical_knn import FEATURES, FRACTIONS, ML, VectorizedOpticalKNN
from changeover_guard.optical_sequence import SequenceConfig


def calibration_registry(frames):
    """Only settled training frames contribute; preserve unavailable recipes."""
    registry = {}
    if frames.empty:
        return registry
    for recipe, group in frames.groupby("recipe_id", sort=True):
        valid = group[group.usable]
        # Equal vial weighting prevents unequal valid-frame counts shifting centroids.
        centroids = valid.groupby("vial_key")[FEATURES].mean()
        center = centroids.mean().to_numpy() if len(centroids) else None
        clipped = group.quality_flags.str.contains("CLIPPED").any()
        registry[int(recipe)] = {
            "mu_Lab": center[:3].tolist() if center is not None else None,
            "mu_OD": center[3:].tolist() if center is not None else None,
            "target_fractions": group[FRACTIONS].iloc[0].tolist(),
            "target_ml": group[ML].iloc[0].tolist(),
            "quality_status": "CLIPPED_WARNING"
            if clipped
            else "CALIBRATED"
            if center is not None
            else "UNAVAILABLE",
            "reference_available": center is not None,
            "training_vial_keys": sorted(valid.vial_key.unique().tolist()),
            "training_frames": len(valid),
        }
    return registry


def fit_model(frames):
    valid = frames[frames.usable] if not frames.empty else frames
    if valid.empty:
        return None
    return VectorizedOpticalKNN().fit(
        valid[FEATURES], valid.recipe_id, valid[FRACTIONS]
    )


def training_ood_threshold(model, frames):
    """95th percentile * 1.5 of nearest OTHER-vial distances, training only."""
    if model is None:
        return np.nan
    valid = frames[frames.usable]
    distances = model.distances(valid[FEATURES])
    keys = valid.vial_key.to_numpy()
    distances[keys[:, None] == keys[None, :]] = np.inf
    nearest = distances.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return max(float(np.quantile(finite, 0.95) * 1.5), 1e-6) if len(finite) else np.nan


def mixing_projection(observed, current, previous):
    vector = np.asarray(previous) - current
    difference = np.asarray(observed) - current
    norm = float(vector @ vector)
    if norm < 1e-8:
        return np.nan, np.nan
    alpha = float(difference @ vector / (norm + 1e-8))
    residual = float(np.linalg.norm(difference - np.clip(alpha, 0, 1) * vector))
    return alpha, residual


def attribute(
    row, registry, *, previous_recipe=None, config=None, ood_threshold=np.nan
):
    config = config or SequenceConfig()
    result = dict(row)
    recipe = row.get("recipe_id")
    reference = registry.get(recipe, {})
    if "target_fractions" in reference:
        result["target_fractions"] = reference["target_fractions"]
    result.update(
        delta_e=np.nan,
        alpha=np.nan,
        mixing_residual=np.nan,
        previous_recipe_id=previous_recipe,
        root_cause_confirmed=False,
        production_status="PENDING",
        Causal_State="EVIDENCE_PENDING",
    )
    if not row["usable"]:
        result.update(
            Causal_State="SENSOR_REJECT",
            operator_action="REJECT [Sensor/Gate]: Optical score unreliable; inspect raw ROI. "
            + row["quality_details"],
        )
    elif not reference.get("reference_available"):
        result["operator_action"] = (
            "PENDING [Reference]: No usable settled calibration for this target recipe."
        )
    else:
        observed = np.array([row[c] for c in FEATURES], float)
        de = float(deltaE_ciede2000(observed[:3], np.array(reference["mu_Lab"])))
        result["delta_e"] = de
        prev = registry.get(previous_recipe, {})
        if previous_recipe != recipe and prev.get("reference_available"):
            result["alpha"], result["mixing_residual"] = mixing_projection(
                observed[3:], np.array(reference["mu_OD"]), np.array(prev["mu_OD"])
            )
        result["target_fractions"] = reference["target_fractions"]
        distance = row.get("nearest_distance", np.nan)
        ood = np.isfinite([distance, ood_threshold]).all() and distance > ood_threshold
        result["out_of_distribution"] = bool(ood)
        # Inferred targets cannot confer an in-spec result when the query is OOD.
        fill_bad = (
            bool(row.get("fill_deviation_verified", False))
            and abs(row.get("fill_deviation_fraction", 0)) > 0.10
        )
        if (ood and row.get("partition") == "unlabeled") or fill_bad:
            result.update(
                Causal_State="SYSTEMIC_FAULT",
                operator_action="FAULT [Suspected]: Outside training support or verified abnormal fill. Inspect valves, tank levels and ROI.",
            )
        elif de <= config.tolerance:
            purpose = str(row.get("color_purpose", ""))
            result.update(
                Causal_State="IN_SPEC_PASS",
                operator_action=f"PASS [Optical In-Spec]: ΔE00={de:.2f} ≤ {config.tolerance}. Matches Recipe {recipe} ({purpose}); production remains pending.",
            )
        elif (
            row.get("vial_in_run", 0) in (1, 2)
            and 0.05 <= result["alpha"] <= 0.85
            and result["mixing_residual"] <= config.tau_mix
        ):
            result.update(
                Causal_State="TRANSIENT_CARRYOVER",
                operator_action=f"CARRYOVER [Suspected]: ΔE00={de:.2f}; optical mixture weight {result['alpha']:.0%} toward Recipe {previous_recipe}. This is not a measured residual-fluid fraction.",
            )
        elif (
            row.get("vial_in_run", 0) > 1 or result["mixing_residual"] > config.tau_mix
        ):
            fractions = [
                row.get(c, np.nan) for c in ("pred_f100", "pred_f200", "pred_f400")
            ]
            result.update(
                Causal_State="SYSTEMIC_FAULT",
                operator_action=f"FAULT [Suspected]: ΔE00={de:.2f}, persistent or off the prior-recipe path. Closest recipe {row.get('predicted_recipe_id')}; estimated fractions {np.round(fractions, 3).tolist()}. Check valves/tank levels.",
            )
        else:
            result["operator_action"] = (
                f"PENDING [Cause]: ΔE00={de:.2f}; insufficient evidence to separate carryover from another fault."
            )
    result["prefill_inspection_suggestion"] = (
        ""
        if result["Causal_State"] == "IN_SPEC_PASS"
        else "Inspect vial position/glass; consider capturing a pre-fill image on a repeat. No subtraction is applied."
    )
    return result


def _add_prediction(row, model, total_ml):
    row = dict(row)
    row.update(
        predicted_recipe_id=np.nan,
        confidence=np.nan,
        nearest_distance=np.nan,
        pred_f100=np.nan,
        pred_f200=np.nan,
        pred_f400=np.nan,
        pred_t100_ml=np.nan,
        pred_t200_ml=np.nan,
        pred_t400_ml=np.nan,
    )
    if model is not None and row["usable"]:
        prediction = model.predict([[row[c] for c in FEATURES]])
        row.update(
            predicted_recipe_id=int(prediction["recipe_id"][0]),
            confidence=float(prediction["confidence"][0]),
            nearest_distance=float(prediction["nearest_distance"][0]),
        )
        for i, column in enumerate(FRACTIONS):
            row[f"pred_{column}"] = float(prediction["fractions"][0, i])
            row[f"pred_{ML[i]}"] = float(prediction["fractions"][0, i] * total_ml)
    return row


def evaluate_labeled(measurements, frames, config=None):
    config = config or SequenceConfig()
    if measurements.empty:
        return {
            "results": measurements.copy(),
            "folds": [],
            "registry": {},
            "model": None,
            "metrics": pd.DataFrame(),
            "confusion": pd.DataFrame(),
            "decay": pd.DataFrame(),
            "ood_threshold": np.nan,
        }
    training = frames[frames.vial_in_run.isin([2, 3])].copy()
    model = fit_model(training)
    registry = calibration_registry(training)
    threshold = training_ood_threshold(model, training)
    rows, folds = [], []
    for video_id, sequence in measurements.groupby("video_id", sort=True):
        previous_recipe, current_recipe = None, None
        for row in sequence.sort_values("sequence_index").to_dict("records"):
            if row["recipe_id"] != current_recipe:
                previous_recipe, current_recipe = current_recipe, row["recipe_id"]
            # Exclude ALL frames of the tested settled vial from scaler, model,
            # references and OOD calibration, including its other sampled frames.
            fold_training = training[training.vial_key.ne(row["vial_key"])]
            fold_model = fit_model(fold_training) if row["vial_in_run"] > 1 else model
            fold_registry = (
                calibration_registry(fold_training)
                if row["vial_in_run"] > 1
                else registry
            )
            fold_ood = (
                training_ood_threshold(fold_model, fold_training)
                if row["vial_in_run"] > 1
                else threshold
            )
            predicted = _add_prediction(row, fold_model, row["total_ml"])
            predicted["fold_id"] = row["vial_key"]
            predicted["volume_basis"] = (
                "requested_total_ml; not measured delivered volume"
            )
            rows.append(
                attribute(
                    predicted,
                    fold_registry,
                    previous_recipe=previous_recipe,
                    config=config,
                    ood_threshold=fold_ood,
                )
            )
            valid = fold_training[fold_training.usable]
            folds.append(
                {
                    "fold_id": row["vial_key"],
                    "partition": row["partition"],
                    "train_vial_keys": sorted(valid.vial_key.unique().tolist()),
                    "train_frame_keys": (
                        valid.vial_key + "/frame" + valid.frame_index.astype(str)
                    ).tolist(),
                    "test_vial_keys": [row["vial_key"]],
                    "ood_threshold": fold_ood,
                    "model": fold_model.to_dict() if fold_model is not None else None,
                    "registry": fold_registry,
                }
            )
    results = pd.DataFrame(rows)
    settled = results[results.vial_in_run.isin([2, 3])]
    scored = settled[settled.usable & settled.predicted_recipe_id.notna()]
    truth = scored.recipe_id.to_numpy(int)
    predicted = scored.predicted_recipe_id.to_numpy(int)
    confusion = pd.DataFrame(0, index=range(5, 15), columns=range(5, 15))
    for actual, guess in zip(truth, predicted):
        confusion.loc[actual, guess] += 1
    first = results[results.vial_in_run.eq(1)]
    transitions = first[first.previous_recipe_id.notna()]
    metrics = [
        {
            "settled_vials": len(settled),
            "scored_settled_vials": len(scored),
            "settled_coverage": len(scored) / len(settled) if len(settled) else 0,
            "recipe_accuracy": float((truth == predicted).mean())
            if len(scored)
            else np.nan,
            "volume_mae_ml": float(
                np.abs(
                    scored[[f"pred_{c}" for c in ML]].to_numpy() - scored[ML].to_numpy()
                ).mean()
            )
            if len(scored)
            else np.nan,
            "volume_metric_scope": "requested-recipe mL equivalents; no actual volume ground truth",
            "first_vial_samples": len(first),
            "known_recipe_transitions": len(transitions),
            "transition_proxy_detection_rate": float(
                transitions.Causal_State.eq("TRANSIENT_CARRYOVER").mean()
            )
            if len(transitions)
            else np.nan,
            "carryover_detection_recall": np.nan,
            "recall_status": "unavailable: recipe-change position is not a positive carryover label",
        }
    ]
    decay = []
    for (video, recipe), group in results.groupby(["video_id", "recipe_id"]):
        de = group.sort_values("vial_in_run").delta_e.to_numpy()
        observable = len(de) == 3 and np.isfinite(de).all()
        decay.append(
            {
                "video_id": video,
                "recipe_id": recipe,
                "observable": observable,
                "monotonic_decay": bool(de[0] > de[1] >= de[2]) if observable else None,
                "scope": "retrospective proxy; vial-disjoint references differ between folds",
            }
        )
    return {
        "results": results,
        "folds": folds,
        "registry": registry,
        "model": model,
        "metrics": pd.DataFrame(metrics),
        "confusion": confusion,
        "decay": pd.DataFrame(decay),
        "ood_threshold": threshold,
    }


def fit_height_calibration(measurements):
    """Monotone height-to-requested-volume proxy, never a verified fill sensor."""
    if measurements.empty:
        return {"status": "unavailable", "verified": False}
    selected = measurements[
        measurements.usable
        & measurements.vial_in_run.isin([2, 3])
        & np.isfinite(measurements.height_px)
    ]
    points = selected.groupby("total_ml").height_px.median().sort_index()
    if len(points) < 3 or not np.all(np.diff(points.to_numpy()) > 0):
        return {
            "status": "unavailable",
            "verified": False,
            "reason": "need at least three monotone observed height/volume levels",
        }
    return {
        "status": "proxy_calibrated",
        "verified": False,
        "height_px": points.tolist(),
        "total_ml": points.index.tolist(),
        "scope": "requested-volume calibration; optical meniscus estimate, not measured delivery",
    }


def smooth_recipes(probabilities, classes, self_transition=0.70):
    """Viterbi dynamic program over time; no assumed fixed run lengths."""
    probabilities = np.asarray(probabilities, float)
    if not len(probabilities):
        return np.array([], int)
    n, c = probabilities.shape
    if c == 1:
        return np.repeat(classes[0], n)
    transition = np.full((c, c), (1 - self_transition) / (c - 1))
    np.fill_diagonal(transition, self_transition)
    log_transition = np.log(transition)
    emission = np.log(np.maximum(probabilities, 1e-12))
    score = emission[0] - np.log(c)
    back = np.zeros((n, c), int)
    for i in range(1, n):
        options = score[:, None] + log_transition
        back[i] = options.argmax(axis=0)
        score = options.max(axis=0) + emission[i]
    path = np.empty(n, int)
    path[-1] = score.argmax()
    for i in range(n - 2, -1, -1):
        path[i] = back[i + 1, path[i + 1]]
    return np.asarray(classes)[path]


def evaluate_unlabeled(events, benchmark, height, config=None):
    if events.empty:
        return events.copy()
    config = config or SequenceConfig()
    model, registry = benchmark["model"], benchmark["registry"]
    records = []
    probabilities = None
    if model is not None and events.usable.any():
        probabilities = np.full(
            (len(events), len(model.classes_)), 1 / len(model.classes_)
        )
        valid = events.usable.to_numpy(bool)
        if valid.any():
            probabilities[valid] = model.predict(events.loc[valid, FEATURES])[
                "probabilities"
            ]
        targets = smooth_recipes(probabilities, model.classes_)
    else:
        targets = [None] * len(events)
    current, previous, position = None, None, 0
    for i, row in enumerate(events.to_dict("records")):
        target = targets[i]
        if target != current:
            previous, current, position = current, target, 0
        position += 1
        total = np.nan
        if height["status"] == "proxy_calibrated" and np.isfinite(row["height_px"]):
            h = row["height_px"]
            if height["height_px"][0] <= h <= height["height_px"][-1]:
                total = float(np.interp(h, height["height_px"], height["total_ml"]))
        predicted = _add_prediction(row, model, total)
        predicted.update(
            recipe_id=target,
            vial_in_run=position,
            estimated_total_ml=total,
            target_source="retrospective Viterbi optical consensus; intended recipe unknown",
            volume_basis="meniscus proxy"
            if np.isfinite(total)
            else "unavailable; fractions only",
            fill_deviation_verified=False,
        )
        records.append(
            attribute(
                predicted,
                registry,
                previous_recipe=previous,
                config=config,
                ood_threshold=benchmark["ood_threshold"],
            )
        )
    return pd.DataFrame(records)
