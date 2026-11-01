"""Controlled exploratory baseline comparison; neither camera is ground truth.

Whole recipe runs are held out from empirical references. Both methods are
scored on the same quality-gated rows, and comparative metrics use their common
coverage. Stable tails and first changeover vials are proxy groups, not labeled
chemical outcomes. No transfer calibration is learned on the evaluation set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from changeover_guard.contracts import TANKS, Recipe
from changeover_guard.optical_model import fractions
from changeover_guard.vial_color import color_errors, encoded_rgb, linear_rgb

RGB = ["r", "g", "b"]
ML = ["t100_ml", "t200_ml", "t400_ml"]
METHODS = ("A_tank_camera", "B_vial_camera")

# Normalized liquid patches for the 4:3 overhead photograph supplied in chat.
# Physical left-to-right identities are fixed, never inferred from liquid hue.
# T300 is the third vessel and is deliberately absent from recipe predictions.
OCT29_TANK_ROIS = {
    "T100": (0.220, 0.488, 0.285, 0.528),
    "T200": (0.415, 0.484, 0.480, 0.524),
    "T400": (0.875, 0.487, 0.930, 0.531),
}


def truth(values: pd.Series) -> pd.Series:
    """CSV booleans must never interpret the string 'False' as true."""
    return values.astype(str).str.lower().isin(["true", "1", "1.0"])


def trusted_rows(measurements: pd.DataFrame) -> pd.Series:
    finite = np.isfinite(measurements[RGB].to_numpy(float)).all(axis=1)
    in_range = measurements[RGB].ge(0).all(axis=1) & measurements[RGB].le(255).all(
        axis=1
    )
    flags = measurements.quality_flags.fillna("").astype(str).eq("")
    return (
        truth(measurements.usable)
        & truth(measurements.source_clock_verified)
        & finite
        & in_range
        & flags
    )


def extract_tank_baselines(image_path: Path, rois=None) -> dict:
    """Read explicit ROIs; a missing chat image remains a missing input.

    Returned RGB is in the tank camera domain. It does not authorize vial
    acceptance, establish tank concentration, or measure a liquid level.
    """
    import cv2

    image_path = Path(image_path)
    if not image_path.is_file():
        return {
            "status": "missing_input",
            "path": str(image_path),
            "tanks": {},
            "reason": "Save the original tank JPG at this path or upload it in the notebook.",
        }
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("tank image could not be decoded")
    height, width = image.shape[:2]
    rois = OCT29_TANK_ROIS if rois is None else rois
    if set(rois) != set(TANKS):
        raise ValueError("provide exactly T100, T200, T400 ROIs; T300 is unavailable")
    tanks = {}
    for tank, box in rois.items():
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError(f"invalid ROI for {tank}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"ROI outside image for {tank}")
        pixels_box = [
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        ]
        left, top, right, bottom = pixels_box
        pixels = image[top:bottom, left:right, ::-1].reshape(-1, 3).astype(float)
        valid = (pixels.max(axis=1) > 8) & (pixels.max(axis=1) < 252)
        if valid.sum() < 20 or valid.mean() < 0.5:
            raise ValueError(
                f"{tank}: too few unobstructed liquid pixels; review the ROI"
            )
        rgb = np.median(pixels[valid], axis=0)
        tanks[tank] = {
            "rgb": rgb.tolist(),
            "roi_pixels": pixels_box,
            "valid_fraction": float(valid.mean()),
            "clipped_channel": bool((rgb <= 1).any()),
        }
    import hashlib

    return {
        "status": "measured",
        "path": str(image_path),
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "image_size": [width, height],
        "tanks": tanks,
        "transfer_calibrated": False,
        "roi_review_required": True,
        "scope": "static tank-camera color; no measured fill or concentration",
    }


def tank_optical_prediction(recipe: Recipe, tank_colors: dict) -> np.ndarray:
    """Geometric linear-RGB mixture, preserving each measured pure endpoint.

    OD is relative to the measured T100 blank. Signed OD is retained for this
    cross-camera candidate: forcing nonnegativity would fail to reproduce a
    colored tank channel brighter than T100. This is an effective color proxy,
    not a vial-path Beer-Lambert calibration. Total path length is unestimated.
    """
    recipe.validate()
    if not recipe.ml:
        raise ValueError("recipe must contain positive volumes")
    if set(tank_colors) != set(TANKS):
        raise ValueError("need all three active tank colors")
    colors = np.asarray([tank_colors[tank] for tank in TANKS], dtype=float)
    linear = linear_rgb(colors)
    if (colors <= 1).any():
        raise ValueError("clipped tank channels cannot identify optical density")
    f = fractions([[recipe.ml.get(tank, 0) for tank in TANKS]])[0]
    od = -np.log(linear / linear[0])
    return encoded_rgb(linear[0] * np.exp(-f @ od))


def stable_tails(measurements: pd.DataFrame, threshold=2.2, tail_count=2) -> list[dict]:
    """Replicate the original stable-tail rule while retaining source identities."""
    tails = []
    for run_id, run in measurements.groupby("run_id", sort=False):
        run = run.sort_values("vial")
        tail = run.tail(tail_count)
        if (
            len(run) <= tail_count
            or len(tail) < tail_count
            or not trusted_rows(tail).all()
        ):
            continue
        rgb = tail[RGB].median().to_numpy(float)
        if color_errors(tail[RGB].to_numpy(float), rgb).max() >= threshold / 3:
            continue
        tails.append(
            {
                "run_id": run_id,
                "recipe_id": int(run.recipe_id.iloc[0]),
                "rgb": rgb.tolist(),
                "composition": run[ML].iloc[0].tolist(),
                "vial_keys": tail.vial_key.tolist(),
            }
        )
    return tails


def empirical_target(tails, recipe_id, composition, excluded_run):
    candidates = [
        t
        for t in tails
        if t["run_id"] != excluded_run
        and t["recipe_id"] == recipe_id
        and np.allclose(t["composition"], composition)
    ]
    if not candidates:
        return None, []
    return np.median([t["rgb"] for t in candidates], axis=0), candidates


def compare_baselines(measurements: pd.DataFrame, tank: dict, threshold=2.2) -> dict:
    """All usable vials retained; absent targets stay explicit, never imputed."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if measurements.vial_key.duplicated().any():
        raise ValueError("measurement identities must be unique")
    data = measurements.sort_values(["session_id", "vial"]).copy()
    data["trusted"] = trusted_rows(data)
    tails = stable_tails(data, threshold)
    # Evaluation tails are chosen by sequence position before inspecting their
    # color agreement. Only training references are filtered for stable color.
    # Otherwise proxy selection would guarantee a low-error 'clean' group.
    tail_keys = {
        key
        for _, run in data.groupby("run_id", sort=False)
        if len(run) > 2
        for key in run.sort_values("vial").tail(2).vial_key
    }
    run_previous = {}
    for _, session in data.groupby("session_id", sort=False):
        previous = None
        for run_id, run in session.groupby("run_id", sort=False):
            run_previous[run_id] = previous
            previous = run.iloc[0]
    rows = []
    for _, vial in data.iterrows():
        previous = run_previous[vial.run_id]
        recipe = Recipe(
            int(vial.recipe_id),
            {
                tank_id: float(vial[col])
                for tank_id, col in zip(TANKS, ML)
                if vial[col] > 0
            },
        )
        b_rgb, candidates = empirical_target(
            tails, int(vial.recipe_id), vial[ML].to_numpy(float), vial.run_id
        )
        a_rgb, a_reason = None, tank.get("reason", "Tank image missing")
        if tank.get("status") == "measured":
            try:
                a_rgb = tank_optical_prediction(
                    recipe, {t: tank["tanks"][t]["rgb"] for t in TANKS}
                )
                a_reason = "Uncalibrated camera transfer; candidate only"
            except ValueError as exc:
                a_reason = str(exc)
        role = (
            "first_changeover_proxy"
            if previous is not None and vial.vial_in_run == 1
            else ("late_run_proxy" if vial.vial_key in tail_keys else "followup")
        )
        for method, target in zip(METHODS, (a_rgb, b_rgb)):
            record = {
                "vial_key": vial.vial_key,
                "vial": int(vial.vial),
                "recipe_id": int(vial.recipe_id),
                "run_id": vial.run_id,
                "vial_in_run": int(vial.vial_in_run),
                "method": method,
                "trusted": bool(vial.trusted),
                "evaluation_role": role,
                "previous_recipe_id": None
                if previous is None
                else int(previous.recipe_id),
                "reference_available": target is not None,
                "delta_e": np.nan,
                "threshold": threshold,
                "below_threshold": None,
                "reference_scope": "tank_camera_unmapped"
                if method == METHODS[0]
                else "other_run_tail_surrogate",
                "reference_reason": a_reason
                if method == METHODS[0]
                else (
                    "Other-run stable tail; exploratory, not certified clean"
                    if target is not None
                    else "No usable stable tail outside the held-out run"
                ),
                "reference_runs": json.dumps([t["run_id"] for t in candidates])
                if method == METHODS[1]
                else "[]",
                "reference_vial_keys": json.dumps(
                    [k for t in candidates for k in t["vial_keys"]]
                )
                if method == METHODS[1]
                else "[]",
            }
            record.update(
                {
                    f"target_{c}": np.nan if target is None else float(value)
                    for c, value in zip(RGB, [np.nan] * 3 if target is None else target)
                }
            )
            if target is not None and vial.trusted:
                record["delta_e"] = float(
                    color_errors(vial[RGB].to_numpy(float), target)
                )
                record["below_threshold"] = record["delta_e"] < threshold
            rows.append(record)
    scores = pd.DataFrame(rows)
    usable = scores[scores.trusted]
    counts = usable.groupby("vial_key").delta_e.count()
    common_keys = counts[counts.eq(2)].index
    scores["paired_comparison"] = scores.vial_key.isin(common_keys)
    metrics = []
    for method in METHODS:
        method_rows = usable[usable.method.eq(method)]
        for scope, subset in [
            ("available_coverage", method_rows),
            ("common_holdout", method_rows[method_rows.vial_key.isin(common_keys)]),
        ]:
            first = subset[
                subset.evaluation_role.eq("first_changeover_proxy")
            ].delta_e.dropna()
            clean = subset[subset.evaluation_role.eq("late_run_proxy")].delta_e.dropna()
            identifiable = len(first) >= 2 and len(clean) >= 2
            metrics.append(
                {
                    "method": method,
                    "scope": scope,
                    "eligible_vials": len(method_rows),
                    "scored_vials": int(subset.delta_e.notna().sum()),
                    "first_proxy_n": len(first),
                    "settled_proxy_n": len(clean),
                    "settled_median_delta_e": float(clean.median())
                    if len(clean)
                    else None,
                    "first_median_delta_e": float(first.median())
                    if len(first)
                    else None,
                    "proxy_separation_delta_e": float(first.median() - clean.median())
                    if identifiable
                    else None,
                    "balanced_proxy_accuracy": float(
                        ((first >= threshold).mean() + (clean < threshold).mean()) / 2
                    )
                    if identifiable
                    else None,
                    "status": "exploratory_proxy_comparison"
                    if identifiable
                    else "insufficient_proxy_groups",
                }
            )
    metrics = pd.DataFrame(metrics)
    shared = metrics[
        metrics.scope.eq("common_holdout")
        & metrics.status.eq("exploratory_proxy_comparison")
    ]
    if len(shared) == 2:
        winner = shared.sort_values(
            ["balanced_proxy_accuracy", "settled_median_delta_e"],
            ascending=[False, True],
        ).iloc[0]
        primary = winner.method
        reason = "Exploratory choice on identical held-out vials and proxy roles; independent validation still required."
        status = "exploratory_comparison"
    elif usable[usable.method.eq(METHODS[1])].delta_e.notna().any():
        primary, status = METHODS[1], "provisional_only"
        reason = "Use available in-camera references. No defensible head-to-head winner: missing A input or insufficient common proxy groups."
    else:
        primary, status = None, "insufficient_data"
        reason = "No supported in-camera target and no identifiable common comparison. Do not promote unmapped tank colors."
    selection = {
        "primary_method": primary,
        "status": status,
        "reason": reason,
        "quality_eligible_vials": int(data.trusted.sum()),
        "common_scored_vials": len(common_keys),
        "threshold": threshold,
        "holdout": "entire evaluated recipe run",
        "scope": "retrospective single-session surrogate benchmark",
        "production_validated": False,
        "limitation": "First-changeover and late-run positions are proxy groups, not clean/contaminated truth. No nested selection test exists.",
    }
    return {
        "scores": scores,
        "metrics": metrics,
        "selection": selection,
        "tails": tails,
    }
