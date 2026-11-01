"""Paired empty-vial correction and exploratory changeover-effect tests.

The correction removes the Lab offset measured from the same vial immediately
before filling.  It is deliberately paired: an empty vial from another cycle
is never substituted.  This addresses vial tint and local illumination changes
without claiming that an additive Lab correction is a calibrated optical model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .color import de00
from .reference import build_references

LAB_COLUMNS = ["L", "a", "b"]
EMPTY_LAB_COLUMNS = ["empty_L", "empty_a", "empty_b"]
CORRECTED_LAB_COLUMNS = ["corrected_L", "corrected_a", "corrected_b"]
RECIPE_ML_COLUMNS = ["t100_ml", "t200_ml", "t400_ml"]


def add_paired_empty_correction(
    vials: pd.DataFrame,
    *,
    max_empty_spread_de: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Return vials with a paired empty-vial Lab offset removed.

    A common median empty-vial color anchors the corrected values so they stay
    in the familiar Lab coordinate system.  Rows without a stable paired empty
    measurement remain NaN instead of borrowing a background from another vial.
    Empty-frame spread is retained as a diagnostic.  Pass an explicit
    ``max_empty_spread_de`` only after establishing a repeatability limit; the
    default does not silently discard every pair using the filled-vial tolerance.
    """
    required = set(LAB_COLUMNS + EMPTY_LAB_COLUMNS)
    missing = required - set(vials)
    if missing:
        raise ValueError(f"paired correction missing columns: {sorted(missing)}")
    if max_empty_spread_de is not None and (
        not np.isfinite(max_empty_spread_de) or max_empty_spread_de <= 0
    ):
        raise ValueError("max_empty_spread_de must be finite and positive")

    result = vials.copy()
    filled = result[LAB_COLUMNS].apply(pd.to_numeric, errors="coerce")
    empty = result[EMPTY_LAB_COLUMNS].apply(pd.to_numeric, errors="coerce")
    paired = filled.notna().all(axis=1) & empty.notna().all(axis=1)
    if "empty_n_frames" in result:
        paired &= pd.to_numeric(result.empty_n_frames, errors="coerce").fillna(0) > 0
    spread = pd.Series(np.nan, index=result.index, dtype=float)
    if "empty_spread_de" in result:
        spread = pd.to_numeric(result.empty_spread_de, errors="coerce")
        if max_empty_spread_de is not None:
            paired &= spread.notna() & spread.le(max_empty_spread_de)

    result["empty_pair_usable"] = paired
    result["empty_pair_reason"] = "usable"
    result.loc[~filled.notna().all(axis=1), "empty_pair_reason"] = "filled_color_missing"
    result.loc[
        filled.notna().all(axis=1) & ~empty.notna().all(axis=1),
        "empty_pair_reason",
    ] = "paired_empty_color_missing"
    if max_empty_spread_de is not None and "empty_spread_de" in result:
        result.loc[
            empty.notna().all(axis=1)
            & spread.gt(max_empty_spread_de),
            "empty_pair_reason",
        ] = "paired_empty_color_unstable"

    result[CORRECTED_LAB_COLUMNS] = np.nan
    result["background_adjustment_delta_e00"] = np.nan
    if not paired.any():
        result["empty_delta_e00_to_canonical"] = np.nan
        return result, None

    canonical = np.median(empty.loc[paired].to_numpy(float), axis=0)
    corrected = (
        filled.loc[paired].to_numpy(float)
        - empty.loc[paired].to_numpy(float)
        + canonical
    )
    result.loc[paired, CORRECTED_LAB_COLUMNS] = corrected
    result.loc[paired, "background_adjustment_delta_e00"] = de00(
        filled.loc[paired].to_numpy(float), corrected
    )
    result["empty_delta_e00_to_canonical"] = np.nan
    result.loc[paired, "empty_delta_e00_to_canonical"] = de00(
        empty.loc[paired].to_numpy(float), canonical.reshape(1, 3)
    )
    return result, canonical


def corrected_reference_input(corrected_vials: pd.DataFrame) -> pd.DataFrame:
    """Present corrected Lab values to the existing reference builder."""
    missing = set(CORRECTED_LAB_COLUMNS) - set(corrected_vials)
    if missing:
        raise ValueError(f"corrected vial table missing columns: {sorted(missing)}")
    result = corrected_vials.copy()
    result[LAB_COLUMNS] = result[CORRECTED_LAB_COLUMNS].to_numpy()
    result.loc[~result.empty_pair_usable, "status"] = "paired empty unavailable"
    return result


def corrected_vial_report(
    corrected_vials: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate leave-one-vial-out corrected Delta E and verdicts."""
    view = corrected_reference_input(corrected_vials)
    full_refs = build_references(view, catalog)
    rows: list[dict] = []
    for vial in view.sort_values("vial_id").itertuples():
        refs = build_references(view, catalog, exclude_vial=vial.vial_id, min_vials=1)
        lab = np.array([vial.L, vial.a, vial.b], dtype=float)
        if vial.status != "ok":
            delta_e, verdict = np.nan, vial.status
        elif vial.recipe_id not in refs.index or refs.loc[vial.recipe_id, "status"] != "ok":
            delta_e, tolerance, verdict = (
                np.nan,
                np.nan,
                "no reference (insufficient data)",
            )
        else:
            ref = refs.loc[vial.recipe_id]
            delta_e = float(de00(lab, ref[LAB_COLUMNS].to_numpy(float))[0])
            tolerance = float(ref.tolerance_de)
            verdict = (
                "consistent"
                if delta_e <= tolerance
                else "OUT OF RANGE"
            )
        if vial.status != "ok":
            tolerance = np.nan
        rows.append(
            {
                "vial_id": vial.vial_id,
                "run_id": getattr(vial, "run_id", np.nan),
                "recipe_id": vial.recipe_id,
                "previous_recipe_id": vial.previous_recipe_id,
                "position_in_run": vial.position_in_run,
                "corrected_L": vial.L,
                "corrected_a": vial.a,
                "corrected_b": vial.b,
                "corrected_delta_e00_to_recipe": delta_e,
                "corrected_tolerance_de": tolerance,
                "corrected_verdict": verdict,
                "empty_pair_usable": bool(vial.empty_pair_usable),
                "empty_pair_reason": vial.empty_pair_reason,
            }
        )
    return pd.DataFrame(rows), full_refs


def transition_effect_table(
    corrected_report: pd.DataFrame,
    corrected_references: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    consecutive: int = 1,
) -> pd.DataFrame:
    """Summarize recovery and target-recipe strength for each recipe change."""
    if consecutive < 1:
        raise ValueError("consecutive must be at least one")
    report = corrected_report.sort_values("vial_id").copy()
    if "run_id" not in report or report.run_id.isna().all():
        changed = report.recipe_id.ne(report.recipe_id.shift()).cumsum()
        report["run_id"] = changed.map(lambda value: f"run{int(value)}")

    rows: list[dict] = []
    for run_id, run in report.groupby("run_id", sort=False):
        run = run.sort_values("position_in_run")
        first = run.iloc[0]
        if pd.isna(first.previous_recipe_id) or first.previous_recipe_id == first.recipe_id:
            continue
        recipe_id = first.recipe_id
        composition = catalog.loc[recipe_id, RECIPE_ML_COLUMNS].astype(float)
        acceptable = run.corrected_verdict.eq("consistent").to_numpy()
        settled_at = next(
            (
                index + 1
                for index in range(len(acceptable) - consecutive + 1)
                if acceptable[index:index + consecutive].all()
            ),
            None,
        )
        ref_l = (
            float(corrected_references.loc[recipe_id, "L"])
            if recipe_id in corrected_references.index
            and corrected_references.loc[recipe_id, "status"] == "ok"
            else np.nan
        )
        total_ml = float(composition.sum())
        dominant_ml = float(composition.max())
        rows.append(
            {
                "run_id": run_id,
                "previous_recipe_id": first.previous_recipe_id,
                "recipe_id": recipe_id,
                "target_darkness": 100.0 - ref_l if np.isfinite(ref_l) else np.nan,
                "dominant_tank": composition.idxmax().replace("_ml", "").upper(),
                "dominant_ml": dominant_ml,
                "dominant_fraction": dominant_ml / total_ml if total_ml > 0 else np.nan,
                "total_ml": total_ml,
                "observed_vials": len(run),
                "vials_to_acceptable": settled_at,
                "pre_acceptable_vials": settled_at - 1 if settled_at is not None else np.nan,
                "recovery_observed": settled_at is not None,
            }
        )
    return pd.DataFrame(rows)


def test_darkness_volume_hypothesis(
    transitions: pd.DataFrame,
    *,
    min_transitions: int = 5,
) -> pd.DataFrame:
    """Test whether darker or more dominant recipes recover in fewer vials.

    Spearman correlation is used because the response is an ordinal vial count.
    A negative rho supports the proposed direction.  The status remains
    exploratory unless there are enough observed transitions; independence
    across sessions must still be assessed by the caller.
    """
    rows = []
    for predictor in ["target_darkness", "dominant_ml", "dominant_fraction"]:
        clean = transitions[[predictor, "pre_acceptable_vials"]].dropna()
        if len(clean) >= 3 and clean[predictor].nunique() >= 2 and clean.pre_acceptable_vials.nunique() >= 2:
            rho, p_value = spearmanr(clean[predictor], clean.pre_acceptable_vials)
        else:
            rho = p_value = np.nan
        if len(clean) < min_transitions:
            status = "insufficient_data"
        elif not np.isfinite(rho):
            status = "not_identifiable"
        elif rho < 0:
            status = "direction_supported_exploratory"
        else:
            status = "direction_not_supported"
        rows.append(
            {
                "predictor": predictor,
                "n_transitions": len(clean),
                "spearman_rho": rho,
                "p_value": p_value,
                "expected_direction": "negative",
                "status": status,
            }
        )
    return pd.DataFrame(rows)
