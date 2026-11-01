import json

import numpy as np
import pandas as pd
import pytest

from changeover_guard.contracts import Recipe
from changeover_guard.csi_training import (
    EMPIRICAL_METHOD,
    RAW_METHOD,
    TRAINED_METHOD,
    _predict_transfer,
    train_test_csi,
)
from changeover_guard.diagnostic_baselines import ML, RGB, tank_optical_prediction
from changeover_guard.optical_model import fractions
from changeover_guard.vial_color import encoded_rgb, linear_rgb


@pytest.fixture
def tank():
    return {
        "status": "measured",
        "tanks": {
            "T100": {"rgb": [180, 170, 160]},
            "T200": {"rgb": [170, 45, 30]},
            "T400": {"rgb": [25, 50, 170]},
        },
    }


@pytest.fixture
def csi_rows(tank):
    # Controlled SOFTWARE camera-transfer fixture. These are not camera-control
    # data and never substitute for real CSI evaluation results.
    compositions = {
        5: {"T100": 10, "T200": 10, "T400": 10},
        6: {"T100": 30, "T200": 10},
        7: {"T100": 20, "T400": 50},
        8: {"T100": 20, "T200": 20, "T400": 20},
        9: {"T100": 10, "T200": 35, "T400": 35},
        10: {"T100": 45, "T200": 15, "T400": 15},
        11: {"T100": 10, "T200": 10, "T400": 50},
        12: {"T100": 10, "T200": 50, "T400": 10},
    }
    colors = {t: r["rgb"] for t, r in tank["tanks"].items()}
    rows = []
    prior = None
    for recipe_id, ml in compositions.items():
        base = tank_optical_prediction(Recipe(recipe_id, ml), colors)
        clean = encoded_rgb(0.02 + 0.15 * linear_rgb(base))
        for position in range(1, 5):
            color = (
                clean if position > 1 or prior is None else 0.6 * clean + 0.4 * prior
            )
            vial = len(rows) + 1
            rows.append(
                {
                    "session_id": "csi/video5",
                    "run_id": f"csi/video5/run{recipe_id}",
                    "vial_key": f"csi/video5/vial{vial}",
                    "vial": vial,
                    "recipe_id": recipe_id,
                    "vial_in_run": position,
                    "usable": True,
                    "source_clock_verified": True,
                    "quality_flags": "",
                    **{
                        f"{t.lower()}_ml": ml.get(t, 0)
                        for t in ["T100", "T200", "T400"]
                    },
                    **dict(zip(RGB, color)),
                }
            )
        prior = clean
    return pd.DataFrame(rows)


def test_all_heldout_recipe_vials_are_disjoint_from_fit_and_training_pool(
    csi_rows, tank
):
    result = train_test_csi(csi_rows, tank)
    assert len(result["scores"]) == 3 * len(csi_rows)
    assert result["selection"]["fitted_folds"] == 7
    assert result["selection"]["primary_method"] == TRAINED_METHOD
    for fold in result["folds"]:
        test = set(fold["test_vial_keys"])
        assert not test.intersection(fold["train_vial_keys"])
        assert not test.intersection(fold["train_partition_vial_keys"])
        assert not set(fold["held_out_recipe_ids"]).intersection(
            fold["train_recipe_ids"]
        )
        fitted_rows = csi_rows[csi_rows.vial_key.isin(fold["train_vial_keys"])]
        assert fitted_rows.vial_in_run.ge(3).all()
    assert not result["selection"]["production_validated"]


def test_heldout_color_changes_do_not_change_that_folds_fit(csi_rows, tank):
    before = train_test_csi(csi_rows, tank)
    mutated = csi_rows.copy()
    mutated.loc[mutated.recipe_id.isin([5, 8]), RGB] = [120, 20, 80]
    after = train_test_csi(mutated, tank)
    first = next(f for f in before["folds"] if f["held_out_recipe_ids"] == [5, 8])
    second = next(f for f in after["folds"] if f["held_out_recipe_ids"] == [5, 8])
    np.testing.assert_array_equal(
        first["model"]["coefficients"], second["model"]["coefficients"]
    )
    assert first["model"]["training_metrics"] == second["model"]["training_metrics"]


def test_heldout_recipe_quantities_do_not_enter_its_training_fit(csi_rows, tank):
    before = train_test_csi(csi_rows, tank)
    mutated = csi_rows.copy()
    mutated.loc[mutated.recipe_id.isin([5, 8]), ML] = [20, 10, 10]
    after = train_test_csi(mutated, tank)
    first = next(f for f in before["folds"] if f["held_out_recipe_ids"] == [5, 8])
    second = next(f for f in after["folds"] if f["held_out_recipe_ids"] == [5, 8])
    np.testing.assert_array_equal(
        first["model"]["coefficients"], second["model"]["coefficients"]
    )


def test_recipe_holdout_spans_all_recording_sessions(csi_rows, tank):
    other = csi_rows.copy()
    other["session_id"] = "csi/video6"
    other["run_id"] = other.run_id.str.replace("video5", "video6")
    other["vial_key"] = other.vial_key.str.replace("video5", "video6")
    result = train_test_csi(pd.concat([csi_rows, other], ignore_index=True), tank)
    fold = next(f for f in result["folds"] if f["held_out_recipe_ids"] == [5, 8])
    assert set(fold["test_session_ids"]) == {"csi/video5", "csi/video6"}
    assert len(fold["test_vial_keys"]) == 16
    for key in fold["train_vial_keys"]:
        row = pd.concat([csi_rows, other]).set_index("vial_key").loc[key]
        assert row.recipe_id not in [5, 8]


def test_previous_recipe_endpoint_uses_identical_fold_coefficients(csi_rows, tank):
    result = train_test_csi(csi_rows, tank)
    row = result["scores"].query("method == @TRAINED_METHOD and recipe_id == 6").iloc[0]
    fold = next(f for f in result["folds"] if f["fold_id"] == row.fold_id)
    prior = csi_rows[csi_rows.recipe_id.eq(5)].iloc[0]
    recipe = Recipe(5, {t.upper()[:4]: prior[t] for t in ML if prior[t] > 0})
    raw = tank_optical_prediction(
        recipe, {t: v["rgb"] for t, v in tank["tanks"].items()}
    )
    expected = encoded_rgb(np.r_[1.0, linear_rgb(raw)] @ fold["model"]["coefficients"])
    np.testing.assert_allclose(
        row[[f"previous_target_{c}" for c in RGB]].to_numpy(float), expected
    )
    assert "identical fold" in row.previous_reference_reason


def test_same_recipe_empirical_reference_is_unavailable_without_leaking_test_vials(
    csi_rows, tank
):
    result = train_test_csi(csi_rows, tank)
    empirical = result["scores"][result["scores"].method.eq(EMPIRICAL_METHOD)]
    assert not empirical.reference_available.any()
    assert empirical.delta_e.isna().all()
    assert empirical.reference_vial_keys.eq("[]").all()
    assert result["metrics"].partition.eq("test").all()
    assert result["training_metrics"].partition.eq("training_fit").all()


def test_common_cohort_compares_raw_and_trained_using_identical_test_vials(
    csi_rows, tank
):
    result = train_test_csi(csi_rows, tank)
    scores = result["scores"]
    a = scores[scores.method.eq(RAW_METHOD) & scores.paired_comparison]
    learned = scores[scores.method.eq(TRAINED_METHOD) & scores.paired_comparison]
    assert set(a.vial_key) == set(learned.vial_key)
    assert set(a.fold_id) == set(learned.fold_id)
    for row in learned.itertuples():
        assert row.vial_key not in json.loads(row.reference_vial_keys)


def test_missing_tank_photo_keeps_both_training_and_predictions_pending(csi_rows):
    result = train_test_csi(
        csi_rows, {"status": "missing_input", "reason": "photo missing"}
    )
    assert result["selection"]["status"] == "insufficient_data"
    assert result["selection"]["primary_method"] is None
    assert result["scores"].delta_e.isna().all()
    assert result["final_model"]["status"] == "insufficient_data"
    assert len(result["scores"]) == 3 * len(csi_rows)


def test_no_dataset_never_falls_back_to_october29_vials():
    result = train_test_csi(pd.DataFrame(), {"status": "missing_input"})
    assert result["selection"]["status"] == "missing_measurements"
    assert result["scores"].empty
    assert result["selection"]["primary_method"] is None


def test_october29_vial_training_is_explicitly_rejected(csi_rows, tank):
    csi_rows["session_id"] = "10_29_26"
    with pytest.raises(ValueError, match="not CSI training/test"):
        train_test_csi(csi_rows, tank)


def test_single_recipe_cannot_identify_a_camera_transfer(csi_rows, tank):
    result = train_test_csi(csi_rows[csi_rows.recipe_id.eq(5)], tank)
    assert result["selection"]["status"] == "insufficient_data"
    assert result["folds"][0]["model"]["status"] == "insufficient_data"


def test_final_all_data_refit_is_separate_from_heldout_scores(csi_rows, tank):
    result = train_test_csi(csi_rows, tank)
    assert result["final_model"]["status"] == "fitted"
    assert "not used" in result["final_model"]["evaluation_usage"]
    assert len(result["final_model"]["fit_recipe_ids"]) == 8
    assert len(result["final_model"]["fit_fraction_groups"]) == 7
    for fold in result["folds"]:
        assert len(fold["model"]["fit_recipe_ids"]) == 8 - len(
            fold["held_out_recipe_ids"]
        )
        assert len(fold["model"]["fit_fraction_groups"]) == 6


def test_quality_rejects_never_train_or_receive_scores(csi_rows, tank):
    csi_rows.loc[3, "quality_flags"] = "clipped_dark_channels"
    result = train_test_csi(csi_rows, tank)
    rejected_key = csi_rows.loc[3, "vial_key"]
    assert (
        result["scores"][result["scores"].vial_key.eq(rejected_key)]
        .delta_e.isna()
        .all()
    )
    assert all(rejected_key not in fold["train_vial_keys"] for fold in result["folds"])


def test_pre_fill_columns_never_affect_training_or_scores(csi_rows, tank):
    before = train_test_csi(csi_rows, tank)
    csi_rows[["empty_L", "empty_a", "empty_b"]] = [90, -40, 70]
    after = train_test_csi(csi_rows, tank)
    pd.testing.assert_frame_equal(before["scores"], after["scores"])
    assert not after["selection"]["prefill_subtraction_applied"]


def test_invalid_gamut_is_withheld_instead_of_clamped_to_a_color():
    target, reason = _predict_transfer(
        [120, 100, 80],
        {
            "status": "fitted",
            "coefficients": [[-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        },
    )
    assert target is None
    assert "outside the RGB gamut" in reason


def test_alias_recipe_ids_of_identical_composition_share_one_heldout_fold(
    csi_rows, tank
):
    alias = csi_rows[csi_rows.recipe_id.eq(5)].copy()
    alias["recipe_id"] = 99
    alias["run_id"] = "csi/video5/run99"
    alias["vial"] += len(csi_rows)
    alias["vial_key"] = alias.vial.map(lambda vial: f"csi/video5/vial{vial}")
    result = train_test_csi(pd.concat([csi_rows, alias], ignore_index=True), tank)
    shared = next(fold for fold in result["folds"] if 99 in fold["held_out_recipe_ids"])
    assert shared["held_out_recipe_ids"] == [5, 8, 99]
    assert len(shared["test_vial_keys"]) == 12
    assert not {5, 8, 99}.intersection(shared["train_recipe_ids"])


def test_same_fraction_training_recipes_do_not_identify_a_transfer(csi_rows, tank):
    for index, recipe_id in enumerate(sorted(csi_rows.recipe_id.unique()), 1):
        csi_rows.loc[csi_rows.recipe_id.eq(recipe_id), ML] = [index * 10, 0, 0]
    result = train_test_csi(csi_rows, tank)
    assert result["selection"]["fitted_folds"] == 0
    assert result["selection"]["primary_method"] is None
    assert all(
        "four distinct training fraction groups" in fold["model"]["reason"]
        for fold in result["folds"]
    )


def test_no_normalized_feature_group_crosses_the_train_test_boundary(csi_rows, tank):
    result = train_test_csi(csi_rows, tank)
    indexed = csi_rows.set_index("vial_key")
    for fold in result["folds"]:
        train = indexed.loc[fold["train_partition_vial_keys"], ML].to_numpy(float)
        test = indexed.loc[fold["test_vial_keys"], ML].to_numpy(float)
        train_keys = {tuple(np.round(row, 12)) for row in fractions(train)}
        test_keys = {tuple(np.round(row, 12)) for row in fractions(test)}
        assert train_keys.isdisjoint(test_keys)
    shared = next(
        fold for fold in result["folds"] if fold["held_out_recipe_ids"] == [5, 8]
    )
    assert shared["held_out_compositions_ml"] == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
    ]
    np.testing.assert_allclose(shared["held_out_fractions"], [1 / 3, 1 / 3, 1 / 3])
