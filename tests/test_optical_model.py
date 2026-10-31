"""Known mathematical fixtures exercise fitting and gates, not real accuracy."""

import numpy as np
import pandas as pd
import pytest

from changeover_guard.contracts import Recipe
from changeover_guard.optical_model import (
    CleanOpticalModel,
    average_decay,
    fit_carryover_decay,
    fit_clean_optical,
    quality_decision,
)
from changeover_guard.transition_analysis import transition_support
from changeover_guard.vial_color import (
    VialCalibration,
    encoded_rgb,
    linear_rgb,
    repeatability_report,
)

RECIPES = [
    [75, 0, 0],
    [50, 10, 0],
    [20, 50, 0],
    [20, 0, 50],
    [30, 0, 10],
    [20, 20, 20],
    [30, 10, 20],
    [20, 10, 30],
]
COEFFICIENTS = np.array([[0.0, 0.0, 0.0], [0.2, 2.0, 3.0], [3.0, 1.0, 0.1]])
BLANK = [220, 215, 210]


def optical_controls(calibration):
    rows = []
    for session in range(3):
        for recipe_id, quantities in enumerate(RECIPES, 1):
            fractions = np.asarray(quantities) / sum(quantities)
            rgb = encoded_rgb(linear_rgb(BLANK) * np.exp(-fractions @ COEFFICIENTS))
            rows.append(
                {
                    "control_id": f"{session}/{recipe_id}",
                    "source_session_id": f"controls_{session}",
                    "source_run_id": f"controls_{session}/run{recipe_id}",
                    "recipe_id": recipe_id,
                    "camera_id": calibration.camera_id,
                    "calibration_id": calibration.calibration_id,
                    "is_clean_control": True,
                    **dict(zip(["r", "g", "b"], rgb)),
                    **dict(zip(["t100_ml", "t200_ml", "t400_ml"], quantities)),
                }
            )
    return pd.DataFrame(rows)


def proven_study(calibration):
    rows = [
        {
            "sample_id": s,
            "session_id": session,
            "capture_id": f"{s}/{session}/{capture}",
            "camera_id": calibration.camera_id,
            "calibration_id": calibration.calibration_id,
            "r": 50 + s * 30,
            "g": 100,
            "b": 120,
        }
        for s in range(5)
        for session in range(2)
        for capture in range(5)
    ]
    return repeatability_report(pd.DataFrame(rows), calibration)


@pytest.fixture
def optical_setup():
    # Deliberate idealized chart metadata; never written as real camera evidence.
    calibration = VialCalibration(
        settings_locked=True,
        alignment_reviewed=True,
        validation_delta_e=(0.0,) * 6,
        chart_sha256="software_fixture",
    )
    controls = optical_controls(calibration)
    return calibration, controls, fit_clean_optical(controls, calibration)


def test_constrained_model_recovers_known_linear_light_absorbance(optical_setup):
    calibration, _, model = optical_setup
    assert np.asarray(model.absorbance) == pytest.approx(COEFFICIENTS)
    assert np.min(model.absorbance) >= 0
    assert model.validation["independent_recipe_folds"] == 7
    assert model.validation["p95_delta_e"] < 1e-8
    assert model.validation["uncertainty_delta_e"] is None
    restored = CleanOpticalModel.from_dict(model.to_dict())
    assert restored.predict(Recipe(1, {"T100": 75}), calibration.calibration_id)[
        "rgb"
    ] == pytest.approx(BLANK)


def test_equal_fractions_predict_equal_clean_color_and_t300_is_rejected(optical_setup):
    _, _, model = optical_setup
    a = model.predict(Recipe(100, {"T100": 20, "T200": 10}))
    b = model.predict(Recipe(101, {"T100": 40, "T200": 20}))
    assert a["rgb"] == pytest.approx(b["rgb"])
    with pytest.raises(ValueError, match="usable tank"):
        model.predict(Recipe(102, {"T300": 20}))
    with pytest.raises(ValueError, match="positive total"):
        model.predict(Recipe(103, {}))
    with pytest.raises(ValueError, match="different camera"):
        model.predict(Recipe(1, {"T100": 75}), "other_camera")


def test_singular_controls_and_unverified_clean_controls_are_rejected(optical_setup):
    calibration, controls, _ = optical_setup
    with pytest.raises(ValueError, match="independently vary"):
        fit_clean_optical(controls[controls.recipe_id.isin([1, 2, 3])], calibration)
    controls.loc[0, "is_clean_control"] = False
    with pytest.raises(ValueError, match="verified clean"):
        fit_clean_optical(controls, calibration)


def test_quality_remains_pending_without_real_repeatability_evidence(optical_setup):
    calibration, _, model = optical_setup
    recipe = Recipe(1, {"T100": 75})
    result = quality_decision(
        BLANK, recipe, model, calibration, repeatability_report(None, calibration)
    )
    assert (
        result["status"] == "PENDING" and result["prediction_error_p95_delta_e"] is None
    )
    assert (
        quality_decision(BLANK, recipe, model, calibration, {"proven": True})["status"]
        == "PENDING"
    )


def test_quality_uses_clean_color_and_strict_threshold(optical_setup, monkeypatch):
    calibration, _, model = optical_setup
    study = proven_study(calibration)
    assert study["proven"]
    recipe = Recipe(1, {"T100": 75})
    assert (
        quality_decision(BLANK, recipe, model, calibration, study)["status"] == "PASS"
    )
    assert (
        quality_decision([100, 50, 30], recipe, model, calibration, study)["status"]
        == "FAIL"
    )
    monkeypatch.setattr(
        "changeover_guard.optical_model.color_errors", lambda *args: np.asarray(2.2)
    )
    assert (
        quality_decision(BLANK, recipe, model, calibration, study)["status"] == "FAIL"
    )


def test_quality_rejects_extrapolation_and_bad_model_validation(optical_setup):
    calibration, _, model = optical_setup
    study = proven_study(calibration)
    assert (
        quality_decision(
            BLANK, Recipe(99, {"T200": 10, "T400": 50}), model, calibration, study
        )["status"]
        == "PENDING"
    )
    model.validation["p95_delta_e"] = 3.0
    assert (
        quality_decision(BLANK, Recipe(1, {"T100": 75}), model, calibration, study)[
            "status"
        ]
        == "PENDING"
    )


def decay_observations(sessions=3, constant=False):
    records, summaries = [], []
    target = np.array([180, 175, 170])
    for session in range(sessions):
        for transition in range(2):
            name = f"session_{session}/run{transition}"
            starts, ends = np.arange(8) * 25.0, np.arange(1, 9) * 25.0
            residual = np.array([0.3, -0.2, 0.15]) * (1 + transition / 4)
            decay = np.ones(8) if constant else average_decay(starts, ends, 80.0)
            rgb = encoded_rgb(linear_rgb(target) * np.exp(-decay[:, None] * residual))
            for i in range(8):
                records.append(
                    {
                        "transition_id": name,
                        "session_id": f"session_{session}",
                        "usable": True,
                        "target_source": "independent_clean_controls",
                        "delta_e_to_clean": 1.0,
                        "vial_in_run": i + 1,
                        "volume_start_ml": starts[i],
                        "cumulative_ml": ends[i],
                        **dict(zip(["r", "g", "b"], rgb[i])),
                        **dict(zip(["target_r", "target_g", "target_b"], target)),
                    }
                )
            summaries.append(
                {
                    "transition_id": name,
                    "session_id": f"session_{session}",
                    "previous_recipe_id": transition + 1,
                    "next_recipe_id": transition + 2,
                }
            )
    curves, summary = pd.DataFrame(records), pd.DataFrame(summaries)
    return curves, transition_support(summary, curves)


def test_vial_decay_is_integrated_over_collection_volume():
    value = average_decay([0, 20], [20, 40], 50.0)
    assert value[0] == pytest.approx(50 / 20 * (1 - np.exp(-20 / 50)))
    assert value[1] < value[0]


def test_shared_decay_recovers_tau_with_whole_session_validation():
    curves, support = decay_observations()
    result = fit_carryover_decay(curves, support, BLANK)
    assert result["status"] == "fitted_exploratory"
    assert result["tau_ml"] == pytest.approx(80, rel=0.001)
    assert len(result["validation"]) == 3
    assert result["tau_interval_ml"] is None


def test_more_frames_cannot_enable_decay_fitting():
    curves, support = decay_observations(sessions=1)
    result = fit_carryover_decay(pd.concat([curves] * 20), support, BLANK)
    assert result["status"] == "insufficient_data" and result["tau_ml"] is None


def test_constant_residual_does_not_support_exponential_decay():
    curves, support = decay_observations(constant=True)
    result = fit_carryover_decay(curves, support, BLANK)
    assert result["status"] == "unsupported" and result["tau_ml"] is None


def test_repeated_frames_do_not_count_as_four_distinct_vials():
    curves, _ = decay_observations()
    first_frames = curves[curves.vial_in_run.eq(1)]
    repeated_frames = pd.concat([first_frames] * 20)
    summaries = curves[["transition_id", "session_id"]].drop_duplicates()
    summaries["previous_recipe_id"], summaries["next_recipe_id"] = 1, 2
    assert not transition_support(summaries, repeated_frames)["shared_decay_supported"]


def test_uncertainty_requires_proven_camera_and_independent_sessions(optical_setup):
    curves, support = decay_observations(sessions=5)
    calibration, _, _ = optical_setup
    unproven = fit_carryover_decay(curves, support, BLANK, calibration=calibration)
    assert unproven["tau_ml"] == pytest.approx(80, rel=0.001)
    assert unproven["tau_interval_ml"] is None
    proven = fit_carryover_decay(
        curves,
        support,
        BLANK,
        calibration=calibration,
        repeatability=proven_study(calibration),
    )
    assert proven["tau_interval_ml"] == pytest.approx([80, 80], rel=0.001)
