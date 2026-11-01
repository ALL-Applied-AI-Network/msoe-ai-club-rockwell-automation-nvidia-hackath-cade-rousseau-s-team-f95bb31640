import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from skimage.color import lab2rgb

from changeover_guard import advise, measure, mixing
from changeover_guard.contracts import Prediction, Recipe, TankReading, VialMeasurement
from changeover_guard.diagnostic_baselines import (
    METHODS,
    compare_baselines,
    extract_tank_baselines,
    tank_optical_prediction,
)
from changeover_guard.diagnostic_workflow import dashboard_html, run_operator_workflow
from changeover_guard.operator_diagnostics import diagnose_vials, trajectory_evidence
from changeover_guard.vial_color import lab_values


def sequence():
    """Four independent-run SOFTWARE controls, not evidence about camera accuracy."""
    dark, light = [30, 60, 90], [150, 145, 140]
    rows = []
    for run_index, (recipe, rgb) in enumerate(
        [(1, dark), (2, light), (1, dark), (2, light)], 1
    ):
        for position in range(1, 6):
            observed = rgb
            if run_index in (2, 4) and position < 3:
                alpha = 0.5 if position == 1 else 0.2
                lab = (1 - alpha) * lab_values(light) + alpha * lab_values(dark)
                observed = 255 * lab2rgb(lab)
            vial = len(rows) + 1
            rows.append(
                {
                    "session_id": "software",
                    "vial": vial,
                    "vial_key": f"software/vial{vial}",
                    "recipe_id": recipe,
                    "run_id": f"run{run_index}",
                    "vial_in_run": position,
                    "t100_ml": 50 if recipe == 2 else 0,
                    "t200_ml": 0,
                    "t400_ml": 50 if recipe == 1 else 0,
                    "requested_ml": 50.0,
                    "cumulative_ml": position * 50.0,
                    "volume_usable": True,
                    "volume_source": "requested_proxy",
                    "r": observed[0],
                    "g": observed[1],
                    "b": observed[2],
                    "usable": True,
                    "source_clock_verified": True,
                    "quality_flags": "",
                    "source_timestamp_age_s": 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def no_tank():
    return {"status": "missing_input", "reason": "Tank photo missing", "tanks": {}}


def test_whole_run_holdout_has_no_self_reference_and_keeps_all_vials(no_tank):
    data = sequence()
    result = compare_baselines(data, no_tank)
    assert len(result["scores"]) == 2 * len(data)
    for row in result["scores"].query("method == 'B_vial_camera'").itertuples():
        assert row.run_id not in json.loads(row.reference_runs)
        assert row.vial_key not in json.loads(row.reference_vial_keys)
    assert result["selection"]["primary_method"] == METHODS[1]
    assert result["selection"]["status"] == "provisional_only"
    assert result["selection"]["common_scored_vials"] == 0


def test_common_cohort_is_identical_when_both_methods_have_inputs():
    data = sequence()
    tank = {
        "status": "measured",
        "tanks": {
            "T100": {"rgb": [150, 145, 140]},
            "T200": {"rgb": [180, 60, 30]},
            "T400": {"rgb": [30, 60, 90]},
        },
    }
    result = compare_baselines(data, tank)
    common = result["scores"][result["scores"].paired_comparison]
    keys = [set(common[common.method.eq(method)].vial_key) for method in METHODS]
    assert keys[0] == keys[1] == set(data.vial_key)
    assert result["selection"]["status"] == "exploratory_comparison"
    assert not result["selection"]["production_validated"]


def test_changing_held_out_vials_does_not_change_their_empirical_target(no_tank):
    data = sequence()
    before = compare_baselines(data, no_tank)["scores"]
    data.loc[data.run_id.eq("run2"), ["r", "g", "b"]] = [90, 110, 125]
    after = compare_baselines(data, no_tank)["scores"]
    use = before.method.eq(METHODS[1]) & before.run_id.eq("run2")
    np.testing.assert_allclose(
        before.loc[use, ["target_r", "target_g", "target_b"]],
        after.loc[use, ["target_r", "target_g", "target_b"]],
    )


@pytest.mark.parametrize("tank", ["T100", "T200", "T400"])
def test_optical_candidate_reproduces_pure_tank_endpoints(tank):
    colors = {"T100": [170, 150, 120], "T200": [180, 40, 25], "T400": [20, 60, 160]}
    result = tank_optical_prediction(Recipe(1, {tank: 40}), colors)
    np.testing.assert_allclose(result, colors[tank], atol=1e-10)


def test_clipped_tank_colors_cannot_produce_a_precise_candidate():
    with pytest.raises(ValueError, match="clipped"):
        tank_optical_prediction(
            Recipe(1, {"T100": 40}),
            {"T100": [100, 100, 100], "T200": [150, 20, 0], "T400": [20, 30, 60]},
        )


def test_tank_rois_use_exact_photograph_pixels_and_exclude_t300(tmp_path):
    image = np.full((700, 920, 3), 100, dtype=np.uint8)
    from changeover_guard.diagnostic_baselines import OCT29_TANK_ROIS

    colors = {"T100": [180, 180, 170], "T200": [160, 20, 15], "T400": [10, 40, 150]}
    for tank, (x1, y1, x2, y2) in OCT29_TANK_ROIS.items():
        image[round(y1 * 700) : round(y2 * 700), round(x1 * 920) : round(x2 * 920)] = (
            colors[tank][::-1]
        )
    path = tmp_path / "tank.png"
    assert cv2.imwrite(str(path), image)
    result = extract_tank_baselines(path)
    assert set(result["tanks"]) == {"T100", "T200", "T400"}
    assert not result["transfer_calibrated"]
    for tank, rgb in colors.items():
        np.testing.assert_allclose(result["tanks"][tank]["rgb"], rgb)


@pytest.mark.parametrize(
    "issue",
    [
        "stale_camera_frame",
        "clipped_dark_channels",
        "uneven_liquid_patch",
        "unobservable_vial_window",
    ],
)
def test_quality_flags_override_apparent_clean_color_and_other_hypotheses(
    no_tank, issue
):
    data = sequence()
    data.loc[4, "quality_flags"] = issue
    data.loc[4, "source_timestamp_age_s"] = 6.79
    result = diagnose_vials(data, compare_baselines(data, no_tank))
    row = result.iloc[4]
    assert row.diagnostic_state == "measurement_sensor_reject"
    assert pd.isna(row.delta_e)
    assert row.production_status == "PENDING"
    if issue == "stale_camera_frame":
        assert "6.79" in row.explanation


def test_false_csv_clock_string_is_not_trusted(no_tank):
    data = sequence()
    data["source_clock_verified"] = "True"
    data.loc[0, "source_clock_verified"] = "False"
    row = diagnose_vials(data, compare_baselines(data, no_tank)).iloc[0]
    assert row.diagnostic_state == "measurement_sensor_reject"


def test_decay_and_prior_recipe_trajectory_are_both_required(no_tank):
    data = sequence()
    result = diagnose_vials(data, compare_baselines(data, no_tank))
    first = result[result.run_id.eq("run2")].iloc[0]
    assert first.run_decay_supported
    assert first.trajectory_supported
    assert first.diagnostic_state == "expected_changeover_carryover"
    assert not first.root_cause_confirmed
    assert "retrospective" in first.trend_scope
    assert "do not subtract" in first.prefill_inspection_suggestion


def test_color_trajectory_rejects_a_shift_outside_both_recipe_endpoints():
    result = trajectory_evidence([190, 10, 15], [150, 145, 140], [30, 60, 90])
    assert not result["trajectory_supported"]


def test_requested_volume_is_not_a_measured_fill_sensor(no_tank):
    data = sequence()
    result = diagnose_vials(data, compare_baselines(data, no_tank))
    assert result.fill_status.eq("not_measured").all()
    assert result.measured_fill_error_ml.isna().all()
    assert result.production_status.eq("PENDING").all()


def test_verified_fill_fault_generates_action_without_inventing_a_cause(no_tank):
    data = sequence()
    data["measured_fill_ml"] = np.nan
    data["fill_calibration_verified"] = False
    data.loc[9, ["measured_fill_ml", "fill_calibration_verified"]] = [35.0, True]
    result = diagnose_vials(data, compare_baselines(data, no_tank))
    row = result.iloc[9]
    assert row.diagnostic_state == "persistent_tank_or_recipe_fault"
    assert row.measured_fill_error_ml == -15
    assert "dispense" in row.explanation
    assert not row.root_cause_confirmed


def test_no_pre_fill_subtraction_or_mutation_of_measured_colors(no_tank):
    data = sequence()
    data[["empty_L", "empty_a", "empty_b"]] = [90, -50, 40]
    before = data.copy(deep=True)
    first = diagnose_vials(data, compare_baselines(data, no_tank))
    data[["empty_L", "empty_a", "empty_b"]] = [10, 50, -40]
    second = diagnose_vials(data, compare_baselines(data, no_tank))
    pd.testing.assert_series_equal(first.delta_e, second.delta_e)
    pd.testing.assert_frame_equal(data[["r", "g", "b"]], before[["r", "g", "b"]])


def test_persistent_deviation_is_a_hypothesis_not_an_estimated_flush_requirement(
    no_tank,
):
    data = sequence()
    data.loc[data.run_id.eq("run2"), ["r", "g", "b"]] = [160, 90, 30]
    result = diagnose_vials(data, compare_baselines(data, no_tank))
    last = result[result.run_id.eq("run2")].iloc[-1]
    assert last.diagnostic_state == "persistent_tank_or_recipe_fault"
    assert "Expected clearance volume is not established" in last.explanation
    assert last.production_status == "PENDING"


def test_dashboard_escapes_user_supplied_notes(no_tank, tmp_path):
    data = sequence()
    result = diagnose_vials(data, compare_baselines(data, no_tank))
    result.loc[0, "explanation"] = "<script>bad()</script>"
    page = dashboard_html(result, tmp_path)
    assert "<script>bad()" not in page
    assert "&lt;script&gt;bad()&lt;/script&gt;" in page
    assert 'id="diagnostic-state"' in page


def test_missing_dataset_does_not_reuse_prior_results(tmp_path):
    result = run_operator_workflow(tmp_path, plots=False)
    assert result["summary"]["status"] == "missing_data"
    assert result["summary"]["vials"] == 0
    assert not result["summary"]["prefill_subtraction_applied"]
    assert result["tank"]["status"] == "missing_input"


def test_legacy_vial_adapter_measures_color_but_never_invents_fill():
    image = np.full((480, 640, 3), [30, 70, 110], dtype=np.uint8)
    result = measure.vial(image, datetime(2026, 10, 29))
    np.testing.assert_allclose(result.color, [110, 70, 30])
    assert np.isnan(result.fill_ml)
    prediction = Prediction(
        Recipe(1, {"T100": 50}), None, result.color, result.color, 0.0, False
    )
    with pytest.raises(advise.InspectionUnavailable, match="fill"):
        advise.check(prediction, result)


def test_legacy_check_compares_clean_color_not_contaminated_forecast():
    timestamp = datetime(2026, 10, 29)
    recipe = Recipe(1, {"T100": 50})
    prediction = Prediction(recipe, None, (150, 150, 150), (30, 60, 90), 30.0, True)
    result = advise.check(
        prediction, VialMeasurement(timestamp, prediction.first_vial_color, 50)
    )
    assert not result.ok
    advice = advise.advise(prediction, result, [], TankReading(timestamp, {}))
    assert not advice.flush and advice.next_recipe_id is None
    assert "pre-fill" in advice.cause


def test_legacy_prediction_abstains_without_a_calibrated_model():
    with pytest.raises(mixing.PredictionUnavailable, match="validated"):
        mixing.predict(
            None, Recipe(1, {"T100": 50}), TankReading(datetime(2026, 10, 29), {})
        )


def test_unmapped_tank_candidate_cannot_become_a_chemical_diagnosis(no_tank):
    data = sequence()
    benchmark = compare_baselines(data, no_tank)
    benchmark["selection"]["primary_method"] = METHODS[0]
    a = benchmark["scores"].method.eq(METHODS[0])
    benchmark["scores"].loc[a, "delta_e"] = 15.0
    result = diagnose_vials(data, benchmark)
    assert result.diagnostic_state.eq("evidence_pending").all()
    assert result.color_assessment.eq("cross_camera_candidate_only").all()


def test_duplicate_measurements_are_rejected_instead_of_double_counted(no_tank):
    data = sequence()
    with pytest.raises(ValueError, match="unique"):
        compare_baselines(pd.concat([data, data.iloc[:1]]), no_tank)


def test_optional_figures_preserve_missing_targets_and_all_vials(no_tank, tmp_path):
    pytest.importorskip("matplotlib")
    from changeover_guard.diagnostic_workflow import (
        plot_benchmark,
        plot_operator_timeline,
        plot_tank_roi,
    )

    data = sequence()
    benchmark = compare_baselines(data, no_tank)
    report = diagnose_vials(data, benchmark)
    plot_benchmark(data, benchmark["scores"], tmp_path)
    plot_operator_timeline(report, tmp_path)
    plot_tank_roi(no_tank, tmp_path)
    for filename in [
        "baseline_comparison.png",
        "operator_timeline.png",
        "tank_roi_verification.png",
    ]:
        assert (tmp_path / filename).stat().st_size > 1000


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "data/10_29_26/vial_filled/images"
    ).is_dir(),
    reason="October 29 raw images unavailable",
)
def test_oct29_real_workflow_preserves_27_labels_and_sensor_evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = run_operator_workflow(
        root, tank_image=tmp_path / "missing.jpg", plots=False
    )
    report = result["operator_report"]
    assert len(report) == 27
    assert report.trusted_color.sum() == 16
    assert report.diagnostic_state.eq("measurement_sensor_reject").sum() == 11
    assert report.production_status.eq("PENDING").all()
    assert "6.79" in report.loc[report.vial.eq(9), "explanation"].iloc[0]
    assert (
        report.loc[report.vial.eq(10), "diagnostic_state"].iloc[0]
        == "measurement_sensor_reject"
    )
    assert result["summary"]["prefill_subtraction_applied"] is False
    assert result["summary"]["selection"]["common_scored_vials"] == 0
    assert (
        result["benchmark"]["scores"]
        .query("method == 'B_vial_camera'")
        .delta_e.notna()
        .sum()
        == 10
    )
