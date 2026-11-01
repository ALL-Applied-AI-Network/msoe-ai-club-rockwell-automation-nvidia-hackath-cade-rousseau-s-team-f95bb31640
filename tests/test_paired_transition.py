import numpy as np
import pandas as pd

from src.color_monitor.paired_transition import (
    add_paired_empty_correction,
    transition_effect_table,
)
from src.color_monitor.paired_transition import (
    test_darkness_volume_hypothesis as evaluate_hypothesis,
)
from src.color_monitor.segment import measure_cycles


def _frame_rows(values):
    rows = []
    for t_s, (lightness, a_value, b_value) in enumerate(values):
        rows.append(
            {
                "video_id": 5,
                "t_s": float(t_s),
                "L": lightness,
                "a": a_value,
                "b": b_value,
                "carrier_present": True,
            }
        )
    return pd.DataFrame(rows)


def test_measure_cycles_retains_the_immediately_preceding_empty_vial_color():
    frames = _frame_rows(
        [(50.0, 1.0, 2.0)] * 5
        + [(20.0, 5.0, 6.0)] * 6
        + [(55.0, 3.0, 4.0)] * 5
        + [(21.0, 5.0, 6.0)] * 6
    )

    cycles = measure_cycles(frames)

    assert len(cycles) == 2
    assert cycles.empty_L.tolist() == [50.0, 55.0]
    assert cycles.empty_a.tolist() == [1.0, 3.0]
    assert cycles.empty_n_frames.tolist() == [5, 5]
    assert cycles.empty_spread_de.max() == 0.0
    assert cycles.empty_t_measure.tolist() == [0.0, 11.0]
    assert cycles.stable_t_measure.tolist() == [5.0, 16.0]


def test_paired_correction_removes_a_vial_specific_additive_lab_offset():
    vials = pd.DataFrame(
        {
            "L": [30.0, 40.0],
            "a": [1.0, 6.0],
            "b": [2.0, 7.0],
            "empty_L": [50.0, 60.0],
            "empty_a": [0.0, 5.0],
            "empty_b": [0.0, 5.0],
            "empty_n_frames": [5, 5],
            "empty_spread_de": [0.2, 0.2],
        }
    )

    corrected, canonical = add_paired_empty_correction(vials)

    np.testing.assert_allclose(canonical, [55.0, 2.5, 2.5])
    np.testing.assert_allclose(
        corrected[["corrected_L", "corrected_a", "corrected_b"]],
        [[35.0, 3.5, 4.5], [35.0, 3.5, 4.5]],
    )
    assert corrected.empty_pair_usable.all()
    assert corrected.background_adjustment_delta_e00.notna().all()


def test_paired_correction_does_not_borrow_a_different_vials_empty_color():
    vials = pd.DataFrame(
        {
            "L": [30.0, 40.0],
            "a": [1.0, 6.0],
            "b": [2.0, 7.0],
            "empty_L": [50.0, np.nan],
            "empty_a": [0.0, np.nan],
            "empty_b": [0.0, np.nan],
            "empty_n_frames": [5, 0],
            "empty_spread_de": [0.2, np.nan],
        }
    )

    corrected, _ = add_paired_empty_correction(vials)

    assert corrected.empty_pair_usable.tolist() == [True, False]
    assert corrected.loc[1, "empty_pair_reason"] == "paired_empty_color_missing"
    assert corrected.loc[1, "corrected_L"] != corrected.loc[1, "corrected_L"]


def test_empty_spread_is_diagnostic_unless_an_explicit_limit_is_supplied():
    vials = pd.DataFrame(
        {
            "L": [30.0],
            "a": [1.0],
            "b": [2.0],
            "empty_L": [50.0],
            "empty_a": [0.0],
            "empty_b": [0.0],
            "empty_n_frames": [5],
            "empty_spread_de": [8.0],
        }
    )

    diagnostic, _ = add_paired_empty_correction(vials)
    filtered, _ = add_paired_empty_correction(vials, max_empty_spread_de=2.3)

    assert diagnostic.empty_pair_usable.tolist() == [True]
    assert filtered.empty_pair_usable.tolist() == [False]
    assert filtered.empty_pair_reason.tolist() == ["paired_empty_color_unstable"]


def test_transition_table_and_hypothesis_capture_faster_dark_recipe_recovery():
    report_rows = []
    recipes = []
    for run_number, (recipe_id, settled_at) in enumerate(
        [(2, 4), (3, 3), (4, 2), (5, 1), (6, 1)], start=1
    ):
        for position in range(1, 5):
            report_rows.append(
                {
                    "vial_id": run_number * 10 + position,
                    "run_id": f"run{run_number}",
                    "recipe_id": recipe_id,
                    "previous_recipe_id": 1,
                    "position_in_run": position,
                    "corrected_verdict": (
                        "consistent" if position >= settled_at else "OUT OF RANGE"
                    ),
                }
            )
        recipes.append(
            {
                "recipe_id": recipe_id,
                "t100_ml": 10.0,
                "t200_ml": float(recipe_id * 10),
                "t400_ml": 0.0,
            }
        )
    refs = pd.DataFrame(
        {
            "recipe_id": [2, 3, 4, 5, 6],
            "L": [80.0, 70.0, 60.0, 50.0, 40.0],
            "status": ["ok"] * 5,
        }
    ).set_index("recipe_id")
    catalog = pd.DataFrame(recipes).set_index("recipe_id")

    transitions = transition_effect_table(
        pd.DataFrame(report_rows), refs, catalog
    )
    result = evaluate_hypothesis(transitions)

    darkness = result[result.predictor == "target_darkness"].iloc[0]
    dominant_ml = result[result.predictor == "dominant_ml"].iloc[0]
    assert darkness.spearman_rho < 0
    assert dominant_ml.spearman_rho < 0
    assert darkness.status == "direction_supported_exploratory"


def test_hypothesis_stays_insufficient_with_too_few_transitions():
    transitions = pd.DataFrame(
        {
            "target_darkness": [10.0, 20.0],
            "dominant_ml": [20.0, 30.0],
            "dominant_fraction": [0.5, 0.7],
            "pre_acceptable_vials": [2.0, 1.0],
        }
    )

    result = evaluate_hypothesis(transitions)

    assert set(result.status) == {"insufficient_data"}
