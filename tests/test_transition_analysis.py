import pandas as pd
import pytest

from changeover_guard.transition_analysis import (
    label_inventory,
    load_session_labels,
    measure_video,
    recovery_curves,
    transition_support,
    validate_frame_labels,
    validate_vials,
)


def measured_sequence(colors, recipes=None):
    recipes = recipes or [1, 1] + [2] * (len(colors) - 2)
    rows = [
        {
            "vial": i + 1,
            "recipe_id": recipe,
            "start_s": i * 2,
            "end_s": i * 2 + 1,
            "t100_ml": 40 if recipe == 1 else 20,
            "t200_ml": 0 if recipe == 1 else 20,
            "t400_ml": 0,
        }
        for i, recipe in enumerate(recipes)
    ]
    table = validate_vials(pd.DataFrame(rows), "session_a")
    table[["r", "g", "b"]] = colors
    table["usable"] = True
    return table


def clean_target(rgb=(120, 100, 80)):
    return pd.DataFrame(
        [
            {
                "recipe_id": 2,
                "source_run_id": "independent/run",
                "source_session_id": "controls",
                "r": rgb[0],
                "g": rgb[1],
                "b": rgb[2],
                "t100_ml": 20,
                "t200_ml": 20,
                "t400_ml": 0,
            }
        ]
    )


def test_supplied_wall_clock_schema_and_pasted_emphasis(tmp_path):
    recipe = tmp_path / "vial_recipes.csv"
    recipe.write_text(
        "vial,recipe_id,type,t100_ml,t200_ml,t400_ml,total_ml,dispense_order,**start_time,**end_time,status,note\n"
        "1,1,Single 100,75,,,75,T100,**2:04:48,**2:05:17,ok,\n"
    )
    frames = tmp_path / "vial_fill_frame_labels.csv"
    frames.write_text(
        "filename,timestamp,video_s,in_fill,vial,recipe_id,type,t100_ml,**t200_ml,**t400_ml,total_ml,dispense_order,color_purpose,status,note\n"
        "vial_fill_20261030_020255_390000.jpg,2026-10-30 02:02:55.390000,0.000,0,,,between_fills,,**,**,,,,,\n"
    )
    labels, timing = load_session_labels(recipe, frames, "10_29_26")
    assert labels.start_s.iloc[0] == pytest.approx(112.61)
    assert labels.end_s.iloc[0] == pytest.approx(141.61)
    assert labels.requested_ml.iloc[0] == 75
    assert timing["video_origin"] == "2026-10-30T02:02:55.390000"
    assert validate_frame_labels(frames, labels)["status"] == "audited"


def test_midnight_rollover_and_inconsistent_video_origin(tmp_path):
    recipe = tmp_path / "recipes.csv"
    recipe.write_text(
        "vial,recipe_id,t100_ml,t200_ml,t400_ml,start_time,end_time\n1,1,40,0,0,00:00:01,00:00:10\n"
    )
    labels, _ = load_session_labels(recipe, video_start="2026-10-29 23:59:50")
    assert labels.start_s.iloc[0] == 11 and labels.end_s.iloc[0] == 20
    frames = tmp_path / "frames.csv"
    frames.write_text(
        "timestamp,video_s\n2026-10-29 23:59:50,0\n2026-10-29 23:59:52,1\n"
    )
    with pytest.raises(ValueError, match="video origin"):
        load_session_labels(recipe, frames)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"vial": [1, 1, 3, 4, 5]}, "exactly one"),
        ({"vial": [1, 2, 4, 5, 6]}, "missing vial"),
        ({"start_s": [0, 2, 1, 6, 8]}, "ordered"),
        ({"t200_ml": [0, 0, -10, 20, 20]}, "outside"),
        ({"t100_ml": [40, 40, 20, 30, 20]}, "changes composition"),
    ],
)
def test_label_validation_rejects_ambiguous_sequences(change, match):
    table = measured_sequence([[100, 100, 100]] * 5)
    for key, values in change.items():
        table[key] = values
    with pytest.raises(ValueError, match=match):
        validate_vials(table)


def test_same_recipe_vials_keep_one_run_and_no_false_transition():
    table = measured_sequence([[100, 100, 100]] * 5)
    assert table.run_id.nunique() == 2
    assert table.is_transition.sum() == 1
    curves, summary, _ = recovery_curves(table, clean_target())
    assert len(summary) == 1 and len(curves) == 3
    assert curves.cumulative_ml.tolist() == [40, 80, 120]
    assert summary.previous_total_processed_ml.iloc[0] == 80


def test_clearance_needs_three_consecutive_samples_and_reports_confirmation():
    clean = [120, 100, 80]
    table = measured_sequence(
        [[50, 50, 50]] * 2 + [[160, 50, 30], clean, clean, clean, clean]
    )
    curves, summary, _ = recovery_curves(table, clean_target())
    result = summary.iloc[0]
    assert not result.right_censored and result.vials_to_clear == 2
    assert result.volume_to_clear_ml == 80 and result.volume_to_confirm_ml == 160
    assert curves.delta_e_to_clean.iloc[0] > 2.2
    assert result.production_status == "PENDING"


def test_bad_measurement_breaks_consecutive_recovery_instead_of_disappearing():
    table = measured_sequence([[120, 100, 80]] * 6)
    table.loc[3, "usable"] = False
    curves, summary, _ = recovery_curves(table, clean_target())
    assert len(curves) == 4 and summary.right_censored.iloc[0]
    assert curves.delta_e_to_clean.isna().sum() == 1


def test_tail_target_cannot_evaluate_itself_or_train_a_model():
    table = measured_sequence([[120, 100, 80]] * 6)
    curves, summary, targets = recovery_curves(table)
    assert targets[0]["source"] == "same_run_tail_surrogate"
    assert curves.target_withheld.sum() == 2
    assert curves.loc[curves.target_withheld, "delta_e_to_clean"].isna().all()
    assert summary.right_censored.iloc[0]
    assert not transition_support(summary, curves)["shared_decay_supported"]


def test_one_session_and_many_frames_do_not_establish_independent_transitions():
    table = measured_sequence([[120, 100, 80]] * 6)
    curves, summary, _ = recovery_curves(table, clean_target())
    expanded = pd.concat([curves] * 100, ignore_index=True)
    support = transition_support(summary, expanded)
    assert support["sessions"] == 1 and not support["shared_decay_supported"]


def test_frame_recipe_mismatch_is_rejected(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text("filename,in_fill,vial,recipe_id\na.jpg,1,1,99\n")
    with pytest.raises(ValueError, match="disagree"):
        validate_frame_labels(path, measured_sequence([[100, 100, 100]] * 4))


def test_blackout_vial_is_preserved_without_inventing_time_or_volume(tmp_path):
    path = tmp_path / "recipes.csv"
    path.write_text(
        "vial,recipe_id,t100_ml,t200_ml,t400_ml,start_time,end_time,status\n"
        "1,1,40,0,0,00:00:01,00:00:02,ok\n"
        "2,1,40,0,0,,,error\n"
        "3,1,40,0,0,00:00:05,00:00:06,ok\n"
    )
    labels, _ = load_session_labels(path, video_start="2026-10-29 00:00:00")
    assert len(labels) == 3
    assert labels.timing_observable.tolist() == [True, False, True]
    assert labels.volume_usable.tolist() == [True, False, False]
    assert pd.isna(labels.start_s.iloc[1]) and pd.isna(labels.end_s.iloc[1])
    inventory, _, _ = label_inventory(labels)
    assert inventory["unobservable_vials"] == ["video5/vial2"]
    labels.loc[1, "status"] = "ok"
    with pytest.raises(ValueError, match="explicit non-ok"):
        validate_vials(labels)


def test_committed_october_29_labels_and_identity_audit():
    from pathlib import Path

    folder = Path(__file__).parents[1] / "data/10_29_26/vial_filled/labels"
    if not (folder / "vial_recipes.csv").exists():
        pytest.skip("October 29 label CSVs unavailable")
    labels, _ = load_session_labels(
        folder / "vial_recipes.csv", folder / "vial_fill_frame_labels.csv", "10_29_26"
    )
    audit = validate_frame_labels(folder / "vial_fill_frame_labels.csv", labels)
    inventory, runs, changes = label_inventory(labels)
    assert audit["frames"] == 1187 and len(labels) == 27 and len(runs) == 7
    assert len(changes) == 6 and inventory["sessions"] == 1
    assert inventory["mixture_recipes"] == 0
    assert inventory["unobservable_vials"] == ["10_29_26/vial10"]
    assert not transition_support(changes, pd.DataFrame())["shared_decay_supported"]


def test_blackout_measurement_never_seeks_to_an_invented_time(tmp_path, monkeypatch):
    from changeover_guard.vial_color import VialCalibration

    class Capture:
        def isOpened(self):
            return True

        def get(self, _):
            return 10.0

        def set(self, *_):
            raise AssertionError("An unobservable vial must not seek a video frame")

        def release(self):
            pass

    monkeypatch.setattr(
        "changeover_guard.transition_analysis.cv2.VideoCapture", lambda _: Capture()
    )
    table = validate_vials(
        pd.DataFrame(
            [
                {
                    "vial": 1,
                    "recipe_id": 1,
                    "start_s": float("nan"),
                    "end_s": float("nan"),
                    "status": "error",
                    "t100_ml": 40,
                    "t200_ml": 0,
                    "t400_ml": 0,
                }
            ]
        )
    )
    result = measure_video(table, "blackout.mp4", VialCalibration(), tmp_path)
    assert len(result) == 1 and result.sample_frames.iloc[0] == 0
    assert not result.usable.iloc[0] and result[["r", "g", "b"]].isna().all().all()


def test_unknown_error_volume_cannot_become_a_numeric_recovery_volume():
    table = measured_sequence([[120, 100, 80]] * 6)
    table["status"] = "ok"
    table.loc[2, "status"] = "error"
    table = validate_vials(table)
    table.loc[2, "usable"] = False
    _, summary, _ = recovery_curves(table, clean_target())
    assert not summary.right_censored.iloc[0]  # three later color observations
    assert pd.isna(summary.volume_to_clear_ml.iloc[0])
    assert not summary.volume_history_usable.iloc[0]
