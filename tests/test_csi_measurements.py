"""Controlled decoding/identity tests, never real camera validation evidence."""

import cv2
import numpy as np
import pandas as pd
import pytest

from changeover_guard import csi_measurements as csi
from changeover_guard.diagnostic_baselines import trusted_rows


def write_labels(root, *, video_id=5, rows=None):
    labels = root / "vial_fill" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [
            {
                "vial": 1,
                "recipe_id": 1,
                "start_s": 0.0,
                "end_s": 1.5,
                "t100_ml": 40,
                "t200_ml": 0,
                "t400_ml": 0,
                "dispense_order": "T100",
            },
            {
                "vial": 2,
                "recipe_id": 2,
                "start_s": 1.6,
                "end_s": 3.0,
                "t100_ml": 0,
                "t200_ml": 0,
                "t400_ml": 40,
                "dispense_order": "T400",
            },
        ]
    path = labels / f"video{video_id}_vial_recipes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_video(root, *, alternating=False):
    video = root / "vial_fill" / "videos" / "vial_fill_video_5.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480)
    )
    if not writer.isOpened():
        pytest.skip("mp4v software test encoder unavailable")
    for index in range(32):
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        image[210:255, 265:375] = 15  # holder is in the measurement position
        if alternating:
            rgb = [110, 95, 70] if index % 2 == 0 else [30, 60, 100]
        else:
            rgb = [110, 95, 70] if index < 16 else [30, 60, 100]
        image[375:397, 295:327] = rgb[::-1]
        writer.write(image)
    writer.release()
    return video


def test_missing_csi_data_never_falls_back_to_october_collection(tmp_path):
    unrelated = tmp_path / "data" / "10_29_26" / "vial_filled"
    unrelated.mkdir(parents=True)
    result = csi.load_csi_measurements(tmp_path / "data" / "csi-vial-fill-dataset")
    assert result["measurements"].empty
    assert "source_clock_verified" in result["measurements"]
    assert result["audit"] == []
    assert len(result["missing_inputs"]) == 1
    assert "csi-vial-fill-dataset" in result["missing_inputs"][0]


def test_label_rows_survive_a_missing_recording(tmp_path):
    path = write_labels(tmp_path)
    result = csi.load_csi_measurements(tmp_path)
    rows = result["measurements"]
    assert rows.vial.tolist() == [1, 2]
    assert not rows.usable.any()
    assert not trusted_rows(rows).any()
    assert rows.quality_flags.eq("missing_video").all()
    assert rows.source_video.str.endswith("vial_fill_video_5.mp4").all()
    assert result["audit"][0]["recipe_csv"] == str(path)
    assert result["audit"][0]["status"] == "missing_video"


@pytest.mark.parametrize("relative", [".", "vial_fill", "vial_fill/labels"])
def test_loader_accepts_wrapper_camera_or_labels_directory(tmp_path, relative):
    write_labels(tmp_path)
    result = csi.load_csi_measurements(tmp_path / relative)
    assert len(result["measurements"]) == 2


def test_labeled_video_measurement_does_not_discard_bright_filled_vials(tmp_path):
    write_labels(tmp_path)
    video = make_video(tmp_path)
    result = csi.load_csi_measurements(tmp_path, sample_stride_s=0.1)
    rows = result["measurements"]
    assert len(rows) == 2
    assert rows.usable.all()
    assert trusted_rows(rows).all()
    assert rows.session_id.eq("csi/video5").all()
    assert rows.source_video.eq(str(video)).all()
    assert rows.source_fps.eq(10).all()
    assert rows.measured_fill_ml.isna().all()
    assert not rows.fill_calibration_verified.any()
    assert rows.volume_source.eq("requested_proxy").all()
    assert rows.roi_pixels.eq("[295, 375, 327, 397]").all()
    assert rows.iloc[0].L > 33  # old brightness-defined cycle finder would drop this
    np.testing.assert_allclose(
        rows.iloc[0][["r", "g", "b"]].to_numpy(float), [110, 95, 70], atol=5
    )
    np.testing.assert_allclose(
        rows.iloc[1][["r", "g", "b"]].to_numpy(float), [30, 60, 100], atol=5
    )
    for row in rows.itertuples():
        assert (
            row.start_s
            <= row.stable_t_start
            <= row.stable_t_measure
            <= row.stable_t_end
            < row.end_s
        )
        assert row.start_s <= min(row.sample_times_s)
        assert max(row.sample_times_s) < row.end_s
        assert row.stable_decoded_t_measure == pytest.approx(
            row.stable_t_measure, abs=0.15
        )
    assert not any(column.startswith("empty_") for column in rows)
    assert not result["missing_inputs"]
    assert result["audit"][0]["prefill_subtraction_applied"] is False


def test_unstable_fill_does_not_receive_a_guessed_color(tmp_path):
    write_labels(tmp_path)
    make_video(tmp_path, alternating=True)
    rows = csi.load_csi_measurements(tmp_path, sample_stride_s=0.1)["measurements"]
    assert not rows.usable.any()
    assert rows[["r", "g", "b"]].isna().all().all()
    assert rows.quality_flags.str.contains("no_stable_filled_window").all()


def test_window_beyond_video_duration_is_not_silently_truncated(tmp_path):
    path = write_labels(tmp_path)
    labels = pd.read_csv(path)
    labels.loc[1, "end_s"] = 10
    labels.to_csv(path, index=False)
    make_video(tmp_path)
    rows = csi.load_csi_measurements(tmp_path, sample_stride_s=0.1)["measurements"]
    assert rows.iloc[0].usable
    assert not rows.iloc[1].usable
    assert "labeled_window_exceeds_video" in rows.iloc[1].quality_flags


def test_stable_window_cannot_bridge_invalid_or_missing_samples():
    def sample(time, usable=True):
        return {"t_s": time, "usable": usable, "L": 40, "a": 1, "b": 2}

    samples = [sample(0), sample(1), sample(2, False), sample(3), sample(4)]
    assert (
        csi._last_stable_window(samples, stride_s=1, min_frames=3, max_step_delta_e=1.5)
        == []
    )
    samples = [sample(0), sample(1), sample(10), sample(11)]
    assert (
        csi._last_stable_window(samples, stride_s=1, min_frames=3, max_step_delta_e=1.5)
        == []
    )


def test_early_stable_patch_cannot_replace_rejected_later_filled_samples(
    tmp_path, monkeypatch
):
    write_labels(
        tmp_path,
        rows=[
            {
                "vial": 1,
                "recipe_id": 1,
                "start_s": 0.0,
                "end_s": 3.0,
                "t100_ml": 40,
                "t200_ml": 0,
                "t400_ml": 0,
            }
        ],
    )
    make_video(tmp_path)
    original = csi._read_sample

    def later_reject(cap, frame_index, fps):
        record = original(cap, frame_index, fps)
        if frame_index >= 5:
            record["usable"] = False
            record["flags"].append("clipped_dark_channels")
        return record

    monkeypatch.setattr(csi, "_read_sample", later_reject)
    row = csi.load_csi_measurements(tmp_path, sample_stride_s=0.1)["measurements"].iloc[
        0
    ]
    assert not row.usable
    assert pd.isna(row.r)
    assert "stable_window_not_near_labeled_end" in row.quality_flags


def test_mismatched_decoded_clock_never_verifies_an_observation(tmp_path, monkeypatch):
    write_labels(tmp_path)
    make_video(tmp_path)
    original = csi._read_sample

    def mismatched_sample(cap, frame_index, fps):
        record = original(cap, frame_index, fps)
        record.update(usable=False, clock_ok=False)
        record["flags"].append("decoded_time_mismatch")
        return record

    monkeypatch.setattr(csi, "_read_sample", mismatched_sample)
    rows = csi.load_csi_measurements(tmp_path, sample_stride_s=0.1)["measurements"]
    assert not rows.source_clock_verified.any()
    assert not rows.usable.any()
    assert rows.quality_flags.str.contains("decoded_time_mismatch").all()


def test_recipe_order_must_match_dispensed_components(tmp_path):
    path = write_labels(tmp_path)
    labels = pd.read_csv(path)
    labels.loc[0, "dispense_order"] = "T400"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="dispense_order"):
        csi.load_csi_measurements(tmp_path)


def test_duplicate_label_files_for_one_recording_are_rejected(tmp_path):
    path = write_labels(tmp_path)
    path.with_name("vial_recipes.csv").write_text(path.read_text())
    with pytest.raises(ValueError, match="same CSI video"):
        csi.load_csi_measurements(tmp_path)


def test_wall_clock_labels_are_not_treated_as_csi_video_seconds(tmp_path):
    path = write_labels(tmp_path)
    labels = pd.read_csv(path).drop(columns=["start_s", "end_s"])
    labels["start_time"], labels["end_time"] = "02:00:00", "02:00:20"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="explicit start_s/end_s"):
        csi.load_csi_measurements(tmp_path)


def test_unobservable_error_label_is_retained(tmp_path):
    path = write_labels(tmp_path)
    labels = pd.read_csv(path)
    labels["status"] = "ok"
    labels.loc[1, ["start_s", "end_s"]] = np.nan
    labels.loc[1, "status"] = "error"
    labels.to_csv(path, index=False)
    make_video(tmp_path)
    rows = csi.load_csi_measurements(tmp_path, sample_stride_s=0.1)["measurements"]
    assert rows.iloc[0].usable
    assert not rows.iloc[1].usable
    assert "unobservable_vial_window" in rows.iloc[1].quality_flags


def test_distinct_recordings_use_unique_session_identity(tmp_path):
    write_labels(tmp_path, video_id=5)
    write_labels(tmp_path, video_id=6)
    rows = csi.load_csi_measurements(tmp_path)["measurements"]
    assert rows.session_id.nunique() == 2
    assert rows.vial_key.is_unique
    assert len(rows) == 4


def test_recipe_id_cannot_change_composition_between_videos(tmp_path):
    write_labels(tmp_path, video_id=5)
    path = write_labels(tmp_path, video_id=6)
    labels = pd.read_csv(path)
    labels.loc[0, "t100_ml"] = 50
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="changes composition across videos"):
        csi.load_csi_measurements(tmp_path)
