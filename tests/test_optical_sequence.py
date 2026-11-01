"""Controlled software fixtures, not real-camera performance measurements."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from changeover_guard.optical_evaluation import (
    attribute,
    evaluate_labeled,
    evaluate_unlabeled,
    fit_height_calibration,
    mixing_projection,
    smooth_recipes,
)
from changeover_guard.optical_knn import (
    FEATURES,
    FRACTIONS,
    VectorizedOpticalKNN,
    inspect_patch,
    optical_features,
)
from changeover_guard.optical_sequence import (
    SequenceConfig,
    UnlabeledVideoSegmenter,
    liquid_patch,
    load_labeled_sequences,
    portable_path,
    read_labels,
)
from changeover_guard.optical_workflow import (
    render_run_diagnostics,
    run_operator_workflow,
)


def labels():
    rows = []
    for i in range(30):
        group = i // 3
        rows.append(
            {
                "vial": i + 1,
                "recipe_id": group + 5,
                "t100_ml": 10 + group,
                "t200_ml": 20 - group,
                "t400_ml": 10,
                "total_ml": 40,
                "start_s": i * 2,
                "end_s": (i + 1) * 2,
                "start_frame": i * 60,
                "end_frame": (i + 1) * 60,
            }
        )
    return pd.DataFrame(rows)


def dataset(tmp_path, with_video=False):
    camera = tmp_path / "vial_fill"
    (camera / "labels").mkdir(parents=True)
    (camera / "videos").mkdir()
    path = camera / "labels/video5_vial_recipes.csv"
    labels().to_csv(path, index=False)
    if with_video:
        writer = cv2.VideoWriter(
            str(camera / "videos/vial_fill_video_5.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (128, 96),
        )
        if not writer.isOpened():
            pytest.skip("mp4v unavailable")
        rng = np.random.default_rng(101)
        for index in range(1800):
            group = index // 180
            color = np.array([80 + group * 8, 120 - group * 3, 160 - group * 6])
            image = np.clip(color + rng.normal(0, 8, (96, 128, 3)), 0, 255).astype(
                np.uint8
            )
            writer.write(image)
        writer.release()
    return camera, path


def frame_tables(tmp_path):
    _, path = dataset(tmp_path)
    table = read_labels(path, 5)
    rows, frames = [], []
    for row in table.to_dict("records"):
        group = row["recipe_id"] - 5
        rgb = np.array([80 + group * 9, 130 - group * 3, 160 - group * 6], float)
        row.update(dict(zip(["r", "g", "b"], rgb)))
        row.update(dict(zip(FEATURES, optical_features(rgb))))
        row.update(
            usable=True,
            ObservationState="VALID",
            quality_flags="",
            quality_details="",
            height_px=np.nan,
        )
        rows.append(row)
        for j in range(5):
            sample = dict(row)
            sample.update(dict(zip(FEATURES, optical_features(rgb + (j - 2) * 0.1))))
            sample["frame_index"] = row["end_frame"] - 45 + j * 8
            frames.append(sample)
    return pd.DataFrame(rows), pd.DataFrame(frames)


def test_features_use_fixed_reference_without_empty_subtraction():
    x = optical_features([[245, 245, 245], [24.5, 24.5, 24.5]])
    np.testing.assert_allclose(x[:, 3:], [[0, 0, 0], [1, 1, 1]], atol=1e-12)


@pytest.mark.parametrize("k", [1, 5, 12])
def test_knn_matches_direct_distance_vote_and_fraction_calculation(k):
    rng = np.random.default_rng(3)
    X, query = rng.normal(size=(9, 6)), rng.normal(size=(4, 6))
    y = np.arange(9) % 3 + 5
    volumes = np.eye(3)[np.arange(9) % 3]
    model = VectorizedOpticalKNN(k).fit(X, y, volumes)
    expected_d = np.linalg.norm(
        ((query - X.mean(0)) / X.std(0))[:, None] - model.X_[None], axis=2
    )
    np.testing.assert_allclose(model.distances(query), expected_d, atol=1e-7)
    near = np.argsort(expected_d, axis=1)[:, : min(k, 9)]
    weight = 1 / (np.take_along_axis(expected_d, near, axis=1) + 1e-6)
    weight /= weight.sum(axis=1, keepdims=True)
    result = model.predict(query)
    expected_vol = (weight[:, :, None] * volumes[near]).sum(axis=1)
    np.testing.assert_allclose(result["fractions"], expected_vol)
    np.testing.assert_array_equal(result["recipe_id"], expected_vol.argmax(axis=1) + 5)
    np.testing.assert_allclose(result["confidence"], expected_vol.max(axis=1))


def test_clipping_is_per_channel_and_preserves_raw_rgb():
    patch = np.full((10, 10, 3), 100, np.uint8)
    patch[:2, :, 0] = 2
    result = inspect_patch(patch)
    assert result["ObservationState"] == "SENSOR_CLIPPED"
    assert "R-Ch clipped" in result["quality_details"]
    assert result["r"] == 100
    assert np.isfinite([result[f] for f in FEATURES]).all()


@pytest.mark.parametrize("kind", ["saturation", "uneven", "duplicate", "timestamp"])
def test_quality_gates(kind):
    patch = np.full((10, 10, 3), 100, np.uint8)
    kwargs = {}
    if kind == "saturation":
        patch[:2, :, 2] = 255
    elif kind == "uneven":
        patch[:5] = 200
    elif kind == "duplicate":
        kwargs["previous"] = patch.copy()
    else:
        kwargs["timestamp_offset_s"] = 2.01
    result = inspect_patch(patch, **kwargs)
    assert not result["usable"]
    assert result["quality_details"]


def test_roi_excludes_meniscus_and_base():
    image = np.full((480, 640, 3), 180, np.uint8)
    image[310:397, 295:327] = 80
    patch, thumb, geometry = liquid_patch(image, SequenceConfig())
    _left, top, _right, bottom = geometry["roi_pixels"]
    assert 310 < top < bottom < 397
    assert thumb.shape == (32, 32, 3)
    assert (patch == 80).all()


def test_labels_impute_component_volume_and_keep_normalized_targets(tmp_path):
    camera, path = dataset(tmp_path)
    data = labels()
    data.loc[:2, "t400_ml"] = np.nan
    data.loc[:2, "total_ml"] = 30
    data.to_csv(path, index=False)
    loaded = read_labels(path, 5)
    assert loaded.loc[:2, "t400_ml"].eq(0).all()
    np.testing.assert_allclose(loaded[FRACTIONS].sum(axis=1), 1)
    assert portable_path(str(camera).replace("/", "\\")) == camera


def test_invalid_label_block_is_not_silently_repaired(tmp_path):
    _, path = dataset(tmp_path)
    data = labels()
    data.loc[2, "recipe_id"] = 6
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="blocks"):
        read_labels(path, 5)


def test_missing_video_preserves_all_rows_without_opening_capture(
    tmp_path, monkeypatch
):
    camera, _ = dataset(tmp_path)
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda _: pytest.fail("Must check existence first")
    )
    result = load_labeled_sequences(camera / "labels")
    assert len(result["measurements"]) == 30
    assert len(result["frames"]) == 150
    assert result["measurements"].ObservationState.eq("STALE_FRAME").all()
    assert result["measurements"].quality_flags.eq("MISSING_VIDEO").all()
    assert result["rois"] == {}
    assert result["audit"][0]["frame_count"] == 0


def test_vial_holdout_excludes_all_five_frames_and_reference(tmp_path):
    measured, frames = frame_tables(tmp_path)
    evaluated = evaluate_labeled(measured, frames)
    assert len(evaluated["results"]) == 30
    assert evaluated["results"].Causal_State.notna().all()
    assert len(evaluated["folds"][0]["train_frame_keys"]) == 100
    fold = evaluated["folds"][1]
    assert len(fold["train_frame_keys"]) == 95
    for f in evaluated["folds"]:
        assert not set(f["test_vial_keys"]) & set(f["train_vial_keys"])
        for reference in f["registry"].values():
            assert not set(f["test_vial_keys"]) & set(reference["training_vial_keys"])
    # A held-out vial perturbation cannot alter its model/scaler/centroid.
    changed = frames.copy()
    changed.loc[changed.vial.eq(2), FEATURES] += 1000
    rerun = evaluate_labeled(measured, changed)
    assert fold["model"] == rerun["folds"][1]["model"]
    assert fold["registry"] == rerun["folds"][1]["registry"]
    assert evaluated["metrics"].iloc[0].known_recipe_transitions == 9
    assert pd.isna(evaluated["metrics"].iloc[0].carryover_detection_recall)


def test_all_clipped_keeps_raw_values_and_draws_every_vial(tmp_path):
    import matplotlib.pyplot as plt

    measured, frames = frame_tables(tmp_path)
    for table in (measured, frames):
        table["usable"] = False
        table["ObservationState"] = "SENSOR_CLIPPED"
        table["quality_flags"] = "DARK_CHANNEL_CLIPPED"
        table["quality_details"] = "R-Ch clipped (med=2.1)"
    evaluated = evaluate_labeled(measured, frames)
    assert evaluated["model"] is None
    assert evaluated["results"].Causal_State.eq("SENSOR_REJECT").all()
    assert np.isfinite(evaluated["results"][["r", "g", "b"]]).all().all()
    assert all(
        v["quality_status"] == "CLIPPED_WARNING" for v in evaluated["registry"].values()
    )
    rois = {
        key: np.full((32, 32, 3), [2, 30, 70], np.uint8) for key in measured.vial_key
    }
    figure = render_run_diagnostics(evaluated["results"], rois, "All rejected fixture")
    np.testing.assert_array_equal(figure.axes[-1].get_xticks(), np.arange(1, 31))
    assert len(figure.axes[0].collections[-1].get_offsets()) == 30
    assert len(figure.axes[-1].artists) == 30
    figure.savefig(tmp_path / "all_rejected.png")
    plt.close(figure)
    assert render_run_diagnostics(pd.DataFrame(), {}, "missing") is None


def test_causal_projection_priority_and_unknown_cause():
    alpha, error = mixing_projection(
        np.array([0.3, 0.2, 0.1]), np.array([0.1, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])
    )
    assert alpha == pytest.approx(0.5, abs=1e-6)
    assert error < 1e-6
    row = dict(zip(FEATURES, [70, 20, 10, 0.3, 0.2, 0.1]))
    row.update(
        usable=True,
        recipe_id=6,
        vial_in_run=1,
        quality_details="",
        partition="changeover_test",
    )
    registry = {
        6: {
            "reference_available": True,
            "mu_Lab": [50, 20, 10],
            "mu_OD": [0.1, 0.2, 0.1],
            "target_fractions": [0.5, 0.5, 0],
        },
        5: {
            "reference_available": True,
            "mu_Lab": [80, 20, 10],
            "mu_OD": [0.5, 0.2, 0.1],
        },
    }
    assert (
        attribute(row, registry, previous_recipe=5)["Causal_State"]
        == "TRANSIENT_CARRYOVER"
    )
    assert attribute(row, registry)["Causal_State"] == "EVIDENCE_PENDING"
    assert (
        attribute({**row, "usable": False}, registry)["Causal_State"] == "SENSOR_REJECT"
    )
    assert attribute(row, {})["Causal_State"] == "EVIDENCE_PENDING"


def test_viterbi_can_smooth_neutral_reject_and_height_does_not_extrapolate(tmp_path):
    np.testing.assert_array_equal(
        smooth_recipes([[0.99, 0.01], [0.5, 0.5], [0.99, 0.01]], [5, 6]), [5, 5, 5]
    )
    measured, frames = frame_tables(tmp_path)
    evaluated = evaluate_labeled(measured, frames)
    height = {
        "status": "proxy_calibrated",
        "height_px": [20, 40, 60],
        "total_ml": [30, 50, 80],
    }
    events = measured.iloc[:2].copy()
    events["height_px"] = [35, 80]
    events["partition"] = "unlabeled"
    result = evaluate_unlabeled(events, evaluated, height)
    assert result.iloc[0].estimated_total_ml == 45
    assert pd.isna(result.iloc[1].estimated_total_ml)
    assert not result.fill_deviation_verified.any()
    assert fit_height_calibration(measured)["status"] == "unavailable"


def test_real_codec_workflow_exports_30_vials_and_150_frame_candidates(tmp_path):
    camera, _ = dataset(tmp_path, with_video=True)
    result = run_operator_workflow(tmp_path, dataset_root=camera, plots=True)
    assert len(result["df_labeled_results"]) == 30
    assert len(result["measurement_result"]["frames"]) == 150
    assert len(result["rois"]) == 30
    assert result["summary"]["training_instances"] > 0
    assert result["measurement_result"]["audit"][0]["frame_count"] == 1800
    assert result["measurement_result"]["audit"][0]["decoded_samples"] == 150
    assert (result["output_dir"] / "labeled_video5_diagnostics.png").is_file()
    assert len(list((result["output_dir"] / "raw_rois").glob("*.png"))) == 30
    folds = json.loads((result["output_dir"] / "training_folds.json").read_text())
    assert len(folds) == 30


class FakeCapture:
    def __init__(self, frames, fps=10):
        self.frames, self.fps, self.position = frames, fps, 0

    def isOpened(self):
        return True

    def grab(self):
        if self.position >= len(self.frames):
            return False
        self.position += 1
        return True

    def retrieve(self):
        return True, self.frames[self.position - 1]

    def get(self, field):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_POS_MSEC: (self.position - 1) / self.fps * 1000,
        }.get(field, 0)

    def release(self):
        pass


@pytest.mark.parametrize("truncated", [False, True])
def test_segmenter_detects_fill_and_retains_truncated_event(
    tmp_path, monkeypatch, truncated
):
    video = tmp_path / "blind.mp4"
    video.touch()
    rng = np.random.default_rng(5)
    frames = []
    for i in range(50 if truncated else 60):
        frame = np.full((480, 640, 3), 180, np.uint8)
        if 10 <= i < 50:
            frame[210:255, 265:375] = 20
            frame[270:397, 295:327] = rng.integers(78, 82, (127, 32, 3), dtype=np.uint8)
        frames.append(frame)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture(frames))
    result = UnlabeledVideoSegmenter().segment(video)
    assert len(result["events"]) == 1
    event = result["events"].iloc[0]
    assert event.start_frame == 10
    assert np.isfinite(event.settled_frame_idx)
    assert result["audit"]["background_initialized"]
    assert result["audit"]["decoded_samples"] == len(frames) // 2
    assert bool("EVENT_TRUNCATED" in event.quality_flags) == truncated
    assert len(result["rois"]) == 1


def test_missing_dataset_renders_no_blank_charts(tmp_path):
    result = run_operator_workflow(tmp_path)
    assert result["summary"]["status"] == "missing_data"
    assert not list(result["output_dir"].glob("*.png"))
    assert result["df_labeled_results"].empty


def test_blind_all_rejected_does_not_invent_consensus(tmp_path):
    measured, frames = frame_tables(tmp_path)
    evaluated = evaluate_labeled(measured, frames)
    events = measured.iloc[:3].copy()
    events["usable"] = False
    events["quality_flags"] = "DARK_CHANNEL_CLIPPED"
    events["quality_details"] = "R-Ch clipped"
    events["partition"] = "unlabeled"
    report = evaluate_unlabeled(events, evaluated, {"status": "unavailable"})
    assert len(report) == 3
    assert report.recipe_id.isna().all()
    assert report.Causal_State.eq("SENSOR_REJECT").all()


def test_blind_ood_query_cannot_be_an_optical_pass():
    row = dict(zip(FEATURES, [50, 10, 10, 0.2, 0.3, 0.4]))
    row.update(
        usable=True,
        recipe_id=5,
        vial_in_run=1,
        quality_details="",
        partition="unlabeled",
        nearest_distance=10,
    )
    registry = {
        5: {
            "reference_available": True,
            "mu_Lab": [50, 10, 10],
            "mu_OD": [0.2, 0.3, 0.4],
            "target_fractions": [0.5, 0.5, 0],
        }
    }
    result = attribute(row, registry, ood_threshold=2)
    assert result["delta_e"] == 0
    assert result["Causal_State"] == "SYSTEMIC_FAULT"
    assert result["root_cause_confirmed"] is False


def test_segmenter_requires_observed_empty_scene(tmp_path, monkeypatch):
    video = tmp_path / "occupied.mp4"
    video.touch()
    image = np.full((480, 640, 3), 100, np.uint8)
    image[210:255, 265:375] = 20
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture([image] * 40))
    result = UnlabeledVideoSegmenter().segment(video)
    assert result["events"].empty
    assert result["audit"]["initial_occupied_samples_skipped"] == 20
    assert not result["audit"]["background_initialized"]
