"""Controlled pixels and controls verify software; these are not camera evidence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from changeover_guard.vial_color import (
    VialCalibration,
    encoded_rgb,
    fit_camera_chart,
    linear_rgb,
    measure_vial_color,
    repeatability_report,
)


@pytest.fixture
def calibrated_camera():
    rng = np.random.default_rng(42)
    observed = rng.uniform(25, 220, (20, 3))
    transform = np.diag([0.9, 1.05, 1.1])
    reference = encoded_rgb(linear_rgb(observed) @ transform)
    chart = pd.DataFrame(
        {"patch_id": range(20), "split": ["train"] * 10 + ["validation"] * 10}
    )
    for index, channel in enumerate("rgb"):
        chart[f"observed_{channel}"] = observed[:, index]
        chart[f"reference_{channel}"] = reference[:, index]
    calibration = fit_camera_chart(
        chart, VialCalibration(settings_locked=True, alignment_reviewed=True)
    )
    assert np.asarray(calibration.matrix) == pytest.approx(transform)
    return calibration


def stable_controls(calibration):
    rows = []
    for sample in range(5):
        for session in range(2):
            for capture in range(5):
                rows.append(
                    {
                        "sample_id": str(sample),
                        "session_id": str(session),
                        "capture_id": f"{sample}/{session}/{capture}",
                        "camera_id": calibration.camera_id,
                        "calibration_id": calibration.calibration_id,
                        "r": 40 + sample * 30,
                        "g": 70 + sample * 15,
                        "b": 120,
                    }
                )
    return pd.DataFrame(rows)


def test_linear_srgb_round_trip_and_not_gamma_averaging():
    rgb = np.array([[0, 20, 255], [120, 75, 200]])
    assert encoded_rgb(linear_rgb(rgb)) == pytest.approx(rgb)
    assert encoded_rgb(linear_rgb([128, 128, 128])) == pytest.approx([128, 128, 128])
    assert linear_rgb([128, 128, 128])[0] != pytest.approx(128 / 255)


def test_measurement_preserves_bgr_boundary_and_rejects_glare():
    calibration = VialCalibration()
    image = np.full((480, 640, 3), (40, 80, 150), dtype=np.uint8)
    reading = measure_vial_color(image, calibration)
    assert reading["rgb"] == pytest.approx([150, 80, 40])
    assert reading["usable"]
    image[335:370, 300:324] = 255
    assert measure_vial_color(image, calibration)["usable"] is False


def test_clear_liquid_is_not_rejected_for_low_saturation():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    assert measure_vial_color(image, VialCalibration())["usable"]


def test_reference_normalization_is_linear_and_required_each_frame():
    calibration = VialCalibration(
        reference_roi=(20, 20, 40, 40), reference_rgb=(180, 180, 180)
    )
    image = np.full((480, 640, 3), 100, dtype=np.uint8)
    image[20:40, 20:40] = 180
    baseline = measure_vial_color(image, calibration)["rgb"]
    darkened = np.round(encoded_rgb(linear_rgb(image) * 0.7)).astype(np.uint8)
    assert measure_vial_color(darkened, calibration)["rgb"] == pytest.approx(
        baseline, abs=1
    )
    darkened[20:40, 20:40] = 255
    assert not measure_vial_color(darkened, calibration)["usable"]


def test_camera_geometry_and_timestamp_area_are_checked():
    with pytest.raises(ValueError, match="aspect ratio"):
        measure_vial_color(np.zeros((100, 600, 3), dtype=np.uint8), VialCalibration())
    with pytest.raises(ValueError, match="timestamp"):
        VialCalibration(liquid_roi=(300, 440, 324, 470))


def test_calibration_fingerprint_and_repeatability_gate(calibrated_camera):
    calibration = calibrated_camera
    assert (
        VialCalibration.from_dict(calibration.to_dict()).calibration_id
        == calibration.calibration_id
    )
    payload = calibration.to_dict()
    payload["settings_locked"] = False
    with pytest.raises(ValueError, match="calibration_id"):
        VialCalibration.from_dict(payload)
    assert not repeatability_report(None, calibration)["proven"]
    samples = stable_controls(calibration)
    report = repeatability_report(samples, calibration)
    assert report["proven"] and report["worst_pair_delta_e"] == 0
    samples.loc[0, "r"] = 230
    assert not repeatability_report(samples, calibration)["proven"]


def test_adjacent_frames_cannot_replace_independent_controls(calibrated_camera):
    samples = stable_controls(calibrated_camera)
    samples["session_id"] = "one_session"
    assert not repeatability_report(samples, calibrated_camera)["proven"]
    samples.loc[0, "camera_id"] = "tank_top"
    with pytest.raises(ValueError, match="different camera"):
        repeatability_report(samples, calibrated_camera)


def test_identity_transform_and_review_flags_cannot_prove_repeatability():
    calibration = VialCalibration(settings_locked=True, alignment_reviewed=True)
    assert not repeatability_report(stable_controls(calibration), calibration)["proven"]


def test_bad_chart_and_unlocked_settings_are_not_validated(calibrated_camera):
    bad = replace(calibrated_camera, validation_delta_e=(4.0,) * 6)
    assert not repeatability_report(stable_controls(bad), bad)["proven"]
    unlocked = replace(calibrated_camera, settings_locked=False)
    assert not repeatability_report(stable_controls(unlocked), unlocked)["proven"]


def test_training_and_validation_chart_patches_must_be_disjoint():
    rows = pd.DataFrame(
        {"patch_id": ["a"] * 12, "split": ["train"] * 6 + ["validation"] * 6}
    )
    for prefix in ("observed", "reference"):
        for channel in "rgb":
            rows[f"{prefix}_{channel}"] = 100
    with pytest.raises(ValueError, match="disjoint"):
        fit_camera_chart(rows, VialCalibration())
