import numpy as np
import pytest

from changeover_guard.features import DEFAULT_VIAL_ROI, extract_vial_features, scale_roi


def test_extract_vial_features_converts_bgr_to_rgb():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    x1, y1, x2, y2 = DEFAULT_VIAL_ROI
    image[y1:y2, x1:x2] = (200, 100, 50)
    features = extract_vial_features(image)
    assert features.rgb_median == (50.0, 100.0, 200.0)
    assert all(np.isfinite(features.lab_median))


def test_scale_roi_tracks_image_resolution():
    assert scale_roi((960, 1280, 3)) == tuple(value * 2 for value in DEFAULT_VIAL_ROI)


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        extract_vial_features(np.zeros((480, 640), dtype=np.uint8))
