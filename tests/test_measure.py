import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from changeover_guard.contracts import COLOR_WARN_DE, delta_e
from changeover_guard.measure import tank_rois, tanks

DATA_IMAGES = Path(__file__).parents[1] / "generated" / "images" / "tank_color_variation"
COLOR_MANIFEST = DATA_IMAGES / "ai_tank_colors.json"
MANIFEST_RECORDS = json.loads(COLOR_MANIFEST.read_text(encoding="utf-8"))["images"]
REQUESTED_TANK_COLORS = {
    record["filename"]: {
        tank: tuple(float(channel) for channel in rgb)
        for tank, rgb in record["requested_rgb"].items()
    }
    for record in MANIFEST_RECORDS
}
IMAGE_PATHS = [DATA_IMAGES / filename for filename in REQUESTED_TANK_COLORS]
EXPECTED_TANKS = {"T100", "T200", "T400"}


# UPDATED: use the committed manifest as the source of requested colors and
# fixture names instead of duplicating those values in the test.
def test_tank_color_manifest_is_complete_and_valid():
    assert len(MANIFEST_RECORDS) == 20
    assert len(REQUESTED_TANK_COLORS) == 20
    for colors in REQUESTED_TANK_COLORS.values():
        assert set(colors) == EXPECTED_TANKS
        for rgb in colors.values():
            assert len(rgb) == 3
            assert all(0.0 <= channel <= 255.0 for channel in rgb)


def test_tanks_converts_known_bgr_pixels_to_exact_rgb():
    """Validate the measurement algorithm without relying on image files."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # UPDATED: write explicit OpenCV BGR pixels and expect their RGB reversal.
    bgr_colors = {
        "T100": (220, 180, 120),
        "T200": (40, 80, 210),
        "T400": (200, 110, 30),
    }
    expected_rgb = {
        "T100": (120.0, 180.0, 220.0),
        "T200": (210.0, 80.0, 40.0),
        "T400": (30.0, 110.0, 200.0),
    }
    rois = tank_rois(image.shape)
    for tank, bgr in bgr_colors.items():
        x1, y1, x2, y2 = rois[tank]
        image[y1:y2, x1:x2] = np.asarray(bgr, dtype=np.uint8)

    measured = tanks(image, datetime(2026, 9, 22)).colors

    assert measured == expected_rgb


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_tanks_returns_rgb_colors_for_manifest_images(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"

    timestamp = datetime(2026, 9, 22)
    reading = tanks(image, timestamp)

    assert reading.timestamp == timestamp
    assert set(reading.colors) == EXPECTED_TANKS

    for color in reading.colors.values():
        assert len(color) == 3
        assert all(0.0 <= channel <= 255.0 for channel in color)

    # UPDATED: generated fixtures must meet the same CIEDE2000 warning
    # threshold used by the dashboard; inaccurate readings should fail tests.
    for tank, expected_rgb in REQUESTED_TANK_COLORS[image_path.name].items():
        error = delta_e(expected_rgb, reading.colors[tank])
        assert error < COLOR_WARN_DE, (
            f"{image_path.name} {tank}: color error {error:.2f} "
            f"meets/exceeds COLOR_WARN_DE={COLOR_WARN_DE}"
        )
