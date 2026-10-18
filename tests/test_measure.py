from datetime import datetime
from pathlib import Path

import cv2
import pytest

from changeover_guard.measure import tanks

DATA_IMAGES = Path(__file__).parents[1] / "generated" / "images" / "tank_color_variation"
IMAGE_PATHS = sorted(DATA_IMAGES.glob("ai_tank_variant_*.png"))[:20]
EXPECTED_TANKS = {"T100", "T200", "T400"}
COLOR_TOLERANCE = 5.0

# Expected RGB values measured from the AI-generated images with tanks().
EXPECTED_TANK_COLORS: dict[str, dict[str, tuple[float, float, float]]] = {
    "ai_tank_variant_01.png": {"T100": (139.0, 193.0, 198.0), "T200": (201.0, 129.0, 62.0), "T400": (13.0, 70.0, 193.0)},
    "ai_tank_variant_02.png": {"T100": (196.0, 195.0, 132.0), "T200": (206.0, 57.0, 26.0), "T400": (1.0, 47.0, 182.0)},
    "ai_tank_variant_03.png": {"T100": (165.0, 196.0, 179.0), "T200": (210.0, 113.0, 20.0), "T400": (1.0, 77.0, 207.0)},
    "ai_tank_variant_04.png": {"T100": (196.0, 194.0, 129.0), "T200": (194.0, 118.0, 19.0), "T400": (2.0, 58.0, 194.0)},
    "ai_tank_variant_05.png": {"T100": (144.0, 194.0, 213.0), "T200": (211.0, 70.0, 26.0), "T400": (15.0, 39.0, 112.0)},
    "ai_tank_variant_06.png": {"T100": (175.0, 192.0, 190.0), "T200": (193.0, 142.0, 12.0), "T400": (2.0, 141.0, 168.0)},
    "ai_tank_variant_07.png": {"T100": (173.0, 169.0, 222.0), "T200": (205.0, 127.0, 59.0), "T400": (7.0, 66.0, 200.0)},
    "ai_tank_variant_08.png": {"T100": (142.0, 202.0, 203.0), "T200": (204.0, 108.0, 39.0), "T400": (3.0, 139.0, 145.0)},
    "ai_tank_variant_09.png": {"T100": (176.0, 185.0, 184.0), "T200": (200.0, 14.0, 48.0), "T400": (6.0, 64.0, 198.0)},
    "ai_tank_variant_10.png": {"T100": (137.0, 200.0, 157.0), "T200": (210.0, 129.0, 59.0), "T400": (90.0, 16.0, 189.0)},
    "ai_tank_variant_11.png": {"T100": (130.0, 195.0, 209.0), "T200": (206.0, 128.0, 59.0), "T400": (197.0, 8.0, 152.0)},
    "ai_tank_variant_12.png": {"T100": (196.0, 197.0, 131.0), "T200": (205.0, 127.0, 58.0), "T400": (3.0, 111.0, 75.0)},
    "ai_tank_variant_13.png": {"T100": (159.0, 200.0, 226.0), "T200": (195.0, 119.0, 41.0), "T400": (13.0, 138.0, 228.0)},
    "ai_tank_variant_14.png": {"T100": (145.0, 203.0, 211.0), "T200": (201.0, 101.0, 28.0), "T400": (1.0, 55.0, 203.0)},
    "ai_tank_variant_15.png": {"T100": (221.0, 218.0, 183.0), "T200": (218.0, 160.0, 31.0), "T400": (76.0, 51.0, 228.0)},
    "ai_tank_variant_16.png": {"T100": (151.0, 198.0, 224.0), "T200": (223.0, 115.0, 89.0), "T400": (3.0, 159.0, 217.0)},
    "ai_tank_variant_17.png": {"T100": (164.0, 211.0, 195.0), "T200": (206.0, 127.0, 56.0), "T400": (3.0, 126.0, 152.0)},
    "ai_tank_variant_18.png": {"T100": (172.0, 183.0, 187.0), "T200": (205.0, 118.0, 40.0), "T400": (31.0, 30.0, 104.0)},
    "ai_tank_variant_19.png": {"T100": (203.0, 203.0, 143.0), "T200": (222.0, 74.0, 43.0), "T400": (3.0, 118.0, 126.0)},
    "ai_tank_variant_20.png": {"T100": (139.0, 197.0, 169.0), "T200": (222.0, 63.0, 25.0), "T400": (4.0, 69.0, 216.0)},
}


pytestmark = pytest.mark.skipif(
    len(IMAGE_PATHS) < 20,
    reason="20 AI tank images are required in generated/images/tank_color_variation",
)


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_tanks_returns_rgb_colors_for_twenty_images(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"

    timestamp = datetime(2026, 9, 22)
    reading = tanks(image, timestamp)

    assert reading.timestamp == timestamp
    assert set(reading.colors) == EXPECTED_TANKS

    for color in reading.colors.values():
        assert len(color) == 3
        assert all(0.0 <= channel <= 255.0 for channel in color)


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_tanks_matches_expected_colors_when_configured(image_path: Path):
    expected = EXPECTED_TANK_COLORS.get(image_path.name)
    if expected is None:
        pytest.skip("add this image's expected RGB values to EXPECTED_TANK_COLORS")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"
    measured = tanks(image, datetime(2026, 9, 22)).colors

    assert set(expected) == EXPECTED_TANKS
    for tank, expected_rgb in expected.items():
        assert measured[tank] == pytest.approx(expected_rgb, abs=COLOR_TOLERANCE)
