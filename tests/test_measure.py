"""Tests for tank color conversion, ROI calibration, and fill detection.

Generated top and front-view images exercise both color camera layouts.
Small synthetic images isolate exact color conversion and fill-line positions.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from changeover_guard.contracts import (
    COLOR_WARN_DE,
    TANK_FILL_WARN_PERCENT,
    color_difference,
    delta_e,
)
from changeover_guard.measure import (
    _FILL_ROIS,
    _REAL_TANK_ROIS,
    _TANK_ROIS,
    fill_line_row,
    fill_rois,
    tank_fill_levels,
    tank_rois,
    tanks,
)

# Keep image roots and manifests paired: both generated sets reuse filenames,
# so looking up a color target by filename alone would mix the two sources.
DATA_IMAGES = Path(__file__).parents[1] / "generated" / "images" / "top_view_tank_color"
COLOR_MANIFEST = DATA_IMAGES / "ai_tank_colors.json"
FRONT_IMAGES = Path(__file__).parents[1] / "generated" / "images" / "front_view_tank_color"
FRONT_MANIFEST = FRONT_IMAGES / "ai_tank_colors.json"
# The manifest supplies the requested colors and filenames without duplicating
# twenty sets of RGB values inside the tests.
MANIFEST_RECORDS = json.loads(COLOR_MANIFEST.read_text(encoding="utf-8"))["images"]
REQUESTED_TANK_COLORS = {
    record["filename"]: {
        tank: tuple(float(channel) for channel in rgb)
        for tank, rgb in record["requested_rgb"].items()
    }
    for record in MANIFEST_RECORDS
}
IMAGE_PATHS = [DATA_IMAGES / filename for filename in REQUESTED_TANK_COLORS]
FRONT_RECORDS = json.loads(FRONT_MANIFEST.read_text(encoding="utf-8"))["images"]
FRONT_IMAGE_PATHS = [FRONT_IMAGES / record["filename"] for record in FRONT_RECORDS]
EXPECTED_TANKS = {"T100", "T200", "T400"}

# UPDATED: fill fixtures have their own manifest because fill-level detection
# covers all four physical tanks, including out-of-service T300.
FILL_IMAGES = Path(__file__).parents[1] / "generated" / "images" / "tank_fill_level"
FILL_MANIFEST = FILL_IMAGES / "json" / "tank_fill_levels.json"
# Each fill image has a matching JSON record describing the intended line
# position, so tests can verify fixture completeness before reading pixels.
FILL_RECORDS = json.loads(FILL_MANIFEST.read_text(encoding="utf-8"))["images"]
EXPECTED_FILL_TANKS = {"T100", "T200", "T300", "T400"}


# UPDATED: use the committed manifest as the source of requested colors and
# fixture names instead of duplicating those values in the test.
def test_tank_color_manifest_is_complete_and_valid():
    # Confirm that every generated color fixture has exactly the three active
    # tank keys and valid 8-bit RGB channel values.
    assert len(MANIFEST_RECORDS) == 20
    assert len(REQUESTED_TANK_COLORS) == 20
    for colors in REQUESTED_TANK_COLORS.values():
        assert set(colors) == EXPECTED_TANKS
        for rgb in colors.values():
            assert len(rgb) == 3
            assert all(0.0 <= channel <= 255.0 for channel in rgb)


def test_tanks_converts_known_bgr_pixels_to_exact_rgb():
    """Validate the measurement algorithm without relying on image files."""
    # This synthetic frame isolates the BGR-to-RGB conversion from lighting,
    # reflections, and any imperfections in generated image assets.
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
    # Paint each calibrated ROI with one known BGR color, then verify that the
    # public TankReading values reverse the channel order exactly.
    rois = tank_rois(image.shape)
    for tank, bgr in bgr_colors.items():
        x1, y1, x2, y2 = rois[tank]
        image[y1:y2, x1:x2] = np.asarray(bgr, dtype=np.uint8)

    measured = tanks(image, datetime(2026, 9, 22)).colors

    assert measured == expected_rgb


def test_top_view_rois_scale_to_generated_frame():
    """The dashboard overlay and color reader share top-view ROI geometry."""
    # The top-view ROI coordinates can be tuned independently of the real/
    # front camera. Verify the selected calibration scales to this fixture.
    image = cv2.imread(str(IMAGE_PATHS[0]), cv2.IMREAD_COLOR)
    assert image is not None

    scale_x, scale_y = image.shape[1] / 4032, image.shape[0] / 1867
    assert tank_rois(image.shape) == {
        tank: (round(x1 * scale_x), round(y1 * scale_y), round(x2 * scale_x), round(y2 * scale_y))
        for tank, (x1, y1, x2, y2) in _TANK_ROIS.items()
    }


def test_real_camera_shape_keeps_real_frame_rois():
    """Real 640x480 frames must not receive the wide top-view calibration."""
    # The dashboard still offers real tank_level frames, so verify that their
    # 4:3 aspect ratio selects the currently stored real-camera calibration.
    assert tank_rois((480, 640, 3)) == _REAL_TANK_ROIS


def test_front_view_manifest_and_roi_calibration():
    """Front-view assets use their own targets and the 4:3 camera ROIs."""
    # Ensure the third dropdown source has a complete manifest and that its
    # larger images scale the same tank positions as the real camera.
    assert len(FRONT_RECORDS) == 20
    assert len({record["filename"] for record in FRONT_RECORDS}) == 20
    for record in FRONT_RECORDS:
        assert set(record["requested_rgb"]) == EXPECTED_TANKS
        assert (FRONT_IMAGES / record["filename"]).exists()
        for rgb in record["requested_rgb"].values():
            assert len(rgb) == 3
            assert all(0 <= channel <= 255 for channel in rgb)

    image = cv2.imread(str(FRONT_IMAGE_PATHS[0]), cv2.IMREAD_COLOR)
    assert image is not None
    scale_x, scale_y = image.shape[1] / 640, image.shape[0] / 480
    assert tank_rois(image.shape) == {
        tank: (round(x1 * scale_x), round(y1 * scale_y), round(x2 * scale_x), round(y2 * scale_y))
        for tank, (x1, y1, x2, y2) in _REAL_TANK_ROIS.items()
    }


def test_front_view_colors_are_read_from_the_expected_tanks():
    """Measure all front-view fixtures against their own manifest targets."""
    # Rendered translucent liquids can differ from requested RGB. Verify that
    # each ROI is still associated with its own tank instead of requiring a
    # fixed number of warnings, which would fail if the estimator improves.
    correctly_associated = 0
    for record in FRONT_RECORDS:
        image = cv2.imread(str(FRONT_IMAGES / record["filename"]), cv2.IMREAD_COLOR)
        assert image is not None
        measured = tanks(image, datetime(2026, 9, 22)).colors
        assert set(measured) == EXPECTED_TANKS
        for tank, color in measured.items():
            own_error, _ = color_difference(record["requested_rgb"][tank], color)
            other_errors = [
                delta_e(rgb, color)
                for other_tank, rgb in record["requested_rgb"].items()
                if other_tank != tank
            ]
            correctly_associated += own_error < min(other_errors)

    assert correctly_associated >= 0.9 * (len(FRONT_RECORDS) * len(EXPECTED_TANKS))


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_tanks_returns_rgb_colors_for_manifest_images(image_path: Path):
    # Read a real generated asset through OpenCV, just as the dashboard does.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {image_path}"

    # The timestamp is part of TankReading's interface even though color
    # extraction itself does not use it.
    timestamp = datetime(2026, 9, 22)
    reading = tanks(image, timestamp)

    assert reading.timestamp == timestamp
    assert set(reading.colors) == EXPECTED_TANKS

    # Every active tank must produce a finite three-channel RGB measurement.
    for color in reading.colors.values():
        assert len(color) == 3
        assert all(0.0 <= channel <= 255.0 for channel in color)

    # UPDATED: generated image prompts are requests, not pixel-level ground
    # truth.  Exact RGB conversion is covered by the controlled-pixel test;
    # generated fixtures only need to be readable and produce valid colors.


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda path: path.name)
def test_top_view_tanks_classify_requested_colors(image_path: Path):
    """Classify each measured color using the shared contract threshold."""
    # This is the end-to-end color check: the test loads an actual top-view
    # image, applies the calibrated ROIs, and compares each tank to its own
    # manifest target rather than comparing tanks to one another.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None

    measured = tanks(image, datetime(2026, 9, 22)).colors
    # These top-view fixtures were generated for the requested colors, so each
    # calibrated reading must stay below the project's shared warning limit.
    for tank, expected_rgb in REQUESTED_TANK_COLORS[image_path.name].items():
        difference, warning = color_difference(expected_rgb, measured[tank])
        assert difference < COLOR_WARN_DE
        assert not warning


def _synthetic_fill_image(levels: dict[str, float]) -> np.ndarray:
    """Build simple liquid-line fixtures for the calibrated scanline method."""
    # Synthetic fill images make the expected line position exact, which lets
    # the test evaluate the scanline math independently of AI-rendered images.
    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    rois = fill_rois(image.shape)
    for tank, level in levels.items():
        x1, y1, x2, y2 = rois[tank]
        line = round(y2 - (level / 100.0) * (y2 - y1 - 1))
        # Empty tank body is gray; liquid is a distinct BGR color below the
        # horizontal boundary so the detector has one dominant transition.
        image[y1:y2, x1:x2] = (185, 185, 185)
        image[line:y2, x1:x2] = (45, 110, 190)
    return image


def test_fill_manifest_is_complete_and_calibrated():
    """Every generated fill fixture records its expected measurement line."""
    # Check the metadata contract first: every file must cover all four tanks.
    # Actual level and measurement target are separate editable values.
    assert len(FILL_RECORDS) == 20
    for record in FILL_RECORDS:
        assert set(record) == {"filename", "tank_levels"}
        assert set(record["tank_levels"]) == EXPECTED_FILL_TANKS
        # Only the nested level record is needed here; tank-key coverage was
        # already verified by the set comparison above.
        for level in record["tank_levels"].values():
            assert set(level) == {"level_percent", "measure_fill_line_at_percent"}
            assert 0.0 <= level["level_percent"] <= 100.0
            assert 0.0 <= level["measure_fill_line_at_percent"] <= 100.0
            assert (FILL_IMAGES / record["filename"]).exists()


def test_tank_fill_levels_detects_calibrated_horizontal_lines():
    """Verify percent-full values from controlled fill-line positions."""
    # The detector reports percentage full measured upward from the tank
    # bottom; a three-percentage-point tolerance allows for smoothing spread.
    expected = {"T100": 20.0, "T200": 45.0, "T300": 70.0, "T400": 85.0}
    measured = tank_fill_levels(_synthetic_fill_image(expected))

    assert set(measured) == EXPECTED_FILL_TANKS
    for tank, level in expected.items():
        assert measured[tank] == pytest.approx(level, abs=3.0)


def test_fill_line_row_maps_reference_and_measured_levels_to_image_rows():
    """The UI line coordinates use the same bottom-up percentage convention."""
    # A 120-pixel ROI has predictable top, middle, and bottom line positions.
    roi = (10, 320, 90, 440)
    assert fill_line_row(roi, 100.0) == 320
    assert fill_line_row(roi, 50.0) == 380
    assert fill_line_row(roi, 0.0) == 439
    # Clamp out-of-range metadata rather than drawing beyond the tank body.
    assert fill_line_row(roi, -10.0) == 439
    assert fill_line_row(roi, 110.0) == 320


def test_fill_rois_exposes_fill_calibration_used_by_the_ui():
    """The UI ROI coordinates remain tied to measure._FILL_ROIS."""
    # At the calibration resolution no scaling should occur. Coordinates are
    # sorted only to support editing a box by either pair of opposite corners.
    expected = {
        tank: (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        for tank, (x1, y1, x2, y2) in _FILL_ROIS.items()
    }
    assert fill_rois((480, 640, 3)) == expected


@pytest.mark.parametrize("record", FILL_RECORDS, ids=lambda item: item["filename"])
def test_generated_fill_images_are_measurable(record: dict[str, object]):
    """Generated fixtures must read within the dashboard's fill tolerance."""
    # Unlike a bounds-only assertion, this comparison detects an ROI or
    # scanline regression that returns a plausible but incorrect percentage.
    image = cv2.imread(str(FILL_IMAGES / record["filename"]), cv2.IMREAD_COLOR)
    assert image is not None

    measured = tank_fill_levels(image)
    assert set(measured) == EXPECTED_FILL_TANKS
    for tank, measured_level in measured.items():
        target = record["tank_levels"][tank]["measure_fill_line_at_percent"]
        assert measured_level == pytest.approx(target, abs=TANK_FILL_WARN_PERCENT)
