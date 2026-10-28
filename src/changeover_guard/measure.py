"""Camera measurements used by the color, fill, and recipe pipeline.

Images arrive from OpenCV in BGR order, while the rest of the project stores
colors as RGB tuples.  This module keeps that conversion at the measurement
boundary and exposes calibrated ROI helpers so the dashboard and tests draw
the same regions that the measurement functions actually inspect.
"""

from datetime import datetime

import cv2
import numpy as np

from changeover_guard.contracts import TankReading, VialMeasurement

# UPDATED: Narrow liquid strips are calibrated to the supplied top-view frame.
# These ROIs intentionally cover the lower liquid area instead of the complete
# tank.  That avoids empty headspace, curved tank walls, labels, and fixtures
# that would otherwise distort the median color measurement.
# They avoid the tank walls, headspace above the liquid, and the hardware in
# front of each tank. Coordinates are expressed for the 4032x1867 source frame
# and scaled for generated top-view images such as 1842x854.
# T300 is deliberately omitted because it is out of service and must not be
# used by the recipe model.

_TANK_ROIS = {
    "T100": (88, 1335, 500, 1550),
    "T200": (1313, 1250, 1590, 1500),
    "T400": (3650, 1200, 4000, 1400),
}

_REFERENCE_SIZE = (4032, 1867)
# Keep the original real-camera calibration available. The generated front-
# view fixtures share this 4:3 layout and scale from the same 640x480 ROIs;
# wide top-view images require their separate calibration above.

_REAL_TANK_ROIS = {
    "T100": (185, 345, 203, 385),
    "T200": (295, 345, 315, 385),
    "T400": (515, 345, 535, 385),
}
_REAL_REFERENCE_SIZE = (640, 480)
_FILL_REFERENCE_SIZE = (640, 480)

# UPDATED: Fill-level measurement uses a separate calibrated interior ROI for
# every physical tank.  The liquid line is searched vertically inside these
# boxes instead of using the color ROIs above, which are intentionally narrow.
# T300 is included here because a fill-level camera can still detect an
# out-of-service tank as an equipment condition even though recipes cannot use
# it.
_FILL_ROIS = {
    "T100": (180, 322, 246, 395),
    "T200": (288, 322, 360, 394),
    "T300": (398, 320, 467, 394),
    "T400": (510, 318, 578, 390),
}
_FILL_TANKS = tuple(_FILL_ROIS)
# At 640x480 the weakest liquid line in the generated fill fixtures peaks near
# 5.3, while the strongest non-line edge in a typical tank body is about 3.6.
# ponytail: tuned on generated fixtures only; recheck on real tank_level frames.
_MIN_LINE_STRENGTH = 5.0


def _is_top_view(shape: tuple[int, ...]) -> bool:
    """Select calibration by aspect ratio.

    Top-view fixtures are about 2.16:1 while the real tank_level and generated
    front-view frames are 4:3.
    """
    height, width = shape[:2]
    aspect_ratio = width / height
    top_view_aspect = _REFERENCE_SIZE[0] / _REFERENCE_SIZE[1]
    real_view_aspect = _REAL_REFERENCE_SIZE[0] / _REAL_REFERENCE_SIZE[1]
    return abs(aspect_ratio - top_view_aspect) <= abs(aspect_ratio - real_view_aspect)


def tank_rois(shape: tuple[int, ...]) -> dict[str, tuple[int, int, int, int]]:
    """Return the calibrated tank ROIs scaled to an image shape."""
    # The reader and dashboard can receive different resolutions, so reject
    # malformed shapes before calculating scale factors or slicing an image.
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must contain positive height and width")

    height, width = shape[:2]
    if _is_top_view(shape):
        roi_definitions, reference_size = _TANK_ROIS, _REFERENCE_SIZE
    else:
        roi_definitions, reference_size = _REAL_TANK_ROIS, _REAL_REFERENCE_SIZE

    # Calibration coordinates are stored for the original top-view frame;
    # scaling them here keeps the same physical tank regions on images with a
    # different resolution while retaining the selected camera calibration.
    scale_x = width / reference_size[0]
    scale_y = height / reference_size[1]
    return {
        tank: (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        for tank, (x1, y1, x2, y2) in roi_definitions.items()
    }


def fill_rois(shape: tuple[int, ...]) -> dict[str, tuple[int, int, int, int]]:
    """Return the calibrated tank-body ROIs used for fill-line detection."""
    # Fill-level fixtures use a different 640x480 calibration than the
    # top-view color fixtures, so this function has its own reference size.
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must contain positive height and width")

    height, width = shape[:2]
    # Convert the fixed fill-body boxes into coordinates for the current frame.
    scale_x = width / _FILL_REFERENCE_SIZE[0]
    scale_y = height / _FILL_REFERENCE_SIZE[1]
    scaled = {}
    for tank, (x1, y1, x2, y2) in _FILL_ROIS.items():
        # Normalize opposite corners after scaling. This keeps live calibration
        # edits usable even when an operator enters the two corners in reverse.
        left, right = sorted((round(x1 * scale_x), round(x2 * scale_x)))
        top, bottom = sorted((round(y1 * scale_y), round(y2 * scale_y)))
        scaled[tank] = (left, top, right, bottom)
    return scaled


def fill_line_row(roi: tuple[int, int, int, int], level_percent: float) -> int:
    """Convert a tank fill percentage into the corresponding image row.

    The detector reports the liquid height upward from the bottom of the
    calibrated tank ROI. The dashboard uses this same conversion to draw the
    expected and measured horizontal lines over the source image.
    """
    # Clamp display values so malformed metadata cannot produce a line outside
    # the tank body or an invalid coordinate for OpenCV.
    bounded_percent = max(0.0, min(100.0, float(level_percent)))
    _, y1, _, y2 = roi
    return round(y2 - 1 - (bounded_percent / 100.0) * (y2 - y1 - 1))


def liquid_rois(image: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    """Return the color ROIs with headspace above the liquid line removed.

    tanks() measures these boxes, and the dashboard draws them, so the overlay
    always shows the pixels behind each color reading.
    """
    rois = tank_rois(image.shape)
    if _is_top_view(image.shape):
        # Fill ROIs are calibrated for the 4:3 camera only.
        return rois

    levels = tank_fill_levels(image)
    fill = fill_rois(image.shape)
    # Skip a few rows below the line for the meniscus, but always keep a few
    # rows at the bottom of the box so a low tank still has pixels to read.
    scale = image.shape[0] / _FILL_REFERENCE_SIZE[1]
    margin, min_rows = round(3 * scale), max(1, round(4 * scale))
    clipped = {}
    for tank, (x1, y1, x2, y2) in rois.items():
        if levels[tank] is not None:
            # ponytail: with no visible line the box is either all liquid or
            # all air; a nearly empty tank still reads its bottom rows as air.
            line = fill_line_row(fill[tank], levels[tank])
            y1 = min(max(y1, line + margin), y2 - min_rows)
        clipped[tank] = (x1, y1, x2, y2)
    return clipped


def _tank_crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Extract an already-scaled calibrated ROI."""
    # ROIs use the conventional OpenCV order (left, top, right, bottom).
    # Keeping cropping in one helper prevents color and fill readers from
    # duplicating slice-order logic.
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def _robust_rgb(crop: np.ndarray) -> tuple[float, float, float]:
    """Estimate RGB using the median of non-shadow, non-glare pixels."""
    # HSV makes it possible to filter extreme brightness values without
    # discarding low-saturation T100 diluent, which is intentionally clear.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Do not require saturation: T100 contains clear diluent.  Value removes
    # the strongest reflections and dark tank hardware.
    valid = (hsv[:, :, 2] >= 20) & (hsv[:, :, 2] <= 250)
    pixels = crop[valid]
    if pixels.shape[0] < 20:
        raise ValueError("tank ROI contains too few valid pixels")

    # OpenCV stores BGR; TankReading stores RGB.  Median is less sensitive to
    # specular highlights and small reflections than a plain mean.
    median_bgr = np.median(pixels, axis=0)
    return tuple(float(value) for value in median_bgr[::-1])


def tanks(image: np.ndarray, timestamp: datetime) -> TankReading:
    """Read the color of each tank (T100, T200, T400) from a BGR image.

    timestamp is the frame's capture time (the `timestamp` column from `dataset.frames`).
    """
    # Fail early for grayscale, empty, or otherwise malformed camera frames;
    # downstream OpenCV operations require a three-channel BGR image.
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    # Only the active recipe tanks are returned.  T300 remains physically
    # visible but is out of service and must not enter recipe color reasoning.
    rois = liquid_rois(image)
    colors = {tank: _robust_rgb(_tank_crop(image, rois[tank])) for tank in rois}
    return TankReading(timestamp=timestamp, colors=colors)


def _fill_line_percent(crop: np.ndarray) -> float | None:
    """Find a tank's horizontal liquid line and return percent full.

    UPDATED: This is the suggested scanline method.  It samples both side
    strips of the tank body to avoid the front support post, smooths each row,
    and chooses the strongest vertical color/brightness transition.  The
    position is normalized from the calibrated tank bottom, so a line at the
    bottom is 0% and a line at the top is 100%.
    """
    # The crop is already restricted to one tank body; the remaining work is
    # to locate the strongest horizontal transition inside that body.
    height, width = crop.shape[:2]
    if height < 20 or width < 20:
        raise ValueError("fill ROI is too small for line detection")

    # Exclude curved tank walls and the center support post.  The remaining
    # side strips contain the same liquid-line transition from two views while
    # reducing false edges caused by the foreground hardware.
    side_margin = max(2, round(width * 0.14))
    center_margin = max(2, round(width * 0.14))
    center = width // 2
    left = crop[:, side_margin : max(side_margin + 1, center - center_margin)]
    right = crop[:, min(width - side_margin - 1, center + center_margin) : width - side_margin]
    sample = np.concatenate((left, right), axis=1)
    if sample.shape[1] < 4:
        raise ValueError("fill ROI has too few measurement columns")

    # A row median suppresses small reflections and equipment texture while
    # retaining the broad horizontal transition made by the liquid surface.
    row_color = np.median(sample.astype(np.float32), axis=1)
    # Shape (rows, 1, 3) so each channel is smoothed down the rows separately.
    row_color = cv2.GaussianBlur(row_color.reshape(-1, 1, 3), (1, 7), 0).reshape(-1, 3)
    transition = np.linalg.norm(np.diff(row_color, axis=0), axis=1)

    # Ignore the ROI margins where tank rims and the lower fixture create
    # stronger edges than the liquid surface.  The selected offset is the
    # boundary between the upper empty region and lower liquid region.
    first = max(2, round(height * 0.12))
    last = min(height - 2, round(height * 0.88))
    if first >= last:
        raise ValueError("fill ROI has no valid line-search range")
    window = transition[first:last]
    if window.max() < _MIN_LINE_STRENGTH:
        # No line inside the search window: the tank is nearly empty, nearly
        # full, or blocked. Report unknown rather than the window edge.
        return None
    line_offset = first + int(np.argmax(window))
    # Normalize upward from the bottom: a line at the bottom is 0% full and a
    # line at the top is 100% full, matching the JSON fixture convention.
    return round(100.0 * (height - 1 - line_offset) / (height - 1), 1)


def tank_fill_levels(image: np.ndarray) -> dict[str, float | None]:
    """Measure the percent-full level of each tank from a BGR image.

    The result is keyed by tank number and reports the liquid line as a
    percentage of the calibrated tank body height, measured upward from the
    bottom.  This keeps fill measurement independent from the recipe/color
    measurement API and includes T300 for equipment monitoring.  A tank is
    None when no liquid line is visible (nearly empty or nearly full).
    """
    # Use the same input validation as the color reader so callers receive a
    # clear error instead of a confusing OpenCV failure.
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    # Measure all four physical tanks for equipment monitoring, including
    # T300, even though T300 is excluded from recipe/color measurements.
    rois = fill_rois(image.shape)
    return {
        tank: _fill_line_percent(_tank_crop(image, rois[tank]))
        for tank in _FILL_TANKS
    }


def vial(image: np.ndarray, timestamp: datetime) -> VialMeasurement:
    """Measure the color and fill level of a vial from a BGR image.
    timestamp is the frame's capture time (the `timestamp` column from `dataset.frames`).
    """
    # Vial measurement remains a separate contract stub; the changes in this
    # file concern tank color and tank fill-level camera measurements.
    raise NotImplementedError
