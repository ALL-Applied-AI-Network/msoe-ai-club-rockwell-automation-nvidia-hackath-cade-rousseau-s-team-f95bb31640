"""Camera and color measurement. Owners: Braedon, Kal."""

from datetime import datetime

import cv2
import numpy as np

from changeover_guard.contracts import TankReading, VialMeasurement

# Interior liquid regions for the fixed tank_level camera.  Coordinates are
# expressed for the supplied 640x480 frames and scaled for other resolutions.
# T300 is deliberately omitted because it is out of service and must not be
# used by the recipe model.
_TANK_ROIS = {
    "T100": (180, 325, 245, 395),
    "T200": (290, 325, 355, 395),
    "T400": (510, 320, 575, 385),
}
_REFERENCE_SIZE = (640, 480)


def _tank_crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a calibrated ROI, scaling coordinates from the reference size."""
    height, width = image.shape[:2]
    scale_x = width / _REFERENCE_SIZE[0]
    scale_y = height / _REFERENCE_SIZE[1]
    x1, y1, x2, y2 = roi
    x1, x2 = round(x1 * scale_x), round(x2 * scale_x)
    y1, y2 = round(y1 * scale_y), round(y2 * scale_y)
    return image[y1:y2, x1:x2]


def _robust_rgb(crop: np.ndarray) -> tuple[float, float, float]:
    """Estimate RGB using the median of non-shadow, non-glare pixels."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Do not require saturation: T100 contains clear diluent.  Value removes
    # the strongest reflections and dark tank hardware.
    valid = (hsv[:, :, 2] >= 20) & (hsv[:, :, 2] <= 250)
    pixels = crop[valid]
    if pixels.shape[0] < 20:
        raise ValueError("tank ROI contains too few valid pixels")

    # OpenCV stores BGR; TankReading stores RGB.  Median is less sensitive to
    # specular highlights than a plain mean.
    median_bgr = np.median(pixels, axis=0)
    return tuple(float(value) for value in median_bgr[::-1])


def tanks(image: np.ndarray, timestamp: datetime) -> TankReading:
    """Read the color of each tank (T100, T200, T400) from a BGR image.

    timestamp is the frame's capture time (the `timestamp` column from `dataset.frames`).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    colors = {
        tank: _robust_rgb(_tank_crop(image, roi))
        for tank, roi in _TANK_ROIS.items()
    }
    return TankReading(timestamp=timestamp, colors=colors)


def vial(image: np.ndarray, timestamp: datetime) -> VialMeasurement:
    """Measure the color and fill level of a vial from a BGR image.
    timestamp is the frame's capture time (the `timestamp` column from `dataset.frames`).
    """
    raise NotImplementedError
