"""Vial-region color features shared by real and synthetic evaluation."""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2lab

from changeover_guard.contracts import RGB

REFERENCE_SIZE = (640, 480)
DEFAULT_VIAL_ROI = (300, 335, 324, 370)  # x1, y1, x2, y2; twin/calibration.json


@dataclass(frozen=True)
class VialFeatures:
    rgb_median: RGB
    lab_median: tuple[float, float, float]
    hsv_median: tuple[float, float, float]
    glare_fraction: float
    roi: tuple[int, int, int, int]


def scale_roi(
    shape: tuple[int, ...],
    roi: tuple[int, int, int, int] = DEFAULT_VIAL_ROI,
) -> tuple[int, int, int, int]:
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must contain positive height and width")
    height, width = shape[:2]
    scale_x, scale_y = width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]
    x1, y1, x2, y2 = roi
    scaled = (
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    )
    left, top, right, bottom = scaled
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("vial ROI is outside image bounds")
    return scaled


def extract_vial_features(
    image: np.ndarray,
    roi: tuple[int, int, int, int] = DEFAULT_VIAL_ROI,
    *,
    glare_percentile: float = 95.0,
) -> VialFeatures:
    """Measure robust color in the calibrated liquid ROI of a BGR frame."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")
    if not 50 <= glare_percentile <= 100:
        raise ValueError("glare_percentile must be between 50 and 100")
    scaled = scale_roi(image.shape, roi)
    x1, y1, x2, y2 = scaled
    crop = image[y1:y2, x1:x2]
    rgb_pixels = crop[:, :, ::-1].reshape(-1, 3).astype(float)
    brightness = rgb_pixels.max(axis=1)
    cutoff = float(np.percentile(brightness, glare_percentile))
    valid = brightness <= cutoff
    if valid.sum() < 20:
        raise ValueError("vial ROI contains too few valid pixels")
    selected = rgb_pixels[valid]
    rgb = np.median(selected, axis=0)
    lab = rgb2lab((rgb / 255.0).reshape(1, 1, 3)).reshape(3)
    hsv = cv2.cvtColor(np.uint8([[rgb[::-1].round()]]), cv2.COLOR_BGR2HSV).reshape(3)
    return VialFeatures(
        rgb_median=tuple(float(value) for value in rgb),
        lab_median=tuple(float(value) for value in lab),
        hsv_median=tuple(float(value) for value in hsv),
        glare_fraction=float(1 - valid.mean()),
        roi=scaled,
    )
