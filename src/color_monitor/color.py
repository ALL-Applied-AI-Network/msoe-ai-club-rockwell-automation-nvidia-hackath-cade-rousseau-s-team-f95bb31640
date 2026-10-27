import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from . import config as C


def bgr_to_lab(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return rgb2lab(rgb)


def crop(img, roi):
    x, y, w, h = roi
    H, W = img.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
        raise ValueError(f"ROI {roi} outside image {W}x{H}")
    return img[y:y + h, x:x + w]


def de00(lab_a, lab_b):
    a = np.asarray(lab_a, dtype=float).reshape(-1, 1, 3)
    b = np.asarray(lab_b, dtype=float).reshape(-1, 1, 3)
    return deltaE_ciede2000(a, b).reshape(-1)


def measure_frame(bgr):
    """Median CIELAB of valid liquid pixels plus carrier-presence info."""
    lab = bgr_to_lab(crop(bgr, C.LIQUID_ROI)).reshape(-1, 3)
    valid = (lab[:, 0] > C.MIN_L) & (lab[:, 0] < C.MAX_L)
    n_valid = int(valid.sum())
    if n_valid >= C.MIN_VALID_PIXELS:
        L, a, b = np.median(lab[valid], axis=0)
    else:
        L = a = b = np.nan
    holder = bgr_to_lab(crop(bgr, C.HOLDER_ROI))[..., 0]
    dark_frac = float((holder < C.HOLDER_DARK_L).mean())
    return {
        "L": L, "a": a, "b": b,
        "n_valid_px": n_valid,
        "holder_dark_frac": dark_frac,
        "carrier_present": dark_frac >= C.HOLDER_MIN_DARK_FRAC,
    }
