"""Operator-reviewed color initialization for the fixed four-tank top camera.

This is a geometry-assisted liquid detector, not a general tank detector.
CIELAB uses D65 / 2 degrees, assuming the uploaded image is sRGB.
"""

import hashlib
import io
from datetime import UTC, datetime

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.color import rgb2lab

TANK_IDS = ("T_100", "T_200", "T_300", "T_400")
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def decode_image(payload: bytes) -> np.ndarray:
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Select a PNG or JPEG image smaller than 8 MiB.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ValueError("Only PNG and JPEG images are supported.")
            if image.width * image.height > 24_000_000:
                raise ValueError("Image must have at most 24 million pixels.")
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                raise ValueError("Use an opaque image without transparency.")
            return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The image could not be decoded. Select a valid PNG or JPEG.") from exc


def _validate_regions(regions, width, height):
    if set(regions) != set(TANK_IDS):
        raise ValueError("Provide exactly one liquid region for each of the four tanks.")
    for tank, box in regions.items():
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError(f"{tank}: region must have four finite coordinates.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"{tank}: region must lie inside the image.")
        if (x2 - x1) * width < 8 or (y2 - y1) * height < 8:
            raise ValueError(f"{tank}: liquid region is too small.")
    boxes = list(regions.values())
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                raise ValueError("Tank regions must not overlap.")


def detect_regions(image: np.ndarray) -> dict:
    """Locate a consistent liquid patch within four calibrated search windows.

    Other camera poses require manual region selection.
    IDs are assigned by physical left-to-right order, never inferred from hue.
    """
    height, width = image.shape[:2]
    if not 1.8 <= width / height <= 2.6:
        raise ValueError("Use the wide tank top view, or mark all four liquid regions manually.")
    thumbnail = image[::max(1, height // 128), ::max(1, width // 256)]
    if float(thumbnail.reshape(-1, 3).std(axis=0).max()) < 4:
        raise ValueError("Could not locate liquid in a featureless image. Mark regions on a tank image.")
    # Interior search windows from this camera geometry. Wider windows include
    # colored reflections on steel below T400 and fixtures above T100.
    windows = ((.022, .715, .124, .83), (.325, .67, .395, .80),
               (.60, .70, .70, .80), (.905, .645, .992, .75))
    result = {}
    for tank, (left, top, right, bottom) in zip(TANK_IDS, windows):
        x0, xend = round(left * width), round(right * width)
        y0, yend = round(top * height), round(bottom * height)
        crop = image[y0:yend, x0:xend]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        valid = (hsv[:, :, 2] > 8) & (hsv[:, :, 2] < 252)
        if valid.sum() < 40:
            raise ValueError(f"Could not locate liquid in {tank}. Mark liquid regions manually.")
        labs = rgb2lab(crop[:, :, ::-1] / 255.0)
        distances = np.linalg.norm(labs - np.median(labs[valid], axis=0), axis=2)
        # Local consistency supports pale/clear fluids as well as saturated dyes.
        mask = (valid & (distances <= max(8, float(np.percentile(distances[valid], 65))))).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            raise ValueError(f"Could not locate liquid in {tank}. Mark the liquid regions manually.")
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[component, cv2.CC_STAT_AREA]
        if area < max(64, crop.shape[0] * crop.shape[1] * .08):
            raise ValueError(f"{tank} is unclear. Mark a larger unobstructed liquid region.")
        # A component can curve around the tank rim; its bounding box can
        # include mostly background. Find an interior patch instead. Padding
        # treats clipped image/window boundaries as edges too.
        interior = np.pad((labels == component).astype(np.uint8), 1)
        distance = cv2.distanceTransform(interior, cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        radius = float(distance[y, x]) * .6  # square entirely inside the inscribed circle
        result[tank] = [(x0 + x - radius) / width, (y0 + y - radius) / height,
                        (x0 + x + radius) / width, (y0 + y + radius) / height]
    _validate_regions(result, width, height)
    return result


def analyze(payload: bytes, source_name: str, regions: dict | None = None) -> tuple[dict, np.ndarray]:
    image = decode_image(payload)
    height, width = image.shape[:2]
    method = "operator_regions" if regions is not None else "geometry_assisted"
    regions = regions if regions is not None else detect_regions(image)
    _validate_regions(regions, width, height)
    tanks = {}
    for tank in TANK_IDS:
        x1, y1, x2, y2 = regions[tank]
        crop = image[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
        rgb = crop[:, :, ::-1].reshape(-1, 3).astype(float)
        # Reject clipped glare and black fixtures, but preserve clear and dark dyes.
        valid = (rgb.max(axis=1) > 8) & (rgb.max(axis=1) < 252)
        pixels = rgb[valid]
        if len(pixels) < 40 or len(pixels) < .25 * len(rgb):
            raise ValueError(f"{tank}: too much glare or shadow. Select a different liquid patch.")
        labs = rgb2lab(pixels / 255.0)
        distances = np.linalg.norm(labs - np.median(labs, axis=0), axis=1)
        keep = distances <= np.percentile(distances, 80)
        color = np.median(pixels[keep], axis=0)
        lab = rgb2lab(color.reshape(1, 1, 3) / 255.0).reshape(3)
        spread = float(np.median(distances))
        tanks[tank] = {
            "rgb": color.tolist(), "lab": lab.tolist(), "region": list(regions[tank]),
            "sample_pixels": int(keep.sum()), "lab_spread": spread,
            "recipe_enabled": tank != "T_300",
            "warnings": ["Uneven color: check for walls, reflections or mixed lighting."] if spread > 10 else [],
        }
    return {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "source_name": source_name, "source_sha256": hashlib.sha256(payload).hexdigest(),
        "image_width": width, "image_height": height, "method": method,
        "color_space": "CIELAB", "illuminant": "D65", "observer": "2 degrees",
        "calibration": "Camera-relative estimate; input assumed sRGB, no color-chart calibration.",
        "confirmed": False, "tanks": tanks,
    }, image
