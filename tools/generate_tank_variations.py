"""Create deterministic tank-color variants for the tank measurement tests."""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data" / "tank_level" / "images" / "tank_level_20260922_170508_537089.jpg"
OUTPUT_DIR = ROOT / "generated" / "images"
MANIFEST = ROOT / "generated" / "tank_colors.json"

# These ROIs match changeover_guard.measure._TANK_ROIS for the 640x480 camera.
TANK_ROIS = {
    "T100": (100, 345, 140, 395),
    "T200": (290, 345, 330, 395),
    "T400": (530, 345, 570, 395),
}

# Exact RGB values written into each tank ROI and recorded in tank_colors.json.
VARIATIONS = [
    {"T100": (220, 220, 220), "T200": (245, 165, 0), "T400": (0, 90, 245)},
    {"T100": (200, 230, 245), "T200": (245, 90, 0), "T400": (0, 90, 245)},
    {"T100": (220, 220, 220), "T200": (220, 60, 30), "T400": (20, 100, 220)},
    {"T100": (220, 220, 220), "T200": (240, 190, 20), "T400": (20, 80, 220)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (30, 180, 70)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (120, 40, 210)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (0, 180, 220)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (220, 30, 80)},
    {"T100": (180, 230, 220), "T200": (230, 80, 40), "T400": (20, 100, 220)},
    {"T100": (200, 240, 245), "T200": (230, 80, 40), "T400": (20, 100, 220)},
    {"T100": (230, 230, 190), "T200": (230, 80, 40), "T400": (20, 100, 220)},
    {"T100": (180, 220, 245), "T200": (230, 80, 40), "T400": (20, 100, 220)},
    {"T100": (220, 220, 220), "T200": (245, 120, 20), "T400": (20, 100, 220)},
    {"T100": (220, 220, 220), "T200": (245, 210, 30), "T400": (20, 100, 220)},
    {"T100": (220, 220, 220), "T200": (180, 50, 30), "T400": (20, 100, 220)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (30, 140, 240)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (20, 70, 180)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (40, 160, 230)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (80, 40, 180)},
    {"T100": (220, 220, 220), "T200": (230, 80, 40), "T400": (40, 190, 90)},
]


def main() -> None:
    source = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"could not read source image: {SOURCE}")
    if source.shape[:2] != (480, 640):
        raise ValueError(f"expected a 640x480 source image, got {source.shape[:2][::-1]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for index, colors in enumerate(VARIATIONS, start=1):
        image = source.copy()
        for tank, (x1, y1, x2, y2) in TANK_ROIS.items():
            # cv2 uses BGR, while the manifest intentionally records RGB.
            image[y1:y2, x1:x2] = np.asarray(colors[tank][::-1], dtype=np.uint8)

        filename = f"tank_variant_{index:02d}.png"
        output_path = OUTPUT_DIR / filename
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"could not write {output_path}")
        records.append({"filename": filename, "rgb": colors})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(ROOT)),
                "roi_coordinates": TANK_ROIS,
                "images": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(records)} images in {OUTPUT_DIR}")
    print(f"RGB manifest written to {MANIFEST}")


if __name__ == "__main__":
    main()
