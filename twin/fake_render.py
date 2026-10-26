"""Stand-in renderer until Isaac Sim runs: tint the liquid into a real empty-vial frame. Writes synthetic/.

    uv run python twin/manifest.py --limit 60 && uv run python twin/fake_render.py

ponytail: flat 2D tint, no refraction or meniscus; the Isaac render (render.py) replaces it.
"""
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from writer import write_root

ROOT = Path(__file__).parents[1]
VIDEO_ID = 100
EMPTY_S = 391.0                              # video 5: vial 9 in place, nothing poured yet
X0, X1 = 290, 334                            # vial interior columns
VISIBLE = (270, 410)                         # liquid shows between the holder plate and the holder hole
BOTTOM_Y, PX_PER_ML = 422, 3.2               # from recipes 5 and 6; matches twin geometry (0.57 mL/mm)
FILL_STEPS = (0.33, 0.66, 1.0)               # in_fill frames per vial, as fractions of total mL


def video_frame(cam: str, t: float) -> np.ndarray:
    """BGR frame of video 5 from camera cam at t seconds."""
    cap = cv2.VideoCapture(str(ROOT / "data" / cam / "videos" / f"{cam}_video_5.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, im = cap.read()
    if not ok:
        raise RuntimeError(f"no {cam} frame at {t} s")
    return im


def render(empty: np.ndarray, box_mean: np.ndarray, rgb: list[float], ml: float, light: float) -> np.ndarray:
    """Tint ml of liquid into the empty-vial frame so the box shows rgb; light in [0, 1] scales brightness."""
    im = empty.astype(float)
    mask = np.zeros(im.shape[:2])
    mask[max(int(BOTTOM_Y - ml * PX_PER_ML), VISIBLE[0]):VISIBLE[1], X0:X1] = 1
    mask = cv2.GaussianBlur(mask, (7, 3), 0)[..., None]
    transmission = np.array(rgb[::-1]) / box_mean   # BGR; reproduces the measured color in the box
    im *= 1 - mask + mask * transmission
    gain = 0.85 + 0.3 * light
    im *= gain * np.random.default_rng(int(light * 1e3)).uniform(0.95, 1.05, 3)
    im += np.random.default_rng().normal(0, 2, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def main() -> None:
    with open(ROOT / "synthetic" / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    cal = json.loads((ROOT / "twin" / "calibration.json").read_text())
    empty = video_frame("vial_fill", EMPTY_S)
    tank = video_frame("tank_level", EMPTY_S)
    x0, y0, x1, y1 = cal["box"]
    box_mean = empty[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    frames, images, recipes = [], [], {}
    idle = {"in_fill": 0, "vial": None, "recipe_id": None}
    for r in rows:
        vial, rid = int(r["vial"]), int(r["recipe_id"])
        recipes[vial] = {t: float(r[f"{t.lower()}_ml"]) for t in ("T100", "T200", "T400") if r[f"{t.lower()}_ml"]}
        total = sum(recipes[vial].values())
        rgb = [float(r[c]) for c in "rgb"]
        for f in FILL_STEPS:
            frames.append({"in_fill": 1, "vial": vial, "recipe_id": rid})
            images.append(render(empty, box_mean, rgb, f * total, float(r["light"])))
        frames.append(idle)
        images.append(render(empty, box_mean, rgb, 0, random.random()))

    out = ROOT / "synthetic"
    for (vf, tl), im in zip(write_root(out, VIDEO_ID, frames, recipes), images):
        cv2.imwrite(str(vf), im)
        cv2.imwrite(str(tl), tank)
    cv2.imwrite(str(out / "tank_still.jpg"), tank)   # render.py reuses it: the twin has no tanks
    print(f"wrote {len(rows)} vials, {len(frames)} frames to synthetic/ (video_id {VIDEO_ID})")


if __name__ == "__main__":
    main()
