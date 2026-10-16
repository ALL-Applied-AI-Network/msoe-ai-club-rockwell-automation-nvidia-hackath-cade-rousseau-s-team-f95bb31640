"""Fit liquid colors from the clean (third) vial of each recipe in video 5. Writes twin/calibration.json.

    uv run python twin/calibrate.py

Frames come from the vial_fill video, not the stills: the stills' labels run 12-22 s ahead of the pour.
Color model (Beer-Lambert, T100 clear): log(rgb/255) = log_i0 - c200*k200 - c400*k400, c = tank mL / total mL.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from changeover_guard.contracts import COLOR_WARN_DE, delta_e
from changeover_guard.dataset import vial_labels

ROOT = Path(__file__).parents[1]
VIDEO = ROOT / "data" / "vial_fill" / "videos" / "vial_fill_video_5.mp4"
BOX = (300, 335, 324, 370)                   # x0, y0, x1, y1 inside the liquid, below the holder
OFFSETS_S = (-0.6, -0.45, -0.3)              # before end_s: full vial, cart not moving yet


def measure(cap, end_s):
    x0, y0, x1, y1 = BOX
    px = []
    for off in OFFSETS_S:
        cap.set(cv2.CAP_PROP_POS_MSEC, (end_s + off) * 1000)
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"no frame at {end_s + off:.2f} s")
        px.append(im[y0:y1, x0:x1, ::-1].reshape(-1, 3))
    return np.median(np.concatenate(px), axis=0)


def design(recipes):
    c = [[1, -r.ml.get("T200", 0) / sum(r.ml.values()), -r.ml.get("T400", 0) / sum(r.ml.values())]
         for r in recipes]
    return np.array(c)


def fit(recipes, rgb):
    coef, *_ = np.linalg.lstsq(design(recipes), np.log(np.clip(rgb, 1, 255) / 255), rcond=None)
    return coef                              # rows: log_i0, k200, k400; columns: R, G, B


def predict(coef, recipes):
    return np.exp(design(recipes) @ coef) * 255


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    clean = [(r, end) for v, r, _, end in vial_labels(ROOT / "data") if v % 3 == 0]
    recipes = [r for r, _ in clean]
    rgb = np.array([measure(cap, end) for _, end in clean])

    coef = fit(recipes, rgb)
    loro = []
    for i, r in enumerate(recipes):
        keep = [j for j in range(len(recipes)) if j != i]
        pred = predict(fit([recipes[j] for j in keep], rgb[keep]), [r])[0]
        loro.append(delta_e(tuple(pred), tuple(rgb[i])))
        print(f"recipe {r.recipe_id:2d} {r.ml}  measured {rgb[i].round()}  held-out pred {pred.round()}  dE {loro[-1]:.1f}")
    print(f"median held-out dE {np.median(loro):.1f} (warn threshold {COLOR_WARN_DE})")

    out = {
        "box": BOX,
        "target_rgb": {r.recipe_id: rgb[i].round(1).tolist() for i, r in enumerate(recipes)},
        "i0_rgb": (np.exp(coef[0]) * 255).round(1).tolist(),
        "absorb_T200": coef[1].round(4).tolist(),
        "absorb_T400": coef[2].round(4).tolist(),
        "loro_delta_e": {r.recipe_id: round(d, 2) for r, d in zip(recipes, loro)},
    }
    (ROOT / "twin" / "calibration.json").write_text(json.dumps(out, indent=1))
    print("wrote twin/calibration.json")


if __name__ == "__main__":
    main()
