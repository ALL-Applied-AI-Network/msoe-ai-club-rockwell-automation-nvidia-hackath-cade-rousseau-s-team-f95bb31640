"""List every vial to render: valid recipes in random order, 3 vials each. Writes synthetic/manifest.csv.

    uv run python twin/manifest.py --limit 60

Colors come from twin/calibration.json. The first vial of each recipe carries carry_ml of the previous
recipe's mix; the real nozzle volume is unknown, so it's swept.
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np

from changeover_guard.contracts import Recipe

ROOT = Path(__file__).parents[1]
CARRY_ML = (0, 1, 2, 4)
COLUMNS = ["vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "prev_recipe_id", "carry_ml", "light", "r", "g", "b"]


def recipes() -> list[Recipe]:
    """Every recipe on a 10 mL grid that passes Recipe.validate(), ids from 100."""
    out = []
    for i, (a, b, c) in enumerate(itertools.product(range(0, 90, 10), repeat=3)):
        ml = {t: v for t, v in zip(("T100", "T200", "T400"), (a, b, c)) if v}
        try:
            Recipe(100 + i, ml).validate()
        except ValueError:
            continue
        if ml:
            out.append(Recipe(100 + i, ml))
    return out


def dye(r: Recipe) -> np.ndarray:
    """(c200, c400): each dye's fraction of the vial."""
    total = sum(r.ml.values())
    return np.array([r.ml.get("T200", 0), r.ml.get("T400", 0)]) / total


def color(cal: dict, c: np.ndarray) -> np.ndarray:
    """Camera RGB (0-255) of dye fractions c, from calibration.json's Beer-Lambert fit."""
    log_i0 = np.log(np.array(cal["i0_rgb"]) / 255)
    k = np.array([cal["absorb_T200"], cal["absorb_T400"]])
    return np.clip(np.exp(log_i0 - c @ k) * 255, 0, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max vials")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cal = json.loads((ROOT / "twin" / "calibration.json").read_text())
    rng = random.Random(a.seed)
    order = recipes()
    rng.shuffle(order)
    rows, prev = [], None
    for r in order:
        for k in range(3):
            carry = rng.choice(CARRY_ML) if k == 0 and prev else 0
            total = sum(r.ml.values())
            c = (dye(r) * total + (dye(prev) * carry if carry else 0)) / (total + carry)
            rgb = color(cal, c).round(1)
            rows.append([len(rows) + 1, r.recipe_id, *(r.ml.get(t, "") for t in ("T100", "T200", "T400")),
                         prev.recipe_id if prev else "", carry, round(rng.random(), 3), *rgb])
        prev = r
    rows = rows[:a.limit]

    out = ROOT / "synthetic" / "manifest.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows([COLUMNS, *rows])
    print(f"{len(order)} valid recipes, wrote {len(rows)} vials to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
