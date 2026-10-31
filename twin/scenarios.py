"""Build the three labeled render sets for the optimization model. Writes synthetic/scenarios.csv.

    uv run python twin/scenarios.py

Then render them on Brev with `/isaac-sim/python.sh twin/render.py --scenarios` (see BREV.md).

Sets:
  robustness  named transitions x injected fraction x lighting replicates; tests whether the
              estimator recovers a known fraction through a real image pipeline.
  decay       first five vials after a changeover under fast/medium/slow decay hypotheses.
  demo        first vial of every changeover in two run orders, for the side-by-side visual.

The injected colors use the same composition mixing and Beer-Lambert map as
changeover_guard.appearance (verified identical), so an estimator configured with
residual_model="bulk" is inverting exactly this forward model: the measured error comes from
rendering and measurement, not from the model. That is the point of the robustness set.
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np

from changeover_guard.contracts import DISPENSE_ORDER, TANKS, Recipe
from changeover_guard.dataset import vial_labels

ROOT = Path(__file__).parents[1]
OUT = ROOT / "synthetic" / "scenarios.csv"
CALIBRATION = ROOT / "twin" / "calibration.json"

# Transitions whose color actually moves when contaminated (dE between clean and 40% carryover,
# strongest first). Pairs below the twin's own ~dE 3 render error are not worth GPU time.
# 6->7 is the one changeover observed on the real line; 5->8 is a null control: recipes 5 and 8
# have identical dye ratios, so no fraction can change the color.
TRANSITIONS = [(7, 6), (11, 6), (7, 12), (11, 12), (9, 6), (9, 12), (6, 7), (5, 6), (8, 6), (7, 13)]
NULL_CONTROL = (5, 8)
# The one real changeover best-fits ~0.39 residual, so bracket well past the original 0-10% plan.
FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.40)
REPLICATES = 3
DECAYS = {"fast": 0.2, "medium": 0.5, "slow": 0.8}   # real data shows vials 2-3 clean => fast
DECAY_TRANSITIONS = TRANSITIONS[:6]
DECAY_INITIAL = 0.40
POSITIONS = 5
RESIDUAL_MODEL, ALPHA = "bulk", 1.0
AS_RUN = list(range(5, 15))                          # the order video 5 actually ran

COLUMNS = ["set", "vial", "prev_recipe_id", "recipe_id", "t100_ml", "t200_ml", "t400_ml",
           "injected_fraction", "residual_model", "alpha", "position", "decay_scenario", "decay",
           "light", "exposure", "seed", "r", "g", "b"]


def recipe_fraction(recipe: Recipe) -> np.ndarray:
    """Per-tank fractions of a recipe, ordered as TANKS."""
    total = sum(recipe.ml.values())
    return np.array([recipe.ml.get(t, 0.0) for t in TANKS]) / total


def last_component_fraction(recipe: Recipe) -> np.ndarray:
    """Pure component dispensed last (T200 -> T400 -> T100, skipping unused tanks)."""
    used = [t for t in DISPENSE_ORDER if recipe.ml.get(t, 0) > 0]
    return np.array([float(t == used[-1]) for t in TANKS])


def residual_composition(previous: Recipe, model: str, alpha: float) -> np.ndarray:
    bulk, last = recipe_fraction(previous), last_component_fraction(previous)
    if model == "bulk":
        return bulk
    if model == "last":
        return last
    if model == "hybrid":
        return alpha * bulk + (1 - alpha) * last
    raise ValueError(f"unknown residual model: {model}")


def injected_rgb(cal: dict, previous: Recipe, target: Recipe, fraction: float,
                 model: str = RESIDUAL_MODEL, alpha: float = ALPHA) -> np.ndarray:
    """Camera RGB of target contaminated by `fraction` of previous, as changeover_guard predicts."""
    composition = (1 - fraction) * recipe_fraction(target) + fraction * residual_composition(previous, model, alpha)
    log_i0 = np.log(np.array(cal["i0_rgb"]) / 255)
    absorb = np.array([cal["absorb_T200"], cal["absorb_T400"]])
    return np.clip(np.exp(log_i0 - composition[1:] @ absorb) * 255, 0, 255).round(1)


def row(set_name: str, vial: int, previous: Recipe, target: Recipe, fraction: float, rng: random.Random,
        position: int = 1, decay_scenario: str = "", decay: float = "", cal: dict | None = None) -> list:
    seed = rng.randrange(1_000_000)
    jitter = random.Random(seed)
    return [set_name, vial, previous.recipe_id, target.recipe_id,
            *(target.ml.get(t, "") for t in TANKS),
            round(fraction, 4), RESIDUAL_MODEL, ALPHA, position, decay_scenario, decay,
            round(jitter.random(), 3), round(jitter.uniform(0.9, 1.1), 3), seed,
            *injected_rgb(cal, previous, target, fraction)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimized", type=int, nargs="*", default=None,
                    help="optimized run order for the demo set (default: the as-run order reversed)")
    a = ap.parse_args()

    cal = json.loads(CALIBRATION.read_text())
    catalog = {r.recipe_id: r for _, r, _, _ in vial_labels(ROOT / "data")}
    rng = random.Random(a.seed)
    rows, vial = [], 0

    def add(*args, **kwargs):
        nonlocal vial
        vial += 1
        rows.append(row(*args[:1], vial, *args[1:], rng=rng, cal=cal, **kwargs))

    for prev_id, target_id in [*TRANSITIONS, NULL_CONTROL]:
        for fraction in FRACTIONS:
            for _ in range(REPLICATES):
                add("robustness", catalog[prev_id], catalog[target_id], fraction)

    for prev_id, target_id in DECAY_TRANSITIONS:
        for name, decay in DECAYS.items():
            for position in range(1, POSITIONS + 1):
                fraction = DECAY_INITIAL * decay ** (position - 1)
                add("decay", catalog[prev_id], catalog[target_id], fraction,
                    position=position, decay_scenario=name, decay=decay)

    optimized = a.optimized or AS_RUN[::-1]
    for name, order in (("demo_as_run", AS_RUN), ("demo_optimized", optimized)):
        for prev_id, target_id in itertools.pairwise(order):
            add(name, catalog[prev_id], catalog[target_id], DECAY_INITIAL)
            add(name, catalog[prev_id], catalog[target_id], 0.0)   # the clean reference beside it

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        csv.writer(fh).writerows([COLUMNS, *rows])
    counts = {s: sum(r[0] == s for r in rows) for s in dict.fromkeys(r[0] for r in rows)}
    print(f"wrote {len(rows)} vials to {OUT.relative_to(ROOT)}: " +
          ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
