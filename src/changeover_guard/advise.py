"""Explanations and advice. Owners: Adrian, Niko."""

import math

from changeover_guard import mixing
from changeover_guard.contracts import (
    COLOR_WARN_DE,
    FILL_TOL_ML,
    QUEUE_SIZE,
    Advice,
    Check,
    Prediction,
    Recipe,
    TankReading,
    VialMeasurement,
    delta_e,
)


def _describe(recipe: Recipe) -> str:
    return " + ".join(f"{tank} {ml:g} mL" for tank, ml in recipe.ml.items())


def check(prediction: Prediction, measurement: VialMeasurement) -> Check:
    """Compare a measured vial against its recipe's clean color (and fill, when measurable)."""
    color_de = delta_e(prediction.clean_color, measurement.color)
    fill_error = measurement.fill_ml - sum(prediction.recipe.ml.values())
    fill_ok = math.isnan(fill_error) or abs(fill_error) <= FILL_TOL_ML
    return Check(color_delta_e=color_de, fill_error_ml=fill_error, ok=color_de < COLOR_WARN_DE and fill_ok)


def advise(
    prediction: Prediction,
    check: Check,
    queue: list[Recipe],
    tanks: TankReading,
) -> Advice:
    """Explain a failed check and advise on flushing and the next recipe; queue is in run order."""
    recipe, prev = prediction.recipe, prediction.prev_recipe
    cause = None
    if not check.ok:
        if check.color_delta_e < COLOR_WARN_DE:
            cause = f"Fill is {check.fill_error_ml:+.0f} mL off the {sum(recipe.ml.values()):g} mL target."
        elif prev is not None and prev.recipe_id != recipe.recipe_id:
            cause = (
                f"First vial after the change from Recipe {prev.recipe_id} ({_describe(prev)}): "
                f"leftover Recipe {prev.recipe_id} liquid in the shared mixing path likely shifted the color "
                f"(ΔE {check.color_delta_e:.1f} from Recipe {recipe.recipe_id}'s clean color). "
                "The next vials of this batch should clear it."
            )
        else:
            cause = (
                f"Off-color (ΔE {check.color_delta_e:.1f}) without a recipe change, so not carryover: "
                "check the tank contents and the dispensed volumes."
            )

    # Of the recipes the Batch Scheduler could run next, suggest the one this recipe contaminates least.
    candidates = [r for r in queue[:QUEUE_SIZE] if r.recipe_id != recipe.recipe_id]
    next_recipe = min(candidates, key=lambda r: mixing.predict(recipe, r, tanks).delta_e, default=None)
    return Advice(
        cause=cause,
        flush=False,                         # no approved flush procedure on this line
        next_recipe_id=next_recipe.recipe_id if next_recipe else None,
    )
