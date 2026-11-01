"""Mixing model. Owners: Adrian, Niko."""

from changeover_guard.appearance import clean_rgb
from changeover_guard.contracts import COLOR_WARN_DE, Prediction, Recipe, TankReading
from changeover_guard.costs import transition_cost


def predict(prev: Recipe | None, recipe: Recipe, tanks: TankReading) -> Prediction:
    """Predict the first-vial color and delta E for a recipe after the previous one.

    Uses the same assumed residual volume and residual-model ensemble as the scheduler
    (costs.transition_cost), so inspection and queue planning agree. The color model is
    calibrated per tank number from clean vials, so `tanks` is not read here.
    """
    clean = clean_rgb(recipe)
    if prev is None or prev.recipe_id == recipe.recipe_id:
        return Prediction(recipe, prev, clean, clean, 0.0, False)

    cost = transition_cost(prev, recipe)
    # The model's carryover shift, applied to the measured clean color (model and measured
    # clean colors differ by the calibration fit residual, which is not changeover risk).
    model_clean = clean_rgb(recipe, prefer_measured=False)
    first = tuple(
        min(255.0, max(0.0, c + p - m))
        for c, p, m in zip(clean, cost.first_vial_rgb, model_clean, strict=True)
    )
    return Prediction(
        recipe=recipe,
        prev_recipe=prev,
        clean_color=clean,
        first_vial_color=first,
        delta_e=cost.first_vial_delta_e,
        warn=cost.first_vial_delta_e >= COLOR_WARN_DE,
    )
