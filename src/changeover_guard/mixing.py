"""Mixing model. Owners: Adrian, Niko."""

from changeover_guard.contracts import Prediction, Recipe, TankReading


def predict(prev: Recipe | None, recipe: Recipe, tanks: TankReading) -> Prediction:
    """Predict the first-vial color and delta E for a recipe after the previous one."""
    raise NotImplementedError
