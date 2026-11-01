"""Clean-color adapter with explicit abstention for unsupported first-vial fits."""

from changeover_guard.contracts import Prediction, Recipe, TankReading
from changeover_guard.optical_model import CleanOpticalModel, quality_decision
from changeover_guard.vial_color import VialCalibration


class PredictionUnavailable(NotImplementedError):
    """The legacy API represents unavailable scientific evidence as pending."""


def predict(
    prev: Recipe | None,
    recipe: Recipe,
    tanks: TankReading,
    *,
    model: CleanOpticalModel | None = None,
    calibration: VialCalibration | None = None,
    repeatability: dict | None = None,
) -> Prediction:
    """Predict validated steady-recipe color using an explicit vial-path model.

    Tank-camera colors are never silently treated as calibrated vial colors.
    A change to a different composition still needs an independent first-vial
    carryover predictor. The conditional recovery model cannot supply that.
    Existing API calls stay pending until artifacts are explicitly supplied.
    """
    recipe.validate()
    if not recipe.ml:
        raise ValueError("recipe must contain positive volumes")
    if prev is not None:
        prev.validate()
    if model is None or calibration is None:
        raise PredictionUnavailable(
            "A validated vial-camera optical model and calibration are required."
        )
    predicted = model.predict(recipe, calibration.calibration_id)
    evidence = quality_decision(
        predicted["rgb"], recipe, model, calibration, repeatability
    )
    if evidence["status"] == "PENDING":
        raise PredictionUnavailable(
            "Camera, clean-model validation or interpolation evidence is incomplete."
        )
    if prev is not None and prev.ml != recipe.ml:
        raise PredictionUnavailable(
            "First-vial carryover after a composition change is unvalidated; use the observed-sequence diagnostic engine."
        )
    color = tuple(predicted["rgb"])
    return Prediction(recipe, prev, color, color, 0.0, False)
