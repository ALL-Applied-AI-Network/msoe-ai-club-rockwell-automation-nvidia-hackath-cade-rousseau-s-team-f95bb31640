"""Empirical recipe-composition to camera-appearance model.

The committed calibration is a Beer-Lambert-inspired camera model fitted to
Run 5.  It is useful for ranking and synthetic validation, but it is not a
physical measurement of contamination.  This module keeps that distinction
explicit and does not modify the teammate-owned ``mixing.py`` integration.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np

from changeover_guard.contracts import DISPENSE_ORDER, RGB, TANKS, Recipe

ResidualModel = Literal["bulk", "last", "hybrid"]
ROOT = Path(__file__).parents[2]
DEFAULT_CALIBRATION = ROOT / "twin" / "calibration.json"


@dataclass(frozen=True)
class AppearanceCalibration:
    """Parameters of the empirical RGB appearance model."""

    i0_rgb: RGB
    absorb_t200: RGB
    absorb_t400: RGB
    target_rgb: dict[int, RGB]
    source: str


def _rgb(values: object, field: str) -> RGB:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{field} must contain three channels")
    out = tuple(float(value) for value in values)
    if not all(np.isfinite(out)):
        raise ValueError(f"{field} must contain finite channels")
    return out


@cache
def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> AppearanceCalibration:
    """Load and validate the committed empirical calibration."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    targets = {
        int(recipe_id): _rgb(rgb, f"target_rgb[{recipe_id}]")
        for recipe_id, rgb in raw.get("target_rgb", {}).items()
    }
    calibration = AppearanceCalibration(
        i0_rgb=_rgb(raw.get("i0_rgb"), "i0_rgb"),
        absorb_t200=_rgb(raw.get("absorb_T200"), "absorb_T200"),
        absorb_t400=_rgb(raw.get("absorb_T400"), "absorb_T400"),
        target_rgb=targets,
        source=str(source),
    )
    if any(channel <= 0 for channel in calibration.i0_rgb):
        raise ValueError("i0_rgb channels must be positive")
    return calibration


def recipe_fraction(recipe: Recipe) -> dict[str, float]:
    """Return a validated recipe as fractions over the three active tanks."""
    recipe.validate()
    total = float(sum(recipe.ml.values()))
    if total <= 0:
        raise ValueError("recipe total must be positive")
    return {tank: float(recipe.ml.get(tank, 0.0)) / total for tank in TANKS}


def last_component_fraction(recipe: Recipe) -> dict[str, float]:
    """Return the pure component physically dispensed last for a recipe."""
    recipe.validate()
    used = [tank for tank in DISPENSE_ORDER if recipe.ml.get(tank, 0) > 0]
    if not used:
        raise ValueError("recipe must use at least one tank")
    last = used[-1]
    return {tank: float(tank == last) for tank in TANKS}


def residual_composition(
    previous: Recipe,
    model: ResidualModel = "hybrid",
    alpha: float = 0.5,
) -> dict[str, float]:
    """Composition of the hypothesized residual source."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    bulk = recipe_fraction(previous)
    last = last_component_fraction(previous)
    if model == "bulk":
        return bulk
    if model == "last":
        return last
    if model == "hybrid":
        return {tank: alpha * bulk[tank] + (1 - alpha) * last[tank] for tank in TANKS}
    raise ValueError(f"unknown residual model: {model}")


def mix_compositions(
    target: dict[str, float],
    residual: dict[str, float],
    fraction: float,
) -> dict[str, float]:
    """Mix a final-vial residual fraction into the target composition."""
    if not 0 <= fraction <= 1:
        raise ValueError("carryover fraction must be between 0 and 1")
    out = {
        tank: (1 - fraction) * float(target.get(tank, 0))
        + fraction * float(residual.get(tank, 0))
        for tank in TANKS
    }
    if not np.isclose(sum(out.values()), 1.0):
        raise ValueError("target and residual compositions must each sum to one")
    return out


def predict_rgb(
    composition: dict[str, float],
    calibration: AppearanceCalibration | None = None,
) -> RGB:
    """Predict camera RGB for a normalized composition."""
    calibration = calibration or load_calibration()
    values = np.array([float(composition.get(tank, 0)) for tank in TANKS])
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("composition must be nonnegative and sum to one")
    log_i0 = np.log(np.asarray(calibration.i0_rgb) / 255.0)
    log_rgb = (
        log_i0
        - values[1] * np.asarray(calibration.absorb_t200)
        - values[2] * np.asarray(calibration.absorb_t400)
    )
    rgb = np.clip(np.exp(log_rgb) * 255.0, 0.0, 255.0)
    return tuple(float(channel) for channel in rgb)


def clean_rgb(
    recipe: Recipe,
    calibration: AppearanceCalibration | None = None,
    *,
    prefer_measured: bool = True,
) -> RGB:
    """Return measured clean RGB when available, otherwise model prediction."""
    calibration = calibration or load_calibration()
    if prefer_measured and recipe.recipe_id in calibration.target_rgb:
        return calibration.target_rgb[recipe.recipe_id]
    return predict_rgb(recipe_fraction(recipe), calibration)
