"""Forward and inverse carryover estimates under explicit model assumptions."""

from dataclasses import dataclass

import numpy as np

from changeover_guard.appearance import (
    AppearanceCalibration,
    ResidualModel,
    load_calibration,
    mix_compositions,
    predict_rgb,
    recipe_fraction,
    residual_composition,
)
from changeover_guard.contracts import RGB, Recipe, delta_e


@dataclass(frozen=True)
class ColorPrediction:
    rgb: RGB
    composition: dict[str, float]
    carryover_fraction: float
    residual_model: ResidualModel


@dataclass(frozen=True)
class CarryoverConfig:
    max_fraction: float = 0.20
    grid_step: float = 0.001
    residual_model: ResidualModel = "hybrid"
    alpha: float = 0.5
    likelihood_sigma_de: float = 5.0

    def validate(self) -> None:
        if not 0 < self.max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")
        if not 0 < self.grid_step <= self.max_fraction:
            raise ValueError(
                "grid_step must be positive and no larger than max_fraction"
            )
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.likelihood_sigma_de <= 0:
            raise ValueError("likelihood_sigma_de must be positive")


@dataclass(frozen=True)
class CarryoverProfilePoint:
    fraction: float
    error_delta_e: float
    posterior: float


@dataclass(frozen=True)
class CarryoverEstimate:
    fraction_hat: float
    median_fraction: float
    ci90: tuple[float, float]
    fit_error: float
    residual_model: ResidualModel
    confidence: str
    profile: tuple[CarryoverProfilePoint, ...]
    warnings: tuple[str, ...]


def predict_carryover_color(
    previous: Recipe,
    target: Recipe,
    fraction: float,
    *,
    residual_model: ResidualModel = "hybrid",
    alpha: float = 0.5,
    calibration: AppearanceCalibration | None = None,
) -> ColorPrediction:
    """Predict observed color after residual from ``previous`` enters ``target``."""
    calibration = calibration or load_calibration()
    target_fraction = recipe_fraction(target)
    residual = residual_composition(previous, residual_model, alpha)
    composition = mix_compositions(target_fraction, residual, fraction)
    return ColorPrediction(
        rgb=predict_rgb(composition, calibration),
        composition=composition,
        carryover_fraction=float(fraction),
        residual_model=residual_model,
    )


def carryover_at_position(
    initial_fraction: float, decay: float, position: int
) -> float:
    """Residual fraction for a one-indexed vial position after changeover."""
    if not 0 <= initial_fraction <= 1:
        raise ValueError("initial_fraction must be between 0 and 1")
    if not 0 <= decay <= 1:
        raise ValueError("decay must be between 0 and 1")
    if position < 1:
        raise ValueError("position must be at least 1")
    return float(initial_fraction * decay ** (position - 1))


def _quantile(grid: np.ndarray, posterior: np.ndarray, q: float) -> float:
    index = int(np.searchsorted(np.cumsum(posterior), q, side="left"))
    return float(grid[min(index, len(grid) - 1)])


def estimate_carryover(
    observed_rgb: RGB,
    previous: Recipe,
    target: Recipe,
    *,
    config: CarryoverConfig | None = None,
    calibration: AppearanceCalibration | None = None,
) -> CarryoverEstimate:
    """Estimate an equivalent residual fraction by a one-dimensional grid."""
    config = config or CarryoverConfig()
    config.validate()
    calibration = calibration or load_calibration()
    observed = tuple(float(channel) for channel in observed_rgb)
    if len(observed) != 3 or not all(np.isfinite(observed)):
        raise ValueError("observed_rgb must contain three finite channels")

    count = round(config.max_fraction / config.grid_step)
    grid = np.linspace(0.0, config.max_fraction, count + 1)
    errors = np.asarray(
        [
            delta_e(
                observed,
                predict_carryover_color(
                    previous,
                    target,
                    float(fraction),
                    residual_model=config.residual_model,
                    alpha=config.alpha,
                    calibration=calibration,
                ).rgb,
            )
            for fraction in grid
        ]
    )
    best = int(np.argmin(errors))
    log_likelihood = -0.5 * (errors / config.likelihood_sigma_de) ** 2
    weights = np.exp(log_likelihood - log_likelihood.max())
    posterior = weights / weights.sum()
    low, median, high = (_quantile(grid, posterior, q) for q in (0.05, 0.5, 0.95))
    width = high - low
    confidence = "high" if width <= 0.03 else "medium" if width <= 0.08 else "low"
    profile = tuple(
        CarryoverProfilePoint(float(fraction), float(error), float(probability))
        for fraction, error, probability in zip(grid, errors, posterior)
    )
    return CarryoverEstimate(
        fraction_hat=float(grid[best]),
        median_fraction=median,
        ci90=(low, high),
        fit_error=float(errors[best]),
        residual_model=config.residual_model,
        confidence=confidence,
        profile=profile,
        warnings=(
            "Equivalent residual under the empirical appearance model; not a direct physical measurement.",
            "Interval reflects configured image/model noise and is not yet physically calibrated.",
        ),
    )
