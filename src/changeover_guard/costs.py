"""Directed recipe-transition costs for production-sequence optimization."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from changeover_guard.appearance import (
    AppearanceCalibration,
    ResidualModel,
    clean_rgb,
    load_calibration,
)
from changeover_guard.carryover import carryover_at_position, predict_carryover_color
from changeover_guard.contracts import COLOR_WARN_DE, RGB, Recipe, delta_e


@dataclass(frozen=True)
class TransitionCostConfig:
    """Assumptions used until physical carryover/flush measurements exist."""

    carryover_ml: float = 2.0
    decay: float = 0.25
    vial_count: int = 3
    discount: float = 0.7
    uncertainty_weight: float = 1.0
    residual_models: tuple[ResidualModel, ...] = ("bulk", "last", "hybrid")
    hybrid_alpha: float = 0.5

    def validate(self) -> None:
        if self.carryover_ml < 0:
            raise ValueError("carryover_ml must be nonnegative")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay must be between 0 and 1")
        if self.vial_count < 1:
            raise ValueError("vial_count must be at least 1")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if self.uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be nonnegative")
        if not self.residual_models:
            raise ValueError("at least one residual model is required")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError("hybrid_alpha must be between 0 and 1")


@dataclass(frozen=True)
class TransitionCost:
    previous_recipe_id: int
    target_recipe_id: int
    total: float
    color: float
    uncertainty: float
    first_vial_rgb: RGB
    first_vial_delta_e: float
    warn: bool
    basis: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TransitionMatrix:
    recipes: dict[int, Recipe]
    costs: dict[tuple[int, int], TransitionCost]
    config: TransitionCostConfig

    def transition(
        self, previous_recipe_id: int, target_recipe_id: int
    ) -> TransitionCost:
        try:
            return self.costs[(previous_recipe_id, target_recipe_id)]
        except KeyError as exc:
            raise KeyError(
                f"transition {previous_recipe_id} -> {target_recipe_id} is not in the matrix"
            ) from exc

    def value(self, previous_recipe_id: int, target_recipe_id: int) -> float:
        return self.transition(previous_recipe_id, target_recipe_id).total


def _same_recipe(recipe: Recipe, calibration: AppearanceCalibration) -> TransitionCost:
    rgb = clean_rgb(recipe, calibration)
    return TransitionCost(
        previous_recipe_id=recipe.recipe_id,
        target_recipe_id=recipe.recipe_id,
        total=0.0,
        color=0.0,
        uncertainty=0.0,
        first_vial_rgb=rgb,
        first_vial_delta_e=0.0,
        warn=False,
        basis="same recipe; no changeover",
        warnings=(),
    )


def transition_cost(
    previous: Recipe,
    target: Recipe,
    *,
    config: TransitionCostConfig | None = None,
    calibration: AppearanceCalibration | None = None,
) -> TransitionCost:
    """Risk-adjusted color cost for a directed recipe changeover."""
    config = config or TransitionCostConfig()
    config.validate()
    calibration = calibration or load_calibration()
    previous.validate()
    target.validate()
    if previous.recipe_id == target.recipe_id and previous.ml == target.ml:
        return _same_recipe(target, calibration)

    # Model-predicted clean color, not the measured one: the carryover predictions below are
    # model outputs, and measured-vs-model differs by the calibration fit residual (up to ~10 dE),
    # which would add a per-target floor unrelated to changeover risk.
    clean = clean_rgb(target, calibration, prefer_measured=False)
    target_ml = float(sum(target.ml.values()))
    initial_fraction = (
        config.carryover_ml / (target_ml + config.carryover_ml)
        if config.carryover_ml
        else 0.0
    )
    model_scores: list[float] = []
    first_predictions: list[RGB] = []
    first_errors: list[float] = []
    for model in config.residual_models:
        score = 0.0
        for position in range(1, config.vial_count + 1):
            fraction = carryover_at_position(initial_fraction, config.decay, position)
            prediction = predict_carryover_color(
                previous,
                target,
                fraction,
                residual_model=model,
                alpha=config.hybrid_alpha,
                calibration=calibration,
            )
            error = delta_e(clean, prediction.rgb)
            score += config.discount ** (position - 1) * error
            if position == 1:
                first_predictions.append(prediction.rgb)
                first_errors.append(error)
        model_scores.append(score)

    color_cost = float(np.mean(model_scores))
    uncertainty = float(np.std(model_scores))
    total = color_cost + config.uncertainty_weight * uncertainty
    first_rgb = tuple(float(channel) for channel in np.mean(first_predictions, axis=0))
    first_error = float(np.mean(first_errors))
    return TransitionCost(
        previous_recipe_id=previous.recipe_id,
        target_recipe_id=target.recipe_id,
        total=total,
        color=color_cost,
        uncertainty=uncertainty,
        first_vial_rgb=first_rgb,
        first_vial_delta_e=first_error,
        warn=first_error >= COLOR_WARN_DE,
        basis=(
            f"assumed {config.carryover_ml:g} mL residual; ensemble of "
            + ", ".join(config.residual_models)
        ),
        warnings=(
            "Transition cost is model-derived from an assumed residual volume.",
            "It is a relative scheduling score, not measured scrap or waste.",
        ),
    )


def _unique_recipes(recipes: Iterable[Recipe]) -> dict[int, Recipe]:
    out: dict[int, Recipe] = {}
    for recipe in recipes:
        recipe.validate()
        if recipe.recipe_id in out and out[recipe.recipe_id].ml != recipe.ml:
            raise ValueError(
                f"recipe ID {recipe.recipe_id} has conflicting compositions"
            )
        out[recipe.recipe_id] = recipe
    if not out:
        raise ValueError("at least one recipe is required")
    return out


def build_transition_matrix(
    recipes: Iterable[Recipe],
    *,
    config: TransitionCostConfig | None = None,
    calibration: AppearanceCalibration | None = None,
) -> TransitionMatrix:
    """Build every directed pair once; duplicate batches reuse pair costs."""
    config = config or TransitionCostConfig()
    config.validate()
    calibration = calibration or load_calibration()
    unique = _unique_recipes(recipes)
    costs = {
        (previous_id, target_id): transition_cost(
            previous,
            target,
            config=config,
            calibration=calibration,
        )
        for previous_id, previous in unique.items()
        for target_id, target in unique.items()
    }
    return TransitionMatrix(unique, costs, config)


def sequence_cost(recipe_ids: Iterable[int], matrix: TransitionMatrix) -> float:
    order = list(recipe_ids)
    return float(sum(matrix.value(a, b) for a, b in pairwise(order)))
