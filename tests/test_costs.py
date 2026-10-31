import pytest

from changeover_guard.contracts import Recipe
from changeover_guard.costs import (
    TransitionCostConfig,
    build_transition_matrix,
    sequence_cost,
)

RECIPES = [
    Recipe(6, {"T100": 30, "T200": 10}),
    Recipe(7, {"T100": 20, "T400": 50}),
    Recipe(12, {"T100": 10, "T200": 50, "T400": 10}),
]


def test_same_recipe_cost_is_zero():
    matrix = build_transition_matrix(RECIPES)
    cost = matrix.transition(6, 6)
    assert cost.total == 0
    assert not cost.warn


def test_transition_matrix_is_directed():
    matrix = build_transition_matrix(RECIPES)
    assert matrix.value(6, 7) != pytest.approx(matrix.value(7, 6))


def test_sequence_cost_includes_each_adjacent_pair():
    matrix = build_transition_matrix(RECIPES)
    expected = matrix.value(6, 7) + matrix.value(7, 12)
    assert sequence_cost([6, 7, 12], matrix) == pytest.approx(expected)


def test_uncertainty_weight_only_adds_risk_penalty():
    base = build_transition_matrix(
        RECIPES, config=TransitionCostConfig(uncertainty_weight=0)
    )
    risk = build_transition_matrix(
        RECIPES, config=TransitionCostConfig(uncertainty_weight=2)
    )
    assert risk.value(6, 7) >= base.value(6, 7)


def test_conflicting_duplicate_recipe_definition_raises():
    with pytest.raises(ValueError):
        build_transition_matrix(
            [
                Recipe(6, {"T100": 30, "T200": 10}),
                Recipe(6, {"T100": 20, "T200": 20}),
            ]
        )
