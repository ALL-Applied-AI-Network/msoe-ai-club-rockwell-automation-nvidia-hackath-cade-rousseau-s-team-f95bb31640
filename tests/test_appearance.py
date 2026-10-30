import numpy as np
import pytest

from changeover_guard.appearance import (
    last_component_fraction,
    mix_compositions,
    predict_rgb,
    recipe_fraction,
    residual_composition,
)
from changeover_guard.contracts import Recipe


def test_recipe_fractions_sum_to_one():
    fractions = recipe_fraction(Recipe(7, {"T100": 20, "T400": 50}))
    assert sum(fractions.values()) == pytest.approx(1)
    assert fractions["T200"] == 0


def test_last_component_follows_dispense_order():
    assert last_component_fraction(Recipe(5, {"T100": 10, "T200": 10, "T400": 10})) == {
        "T100": 1.0,
        "T200": 0.0,
        "T400": 0.0,
    }
    assert last_component_fraction(Recipe(13, {"T200": 50, "T400": 20})) == {
        "T100": 0.0,
        "T200": 0.0,
        "T400": 1.0,
    }


def test_hybrid_residual_endpoints_match_bulk_and_last():
    recipe = Recipe(7, {"T100": 20, "T400": 50})
    assert residual_composition(recipe, "hybrid", 1) == residual_composition(
        recipe, "bulk"
    )
    assert residual_composition(recipe, "hybrid", 0) == residual_composition(
        recipe, "last"
    )


def test_zero_and_full_mixture_match_endpoints():
    target = recipe_fraction(Recipe(6, {"T100": 30, "T200": 10}))
    residual = recipe_fraction(Recipe(7, {"T100": 20, "T400": 50}))
    assert mix_compositions(target, residual, 0) == target
    assert mix_compositions(target, residual, 1) == residual


def test_invalid_fraction_raises():
    composition = recipe_fraction(Recipe(6, {"T100": 30, "T200": 10}))
    with pytest.raises(ValueError):
        mix_compositions(composition, composition, 1.01)


def test_prediction_is_finite_rgb():
    rgb = predict_rgb(recipe_fraction(Recipe(9, {"T100": 10, "T200": 35, "T400": 35})))
    assert len(rgb) == 3
    assert all(np.isfinite(rgb))
    assert all(0 <= channel <= 255 for channel in rgb)
