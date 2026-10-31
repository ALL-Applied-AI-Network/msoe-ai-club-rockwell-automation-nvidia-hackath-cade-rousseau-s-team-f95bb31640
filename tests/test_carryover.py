import pytest

from changeover_guard.carryover import (
    CarryoverConfig,
    carryover_at_position,
    estimate_carryover,
    predict_carryover_color,
)
from changeover_guard.contracts import Recipe

PREVIOUS = Recipe(7, {"T100": 20, "T400": 50})
TARGET = Recipe(12, {"T100": 10, "T200": 50, "T400": 10})


def test_inverse_round_trip_recovers_synthetic_fraction():
    truth = 0.073
    observed = predict_carryover_color(
        PREVIOUS, TARGET, truth, residual_model="bulk"
    ).rgb
    estimate = estimate_carryover(
        observed,
        PREVIOUS,
        TARGET,
        config=CarryoverConfig(residual_model="bulk", grid_step=0.001),
    )
    assert estimate.fraction_hat == pytest.approx(truth, abs=0.001)
    assert sum(point.posterior for point in estimate.profile) == pytest.approx(1)


def test_carryover_decays_across_vials():
    values = [carryover_at_position(0.1, 0.25, position) for position in (1, 2, 3)]
    assert values[0] > values[1] > values[2]


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_forward_fraction_outside_unit_interval_raises(fraction):
    with pytest.raises(ValueError):
        predict_carryover_color(PREVIOUS, TARGET, fraction)


def test_unknown_observation_is_not_interpreted_as_zero():
    with pytest.raises(ValueError):
        estimate_carryover((float("nan"), 0, 0), PREVIOUS, TARGET)
