from itertools import permutations

import pytest

from changeover_guard.contracts import Recipe
from changeover_guard.costs import (
    TransitionCost,
    TransitionCostConfig,
    TransitionMatrix,
    sequence_cost,
)
from changeover_guard.optimizer import greedy_sequence, optimize_sequence


def matrix(values):
    recipe_ids = sorted({recipe_id for pair in values for recipe_id in pair})
    recipes = {recipe_id: Recipe(recipe_id, {"T100": 10}) for recipe_id in recipe_ids}
    costs = {}
    for pair, value in values.items():
        costs[pair] = TransitionCost(
            previous_recipe_id=pair[0],
            target_recipe_id=pair[1],
            total=float(value),
            color=float(value),
            uncertainty=0,
            first_vial_rgb=(0, 0, 0),
            first_vial_delta_e=float(value),
            warn=False,
            basis="test",
            warnings=(),
        )
    return TransitionMatrix(recipes, costs, TransitionCostConfig())


VALUES = {
    (1, 1): 0,
    (1, 2): 9,
    (1, 3): 2,
    (1, 4): 7,
    (2, 1): 3,
    (2, 2): 0,
    (2, 3): 6,
    (2, 4): 1,
    (3, 1): 4,
    (3, 2): 1,
    (3, 3): 0,
    (3, 4): 5,
    (4, 1): 2,
    (4, 2): 3,
    (4, 3): 1,
    (4, 4): 0,
}


def test_exact_optimizer_preserves_first_batch_and_matches_brute_force():
    costs = matrix(VALUES)
    submitted = [1, 2, 3, 4]
    result = optimize_sequence(submitted, costs)
    brute = min(
        ([submitted[0], *rest] for rest in permutations(submitted[1:])),
        key=lambda order: (sequence_cost(order, costs), order),
    )
    assert result.suggested[0].recipe_id == submitted[0]
    assert [batch.recipe_id for batch in result.suggested] == brute
    assert result.suggested_cost == pytest.approx(sequence_cost(brute, costs))


def test_optimizer_keeps_duplicate_recipes_as_distinct_batches():
    costs = matrix(VALUES)
    result = optimize_sequence([1, 2, 2, 3], costs)
    assert result.suggested[0].batch_id == "batch-1"
    assert sorted(batch.batch_id for batch in result.suggested) == [
        "batch-1",
        "batch-2",
        "batch-3",
        "batch-4",
    ]
    assert [batch.recipe_id for batch in result.suggested].count(2) == 2


def test_exact_is_no_worse_than_greedy():
    costs = matrix(VALUES)
    exact = optimize_sequence([1, 2, 3, 4], costs)
    greedy = greedy_sequence([1, 2, 3, 4], costs)
    assert exact.suggested_cost <= greedy.suggested_cost


def test_one_batch_returns_itself():
    costs = matrix({(1, 1): 0})
    result = optimize_sequence([1], costs)
    assert result.submitted == result.suggested
    assert result.suggested_cost == 0


def test_empty_queue_raises():
    with pytest.raises(ValueError):
        optimize_sequence([], matrix({(1, 1): 0}))


def test_worst_transition_penalty_avoids_one_large_edge():
    values = {
        (1, 1): 0,
        (1, 2): 0,
        (1, 3): 2,
        (1, 4): 20,
        (2, 1): 20,
        (2, 2): 0,
        (2, 3): 0,
        (2, 4): 1,
        (3, 1): 20,
        (3, 2): 2,
        (3, 3): 0,
        (3, 4): 5,
        (4, 1): 20,
        (4, 2): 20,
        (4, 3): 20,
        (4, 4): 0,
    }
    costs = matrix(values)
    sum_only = optimize_sequence([1, 2, 3, 4], costs)
    risk_aware = optimize_sequence([1, 2, 3, 4], costs, worst_transition_weight=1.0)

    assert [batch.recipe_id for batch in sum_only.suggested] == [1, 2, 3, 4]
    assert [batch.recipe_id for batch in risk_aware.suggested] == [1, 3, 2, 4]
    assert risk_aware.suggested_worst_transition < sum_only.suggested_worst_transition
    assert risk_aware.method == "exact-held-karp-minimax"
