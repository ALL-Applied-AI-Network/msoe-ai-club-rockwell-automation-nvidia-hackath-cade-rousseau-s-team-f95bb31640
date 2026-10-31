"""Scenario analysis for unmeasured physical carryover assumptions."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise, product

import numpy as np

from changeover_guard.appearance import ResidualModel
from changeover_guard.contracts import Recipe
from changeover_guard.costs import TransitionCostConfig, build_transition_matrix
from changeover_guard.optimizer import optimize_sequence

DECAY_SCENARIOS = {"fast": 0.10, "medium": 0.40, "slow": 0.70}


@dataclass(frozen=True)
class SensitivityScenario:
    carryover_ml: float
    decay_name: str
    decay: float
    residual_model: ResidualModel

    @property
    def name(self) -> str:
        return f"{self.carryover_ml:g}mL-{self.decay_name}-{self.residual_model}"


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario: SensitivityScenario
    suggested: tuple[int, ...]
    submitted_score: float
    suggested_score: float
    relative_reduction: float
    submitted_worst: float
    suggested_worst: float
    worst_reduction: float


@dataclass(frozen=True)
class SensitivityReport:
    submitted: tuple[int, ...]
    outcomes: tuple[ScenarioOutcome, ...]
    improved_scenarios: int
    median_reduction: float
    min_reduction: float
    max_reduction: float
    unique_orders: int
    most_common_order: tuple[int, ...]
    most_common_order_count: int
    original_transition_frequency: dict[tuple[int, int], int]


def default_scenarios() -> tuple[SensitivityScenario, ...]:
    return tuple(
        SensitivityScenario(carryover_ml, decay_name, decay, residual_model)
        for carryover_ml, (decay_name, decay), residual_model in product(
            (1.0, 2.0, 4.0, 6.0),
            DECAY_SCENARIOS.items(),
            ("bulk", "last", "hybrid"),
        )
    )


def analyze_sensitivity(
    recipes: Iterable[Recipe],
    submitted: list[int],
    *,
    scenarios: Iterable[SensitivityScenario] | None = None,
    worst_transition_weight: float = 0.5,
) -> SensitivityReport:
    """Optimize the same fixed-start queue under each physical scenario."""
    recipe_list = list(recipes)
    scenario_list = tuple(scenarios or default_scenarios())
    if not scenario_list:
        raise ValueError("at least one sensitivity scenario is required")

    outcomes = []
    for scenario in scenario_list:
        matrix = build_transition_matrix(
            recipe_list,
            config=TransitionCostConfig(
                carryover_ml=scenario.carryover_ml,
                decay=scenario.decay,
                residual_models=(scenario.residual_model,),
            ),
        )
        result = optimize_sequence(
            submitted,
            matrix,
            worst_transition_weight=worst_transition_weight,
        )
        outcomes.append(
            ScenarioOutcome(
                scenario=scenario,
                suggested=tuple(batch.recipe_id for batch in result.suggested),
                submitted_score=result.submitted_cost,
                suggested_score=result.suggested_cost,
                relative_reduction=result.relative_reduction,
                submitted_worst=result.submitted_worst_transition,
                suggested_worst=result.suggested_worst_transition,
                worst_reduction=result.worst_transition_reduction,
            )
        )

    reductions = np.asarray([outcome.relative_reduction for outcome in outcomes])
    order_counts = Counter(outcome.suggested for outcome in outcomes)
    most_common_order, most_common_count = min(
        order_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    optimized_transitions = [set(pairwise(outcome.suggested)) for outcome in outcomes]
    original_frequency = {
        transition: sum(transition in optimized for optimized in optimized_transitions)
        for transition in pairwise(submitted)
    }
    return SensitivityReport(
        submitted=tuple(submitted),
        outcomes=tuple(outcomes),
        improved_scenarios=int(np.count_nonzero(reductions > 1e-12)),
        median_reduction=float(np.median(reductions)),
        min_reduction=float(reductions.min()),
        max_reduction=float(reductions.max()),
        unique_orders=len(order_counts),
        most_common_order=most_common_order,
        most_common_order_count=most_common_count,
        original_transition_frequency=original_frequency,
    )
