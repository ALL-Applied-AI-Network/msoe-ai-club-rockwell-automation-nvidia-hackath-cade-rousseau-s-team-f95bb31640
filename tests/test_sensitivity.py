from changeover_guard.contracts import Recipe
from changeover_guard.sensitivity import (
    SensitivityScenario,
    analyze_sensitivity,
    default_scenarios,
)

RECIPES = [
    Recipe(6, {"T100": 30, "T200": 10}),
    Recipe(7, {"T100": 20, "T400": 50}),
    Recipe(12, {"T100": 10, "T200": 50, "T400": 10}),
]


def test_default_sensitivity_grid_has_36_scenarios():
    scenarios = default_scenarios()
    assert len(scenarios) == 36
    assert {scenario.carryover_ml for scenario in scenarios} == {1, 2, 4, 6}
    assert {scenario.residual_model for scenario in scenarios} == {
        "bulk",
        "last",
        "hybrid",
    }


def test_sensitivity_preserves_first_recipe_and_reports_each_scenario():
    scenarios = (
        SensitivityScenario(1, "fast", 0.1, "bulk"),
        SensitivityScenario(4, "slow", 0.7, "hybrid"),
    )
    report = analyze_sensitivity(RECIPES, [6, 7, 12], scenarios=scenarios)
    assert len(report.outcomes) == 2
    assert all(outcome.suggested[0] == 6 for outcome in report.outcomes)
    assert all(
        outcome.suggested_score <= outcome.submitted_score
        for outcome in report.outcomes
    )
    assert set(report.original_transition_frequency) == {(6, 7), (7, 12)}
