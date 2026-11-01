"""Duplicate-safe production sequencing with a fixed first batch."""

from dataclasses import dataclass
from itertools import pairwise
from math import isclose

from changeover_guard.costs import TransitionMatrix, sequence_cost


@dataclass(frozen=True)
class Batch:
    batch_id: str
    recipe_id: int
    submitted_position: int


@dataclass(frozen=True)
class OptimizationResult:
    submitted: tuple[Batch, ...]
    suggested: tuple[Batch, ...]
    submitted_cost: float
    suggested_cost: float
    submitted_transition_sum: float
    suggested_transition_sum: float
    submitted_worst_transition: float
    suggested_worst_transition: float
    relative_reduction: float
    transition_sum_reduction: float
    worst_transition_reduction: float
    worst_transition_weight: float
    method: str

    @property
    def fixed_start(self) -> Batch:
        return self.submitted[0]


def make_batches(recipe_ids: list[int]) -> tuple[Batch, ...]:
    if not recipe_ids:
        raise ValueError("at least one recipe is required")
    return tuple(
        Batch(
            batch_id=f"batch-{position + 1}",
            recipe_id=int(recipe_id),
            submitted_position=position,
        )
        for position, recipe_id in enumerate(recipe_ids)
    )


def _validate(batches: tuple[Batch, ...], matrix: TransitionMatrix) -> None:
    missing = sorted({batch.recipe_id for batch in batches} - matrix.recipes.keys())
    if missing:
        raise ValueError(f"recipes missing from transition matrix: {missing}")


def _result(
    submitted: tuple[Batch, ...],
    suggested: tuple[Batch, ...],
    matrix: TransitionMatrix,
    method: str,
    worst_transition_weight: float,
) -> OptimizationResult:
    if worst_transition_weight < 0:
        raise ValueError("worst_transition_weight must be nonnegative")

    def metrics(order: tuple[Batch, ...]) -> tuple[float, float, float]:
        transition_sum = sequence_cost((batch.recipe_id for batch in order), matrix)
        transitions = [
            matrix.value(previous.recipe_id, target.recipe_id)
            for previous, target in pairwise(order)
        ]
        worst = max(transitions, default=0.0)
        return transition_sum + worst_transition_weight * worst, transition_sum, worst

    submitted_cost, submitted_sum, submitted_worst = metrics(submitted)
    suggested_cost, suggested_sum, suggested_worst = metrics(suggested)
    reduction = (
        0.0
        if submitted_cost == 0
        else (submitted_cost - suggested_cost) / submitted_cost
    )
    worst_reduction = (
        0.0
        if submitted_worst == 0
        else (submitted_worst - suggested_worst) / submitted_worst
    )
    sum_reduction = (
        0.0 if submitted_sum == 0 else (submitted_sum - suggested_sum) / submitted_sum
    )
    return OptimizationResult(
        submitted=submitted,
        suggested=suggested,
        submitted_cost=submitted_cost,
        suggested_cost=suggested_cost,
        submitted_transition_sum=submitted_sum,
        suggested_transition_sum=suggested_sum,
        submitted_worst_transition=submitted_worst,
        suggested_worst_transition=suggested_worst,
        # signed: the objective trades total against worst edge, so one metric can get worse
        relative_reduction=float(reduction),
        transition_sum_reduction=float(sum_reduction),
        worst_transition_reduction=float(worst_reduction),
        worst_transition_weight=worst_transition_weight,
        method=method,
    )


def greedy_sequence(
    recipe_ids: list[int],
    matrix: TransitionMatrix,
    *,
    worst_transition_weight: float = 0.0,
) -> OptimizationResult:
    """Nearest-next baseline; duplicate recipes remain distinct batch nodes."""
    submitted = make_batches(recipe_ids)
    _validate(submitted, matrix)
    selected = [submitted[0]]
    remaining = list(submitted[1:])
    while remaining:
        current = selected[-1]
        next_batch = min(
            remaining,
            key=lambda batch: (
                matrix.value(current.recipe_id, batch.recipe_id),
                batch.submitted_position,
            ),
        )
        selected.append(next_batch)
        remaining.remove(next_batch)
    return _result(
        submitted,
        tuple(selected),
        matrix,
        "greedy",
        worst_transition_weight,
    )


def _held_karp_path(
    submitted: tuple[Batch, ...],
    matrix: TransitionMatrix,
    *,
    max_edge: float | None = None,
) -> tuple[int, ...] | None:
    """Minimum-sum fixed-start path, optionally restricted by edge risk."""
    if len(submitted) <= 1:
        return (0,)

    remaining_count = len(submitted) - 1
    dp: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    for offset in range(remaining_count):
        index = offset + 1
        bit = 1 << offset
        edge = matrix.value(submitted[0].recipe_id, submitted[index].recipe_id)
        if max_edge is not None and edge > max_edge:
            continue
        dp[(bit, index)] = (
            edge,
            (0, index),
        )

    for mask in range(1, 1 << remaining_count):
        for last in range(1, len(submitted)):
            state = dp.get((mask, last))
            if state is None:
                continue
            cost, path = state
            for next_index in range(1, len(submitted)):
                next_bit = 1 << (next_index - 1)
                if mask & next_bit:
                    continue
                edge = matrix.value(
                    submitted[last].recipe_id, submitted[next_index].recipe_id
                )
                if max_edge is not None and edge > max_edge:
                    continue
                candidate = (
                    cost + edge,
                    path + (next_index,),
                )
                key = (mask | next_bit, next_index)
                incumbent = dp.get(key)
                if (
                    incumbent is None
                    or candidate[0] < incumbent[0]
                    or (
                        isclose(candidate[0], incumbent[0], abs_tol=1e-12)
                        and candidate[1] < incumbent[1]
                    )
                ):
                    dp[key] = candidate

    full_mask = (1 << remaining_count) - 1
    complete = [value for (mask, _), value in dp.items() if mask == full_mask]
    if not complete:
        return None
    return min(
        complete,
        key=lambda value: (value[0], value[1]),
    )[1]


def optimize_sequence(
    recipe_ids: list[int],
    matrix: TransitionMatrix,
    *,
    worst_transition_weight: float = 0.0,
) -> OptimizationResult:
    """Exact fixed-start path for total risk plus an optional worst-edge penalty."""
    if worst_transition_weight < 0:
        raise ValueError("worst_transition_weight must be nonnegative")
    submitted = make_batches(recipe_ids)
    _validate(submitted, matrix)
    if len(submitted) <= 1:
        return _result(
            submitted,
            submitted,
            matrix,
            "exact-held-karp",
            worst_transition_weight,
        )

    if worst_transition_weight == 0:
        path = _held_karp_path(submitted, matrix)
        assert path is not None
        suggested = tuple(submitted[index] for index in path)
        return _result(submitted, suggested, matrix, "exact-held-karp", 0.0)

    # For every possible maximum edge T, find the minimum-sum path whose
    # transitions are all <= T. The global minimax-weighted optimum must be
    # optimal in one of these thresholded subproblems.
    thresholds = sorted(
        {
            matrix.value(previous.recipe_id, target.recipe_id)
            for previous in submitted
            for target in submitted
            if previous.batch_id != target.batch_id
        }
    )
    candidates: dict[tuple[int, ...], tuple[float, float]] = {}
    for threshold in thresholds:
        path = _held_karp_path(submitted, matrix, max_edge=threshold)
        if path is None or path in candidates:
            continue
        order = tuple(submitted[index] for index in path)
        transition_sum = sequence_cost((batch.recipe_id for batch in order), matrix)
        worst = max(
            matrix.value(previous.recipe_id, target.recipe_id)
            for previous, target in pairwise(order)
        )
        candidates[path] = (
            transition_sum + worst_transition_weight * worst,
            transition_sum,
        )
    best_path = min(
        candidates,
        key=lambda path: (candidates[path][0], candidates[path][1], path),
    )
    suggested = tuple(submitted[index] for index in best_path)
    return _result(
        submitted,
        suggested,
        matrix,
        "exact-held-karp-minimax",
        worst_transition_weight,
    )
