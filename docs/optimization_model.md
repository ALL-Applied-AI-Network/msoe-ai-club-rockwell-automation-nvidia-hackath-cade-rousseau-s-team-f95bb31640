# Optimization model

Changeover Guard ranks production sequences with a physics-informed empirical appearance model. It
does not claim a physically measured contamination percentage or waste reduction.

## Pipeline

```text
recipe composition
  -> residual-source hypotheses (bulk, last fluid, hybrid)
  -> empirical RGB appearance
  -> first three vial color deviations
  -> directed transition cost and model disagreement
  -> exact fixed-start sequence optimization
```

The implementation lives in:

- `appearance.py`: recipe fractions, residual composition and the committed Beer-Lambert-style RGB fit.
- `carryover.py`: forward color prediction, decay and grid-based inverse estimation.
- `features.py`: calibrated vial ROI RGB/Lab/HSV features.
- `costs.py`: directed risk-adjusted transition matrix.
- `optimizer.py`: duplicate-safe greedy and exact Held-Karp solvers.

`mixing.py` remains a black-box teammate-owned integration point and is not modified by the scheduler.

## Cost assumptions

The baseline assumes 2 mL residual at a changeover, a per-vial decay of 0.25, and evaluates the first
three vials with a 0.7 discount. It evaluates bulk, last-fluid and 50/50 hybrid residual hypotheses.

For transition `i -> j`:

```text
transition risk = mean discounted Delta E00 across the residual hypotheses
uncertainty = standard deviation of those hypothesis scores
transition cost = transition risk + uncertainty
schedule objective = sum of transition costs + 0.5 * worst transition cost
```

These parameters are centralized in `TransitionCostConfig`. They are provisional until CSI supplies
known-ratio mixtures, repeated transitions and measured flush outcomes.

## Queue behavior

- The first submitted batch is fixed.
- Duplicate recipe IDs are distinct batches with positional IDs such as `batch-1` and `batch-3`.
- Pairwise cost is reused for batches with the same recipe pair.
- A same-recipe transition costs zero.
- Exact optimization uses Held-Karp dynamic programming, `O(n^2 2^n)`.
- Greedy nearest-next remains available as a baseline.

`POST /api/schedule` accepts the existing shape:

```json
{"recipe_ids": [5, 7, 5, 12]}
```

The response includes submitted and suggested recipe orders, distinct batch IDs, per-transition color
and cost predictions, total scores, the fixed starting recipe and predicted relative score reduction.

## Validation

Run:

```bash
uv run python scripts/evaluate_optimization.py
```

Current baseline results:

| Evaluation | Result |
|---|---:|
| Synthetic contaminated vials | 121 |
| Manifest-color fraction MAE | 0.0029 |
| Manifest-color median absolute error | 0.0014 |
| Rendered-image fraction MAE | 0.0481 |
| Rendered-image median absolute error | 0.0323 |
| Rendered-image median fit error | 1.27 Delta E00 |
| Run 5 transition-sum reduction | 3.61% |
| Run 5 total-plus-worst objective reduction | 3.39% |
| Run 5 worst-transition reduction | 1.31% |
| Run 5 exploratory predicted/observed Spearman correlation | 0.40 |

Manifest recovery validates the forward/inverse software because the manifest was generated from the
same empirical model. Rendered-image recovery additionally tests lighting, glass and twin materials.
Neither is physical carryover validation.

For Run 5, the observed order is:

```text
5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14
```

With recipe 5 fixed first, the current baseline recommends:

```text
5 -> 7 -> 14 -> 12 -> 6 -> 11 -> 13 -> 8 -> 9 -> 10
```

This is a model-relative recommendation under the assumed 2 mL residual. It must not be described as
measured waste or scrap reduction.

## Sensitivity analysis

The evaluator tests 36 hypothetical physical scenarios:

```text
holdup: 1, 2, 4 or 6 mL
decay: fast (0.10), medium (0.40) or slow (0.70)
residual source: bulk, last fluid or 50/50 hybrid
```

Current results:

| Metric | Result |
|---|---:|
| Scenarios with a lower-risk schedule | 26 / 36 |
| Median modeled improvement | 4.04% |
| Improvement range | 0–27.69% |
| Unique recommended orders | 7 |
| Most common order frequency | 12 / 36 |
| Scenarios avoiding original 6→7 transition | 26 / 36 (72.2%) |

There is no single stable schedule across all assumptions. Under several last-fluid-only scenarios,
the submitted order is already optimal or tied. This sensitivity is a useful result: physical holdup,
decay and residual-source measurements can materially change the production recommendation.

## Exploratory real Run 5 ranking

For each of the nine observed transitions, the first vial is compared with the median color of the
second and third vial in that recipe block. Predicted transition risk has a Spearman rank correlation
of `0.40` with this observed first-vial deviation.

This is weak-to-moderate exploratory agreement, not validation: every directed pair occurs once and
the later vials are assumed clean. It supports collecting repeated high/medium/low transitions before
real deployment rather than claiming that the current model ranks physical changeovers accurately.

## Data layout

The real CSI package is installed under ignored `data/`; the digital-twin renders are under ignored
`synthetic/`. Expected loader checks are:

```text
frames("data", 5)          = 845
vial_labels("data")        = 30
frames("synthetic", 100)   = 1848
vial_labels("synthetic")   = 462
```

The synthetic manifest records `carry_ml`. Its final-vial residual fraction is:

```text
carry_ml / (target recipe mL + carry_ml)
```

## Remaining calibration work

Replace provisional assumptions with known-ratio CSI mixtures, verified clean references, repeated
directed transitions, measured flush volume/time and operator acceptance labels. Until then, the API
warnings and UI label every score as model-derived and relative.
