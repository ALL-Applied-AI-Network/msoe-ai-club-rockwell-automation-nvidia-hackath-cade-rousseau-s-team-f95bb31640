# Evaluation

This folder measures the benefit of adding recipe context to color inspection.
It compares the legacy behavior described in the challenge with the strongest
comparison that the current repository can support honestly.

## Comparison implemented now

The two methods receive the same measured vial colors, clean references,
CIEDE2000 metric, threshold, and scored vials:

1. **Teledyne-style any-known-color emulation:** pass when the observed color
   resembles any clean color in the database, without considering which recipe
   was scheduled.
2. **Recipe-aware last-vial baseline:** pass only when the observed color
   resembles the clean reference for the intended recipe.

The word *emulation* matters: the dataset contains no exported Teledyne
settings or verdicts, so this evaluates the documented known-color decision
rule rather than claiming a hardware benchmark.

Video 5 contains 10 recipes with three vials each. The third vial supplies the
clean reference and is not scored. The second vial is the clean proxy. The
first vial after a known recipe change is the carryover-risk proxy. Both methods
are evaluated on the same subset, and unavailable measurements are reported as
exclusions instead of silently counted as passes or failures.

The primary operating point uses the shared
`changeover_guard.contracts.COLOR_WARN_DE` threshold. A threshold sweep at
ΔE00 2.3, 5, and 10 shows the detection/false-alarm trade-off so the result is
not based on a hand-picked threshold. The shared value is inherited from the
project; it was not fitted or independently validated on this benchmark.

## Current result

On the 17 cases that both methods could score, the checked-in run at ΔE00 10
found:

| Method | Carryover proxies detected | Clean-proxy false alarms | Balanced accuracy | Proxy-label AUC |
|---|---:|---:|---:|---:|
| Any known color | 0/8 | 0/9 | 50% | 0.44 |
| Intended recipe | 4/8 | 0/9 | 75% | 0.90 |

Recipe context therefore added **50 percentage points of carryover-proxy
recall** at the same observed false-alarm rate at the shared project threshold.
The threshold sweep in the generated report shows how that advantage and the
false-alarm trade-off change at stricter settings.

## Run it

From the repository root:

```powershell
uv run python eval/compare.py --refresh
uv run pytest eval/test_compare.py
```

The first command decodes the real video and takes roughly 15–30 seconds.
Later runs may omit `--refresh` to reuse the derived 30-row measurement cache.
The cache is reused only when its source-data hashes, measurement-code hashes,
and ROI/segmentation settings still match; otherwise it refreshes automatically.

Options:

```text
--data PATH       real dataset root (default: data/)
--output PATH     artifact directory (default: eval/results/)
--threshold N     primary ΔE00 threshold (default: shared project threshold)
--refresh         remeasure video 5 instead of using the cache
```

## Outputs

Everything stays under `eval/results/`:

| File | Purpose |
|---|---|
| `report.md` | Judge-readable result, comparison table, evidence, and caveats |
| `method_comparison.svg` | Side-by-side visual of recall, false alarms, and balanced accuracy |
| `per_vial.csv` | Paired old/new score and decision for every evaluated vial |
| `summary.csv` | Confusion counts and headline metrics at the primary threshold |
| `threshold_sensitivity.csv` | Both methods at every fixed reported threshold |
| `references.csv` | Third-vial reference status for every recipe |
| `excluded.csv` | Planned cases that could not be scored and why |
| `vial_measurements.csv` | Derived stable CIELAB measurement for each real vial |
| `vial_measurements.meta.json` | Source, code, settings, and cache-integrity fingerprints |

## What is not yet measured

The original evaluation plan also calls for leave-one-recipe-out performance,
first-vial prediction ΔE, explanation correctness, and an ordering-advice
comparison. Those results are deliberately marked pending because:

- `synthetic/` has no vial/carryover training data;
- `measure.vial()`, `mixing.predict()`, and the advice functions are still
  unimplemented;
- the real dataset has no independent defect, cause, explanation, or scheduling
  labels.

The project rule remains: train the final model on the digital twin and evaluate
it on real `data/`. The last-vial comparison here is an offline decision-logic
benchmark, not a substitute for that final evaluation.

## Interpretation

The valid claim is narrow but useful: on the same observed colors, requiring a
match to the intended recipe can flag carryover-risk cases that an “any known
color” rule accepts. It does not yet prove cross-run accuracy, digital-twin
generalization, cause-explanation correctness, or superiority to the physical
Teledyne system.

See [the generated comparison](results/report.md) for the current numbers.
