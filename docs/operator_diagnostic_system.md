# Operator diagnostic notebook

Open [`notebooks/operator_diagnostic_system.ipynb`](../notebooks/operator_diagnostic_system.ipynb)
on `model_implementation`. Install its optional dependencies with:

```bash
uv sync --locked --group analysis
```

Select the repository's `.venv` Python kernel and run all cells. A CLI runs the
same analysis:

```bash
uv run --locked --group analysis python scripts/analyze_operator_diagnostics.py
# Optional alternate original tank image:
uv run --locked --group analysis python scripts/analyze_operator_diagnostics.py --tank-image /path/to/tank.jpg
```

The CLI exits 0 when vial measurements were produced, including partial analyses
with a missing tank image, and 2 when vial measurements are missing. Inspect the
summary's evidence/coverage fields; exit 0 is not a production-quality decision.

## Input and audit

The notebook reads every existing `.ipynb` source and records its imports, source
hash and lessons reused. The implementation reuses `model_workflow.run_workflow`
and the existing October 29 timestamped-still measurement, camera-clock audit,
dark-channel clipping, uneven-patch checks and blackout/operator-error labels.
It does not apply a fixed 30-second video offset. Measurement sources retain
timestamps and image hashes.

Expected collection directory: `data/10_29_26/vial_filled/`, including labels,
images and `image_index.csv`. The clock audit is
`analysis/10_29_26/camera_clock_audit.csv`.

The overhead photograph is expected at:

```
data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg
```

If it is missing, the notebook provides an image-upload widget. Upload the
original JPG/PNG, then rerun Section 3 onward. A picture visible in chat is not
necessarily available as a file to the execution environment. No generated
photograph, manually guessed RGB value or unrelated old tank image is used as
a fallback. The current cloud run lacks those original image bytes, so its
Approach A results explicitly remain missing.

Normalized ROIs are provided for the 4:3 overhead view supplied by the user.
Review the drawn liquid patches before interpreting measurements. Other camera
poses require different rectangles. T300 is excluded. A static top photograph
provides color signatures, not calibrated levels or tank concentration.

## Controlled initialization comparison

Approach A combines the measured pure-tank endpoints in linear RGB / effective
optical density relative to T100. Pure recipes reproduce their corresponding
tank RGB. Signed channel OD is retained for this experimental candidate rather
than distorting a pure endpoint with nonnegativity constraints. It has no learned
camera-transfer or vial-path calibration and is never independently promoted to
a chemical diagnosis or production release. Clipped tank channels are rejected.
Composition fractions are modeled; total-volume/path-length effects are not.

Approach B uses stable two-vial tails from matching recipe/composition runs in
the vial-camera domain. All observations from the evaluated run are excluded
from reference construction. Unlike the original analysis, there is no same-run
tail fallback. Source run and vial identities are exported in every score row.
Other runs can be later in the recording: this is retrospective evaluation.

Both methods retain the same 16/27 quality-eligible vial IDs and threshold
`delta_e < 2.2`. Their comparative metrics use only the intersection of IDs with
both scores. Missing observations and targets remain present with reasons.

First-changeover and late-run positions are **proxy groups**, not independently
certified contaminated and clean samples. Late-run membership is fixed by
sequence position, before inspecting its error; only training-tail candidates
are filtered for stable appearance. Metrics include coverage, first/late median
errors, separation and balanced proxy accuracy. A winner requires both proxy
groups in common coverage. A provisional B engine can still report evidence
without declaring a head-to-head winner. This is exploratory model selection,
not a nested or independent-session test. It cannot establish production accuracy.

## Attribution and evidence gates

`operator_diagnostics.diagnose_vials` emits one report row for every label row:

* **Measurement / sensor reject:** failed quality, stale/unverified clock,
  unavailable color or operator-error/blackout label. Color does not receive a
  numerical clean-reference score.
* **Changeover carryover suspected:** recipe change, distinguishable previous
  color endpoint, agreement with a Lab color segment toward that endpoint, and
  decay over at least three consecutive usable observations beginning with the
  first vial. Future observations make this retrospective. A trajectory weight
  is not an estimated chemical fraction.
* **Persistent tank / recipe / fill fault suspected:** a sustained nondecaying
  deviation, an unrequested in-camera signature, a verified tank-state alert, or
  independently calibrated abnormal measured fill. Possible hardware checks are
  suggested; no exact root cause or flush quantity is invented. Persistent
  carryover remains an alternative when clearance volume is unknown.
* **In-spec / clean production:** additionally requires validated independent
  clean references and camera evidence. Surrogate references cannot qualify.
* **Evidence pending:** absent targets or inconclusive causal evidence. An
  exploratory color match can coexist with pending production status.

Measured fill is accepted only from explicit `measured_fill_ml` and
`fill_calibration_verified` evidence. `requested_ml`, `dispensed_ml` when marked
`requested_proxy`, and image patch height are not measured volume. Likewise a
tank alert requires `tank_state_verified` and `tank_state_alert`; a static color
image cannot establish a tank/valve fault. October 29 supplies neither calibrated
fill evidence nor verified tank/valve telemetry.

An optional `expected_clearance_ml` must come from independently validated
experiments. The workflow never estimates it from the same evaluated run.

**No pre-fill subtraction is applied.** The engine does not read pre-fill Lab
values. For an outside-tolerance, rejected or reference-missing vial, an operator
suggestion can ask for visual inspection of the same physical vial before
dispensing on a repeat. That suggestion never modifies RGB, Lab, Delta E,
tolerance or attribution numerically. The older experimental notebook is
preserved as historical work, not imported into this workflow.

## Observed October 29 execution

The executed notebook reproduces 27 label rows, 26 sampled images and 16 usable
readings. It reports 11 sensor/label rejects, including vial 9's 6.79-second stale
frame and vial 10's unobservable blackout.

Whole-run holdout permits B scores for 10 vials: three exploratory matches and
seven outside-tolerance readings. Six usable vials have no qualifying reference
outside their run. There is no usable T200 target.

With the overhead image unavailable, A scores and common-coverage comparison
counts are zero. B is selected **provisionally**, not as a benchmark winner.
Within B's available coverage, the first-changeover proxy median is 15.09 Delta
E00 (n=2) and the late-run proxy median is 3.28 (n=4), a separation of 11.81. The
balanced proxy score is 0.625: both first proxies exceed tolerance, while only one
of the four late proxies is below tolerance. These tiny same-session proxy counts
must not be presented as general diagnostic accuracy.

Vial 13 has initial error 14.94, declining to 1.10 across four usable vials, but
its nearest previous-to-target Lab trajectory residual is about 2.73, exceeding
the 2.2 trajectory criterion. Vial 23 starts at 15.23, has a trajectory residual
about 3.31, and later rebounds away from the target. Thus neither is labeled
confirmed carryover. Their explanations state the decline/rebound and failed
trajectory evidence, and suggest review of vial appearance and process state.

All production decisions remain `PENDING`. No fitted mixture coefficients,
carryover tau, confidence intervals, chemical-purity claims or flush quantities
are fabricated. These results differ from in-sample tail scoring because the
reference holdout is intentionally stricter.

## Outputs and software integration

Each execution writes a fresh `outputs/operator_diagnostics/<run>/` directory:

* notebook audit, audited vial measurements and source metadata;
* tank RGB/ROI provenance or a missing-input reason;
* per-vial A/B scores, proxy metrics, baseline selection and training-tail IDs;
* operator report in CSV/JSON with explanation, action and evidence fields;
* tank/vial ROI verification, A/B curves and all-27-vial timeline;
* standalone `operator_dashboard.html` with attribution-state filtering;
* original camera/clean-model/carryover evidence artifacts.

The legacy `measure.vial`, `mixing.predict`, `advise.check` and `advise.advise`
adapters now perform the supported calculations rather than bare stubs.
Measurement returns color with NaN for unavailable vial fill. Checks compare to
clean color, not a contaminated forecast, and abstain when fill is unmeasured.
Mixing requires an explicit validated vial-camera model/calibration and abstains
for unsupported composition changes; the conditional recovery fit cannot predict
an unseen first vial. Advice recommends investigation without inventing an
automatic flush or optimized queue. Existing API callers continue to return
pending until the required scientific evidence is available. Shared contracts
and their legacy thresholds are unchanged; the notebook explicitly uses 2.2.
