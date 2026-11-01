# Operator diagnostic notebook

Open [`notebooks/operator_diagnostic_system.ipynb`](../notebooks/operator_diagnostic_system.ipynb)
on `model_implementation`. The workflow uses two distinct sources:

| Purpose | Source |
| --- | --- |
| Initialize pure T100/T200/T400 tank colors | `data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg` |
| Train and test mixture-color predictions | `data/csi-vial-fill-dataset/vial_fill/labels/` and the matching CSI videos |

October 29 vial observations are **not** training or test examples in this
workflow. The overhead image supplies fixed initial tank colors; CSI vial
observations supply the camera-transfer training targets and held-out results.

## Run

```bash
uv sync --locked --group analysis
uv run --locked --group analysis python scripts/analyze_operator_diagnostics.py
```

Select the repository's `.venv` Python kernel to run the notebook. For another
local extraction path or original tank image:

```bash
uv run --locked --group analysis python scripts/analyze_operator_diagnostics.py --dataset-root /path/to/csi-vial-fill-dataset --tank-image /path/to/tank.jpg
```

`--dataset-root` accepts the CSI wrapper, its `vial_fill/` folder, or the exact
`vial_fill/labels/` folder supplied by the user. Labels provide recipe identities, requested
volumes and fill windows; images/video are still needed to measure actual vial
color. The loader discovers labeled CSI videos and records missing or invalid
inputs explicitly. It never falls back to October 29 vial data.

The notebook provides a tank-image upload control when the original JPG is not
at the configured path. Upload it and rerun the analysis cells. A photo visible
in chat is not automatically a local file. Review the displayed tank liquid
ROIs before interpreting measurements; other camera views need new rectangles.
T300 remains excluded from recipe prediction.

## Measurement and provenance

The workflow audits existing notebook sources and records the source files used.
CSI measurements come from the matching video within each vial's labeled fill
interval. Decode timestamps and stable-window quality are checked. Recipes,
run boundaries, rejected observations, sample windows and source video/ROI
identities remain available for inspection.

The October 29 clock mismatch is a lesson reused by the audit, not an offset
applied to CSI videos. The system does not assume that a brightness-defined
cycle boundary identifies the correct physical vial, or that still-image wall
timestamps can be used directly as video playback seconds.

## Tank initialization and CSI training

The initial optical candidate combines tank-camera colors in linear RGB /
effective optical density relative to T100. Pure recipes reproduce their tank
endpoint. Clipped tank channels are rejected. This is an initial color proxy;
the overhead camera and vial camera have different illumination and optical
paths. A static photograph does not measure tank concentration or liquid level.

The trained candidate starts with these tank-derived recipe predictions and
fits a fixed, regularized affine camera-transfer correction in linear RGB using
CSI training observations only. Training uses usable late-vial observations as
settled-color proxies and excludes the first vial after a recipe change. These
proxies are not independently certified clean chemistry.

The transfer fit requires sufficient distinct training recipes and an
identifiable design. Missing tank pixels, inadequate training coverage or
invalid measurements produce an explicit unavailable fit rather than invented
coefficients. Composition fractions are modeled; a separate total-volume /
optical-path-length effect is not established by this model.

## Training/test separation

Evaluation leaves one **whole recipe** out at a time:

1. All CSI observations belonging to the test recipe are excluded from that
   fold's fit. Recipe IDs with identical normalized tank fractions are held out
   together, including different-total-mL aliases, because the model uses ratios.
2. Only the remaining recipes' eligible training observations fit the transfer.
3. The frozen fold predicts every quality-eligible vial of the held-out recipe.
4. Fold artifacts retain train/test vial identities, training-target identities,
   fitted coefficients and training metrics separately from test metrics.

Adjacent frames are not independently split across training and test sets.
Each test vial's color is used to score its prediction, never to fit that
prediction or select a reference for itself. The fixed regularization is not
tuned against the held-out colors.

Training weights distinct fractional compositions equally. Predictions outside
the RGB gamut are withheld. Composition-hull extrapolations retain their test
errors and explicit domain flag but cannot authorize process-cause attribution.
An extrapolating previous-recipe endpoint cannot establish a carryover trajectory
either; both endpoints must have support in the same training fold.
An all-CSI final refit is saved separately as `csi_trained_model.json`; it is
never used to calculate held-out scores. Per-fold in-sample errors are exported
in `training_metrics.csv` separately from test metrics.

The benchmark compares raw tank-camera predictions with trained tank-initialized
predictions using identical test-vial coverage and `Delta E00 < 2.2`. Coverage and
missing-score reasons are reported before interpreting errors. First-changeover
and late-run positions are proxy groups for diagnostic separation, not known
contaminated/clean labels or a production-accuracy measurement.

An empirical per-recipe vial reference cannot predict an unseen held-out recipe
when all of that recipe's observations are excluded. Such Approach B references
remain unavailable in this experiment; the system does not use test vials to
fill them. The trained tank-initialized model is the primary exploratory engine
only when it has valid held-out predictions.

Recipe holdout tests generalization to another composition within the available
collection. It does not establish generalization to independent sessions, days,
cameras or lighting. Camera repeatability and independent clean references are
still required before production release.

## Operator diagnostics

Each labeled CSI vial receives a report row, including rejected or unscored
vials. The report distinguishes:

* **Measurement / sensor reject:** unreliable image, decode timing, clipping,
  missing stable color or other measurement-quality failure.
* **Changeover carryover suspected:** a recipe change, color trajectory toward
  the previous recipe, and a supported recovery trend. The previous and current
  predictions must use the same held-out fold; cross-fold camera predictions
  cannot establish a mixing trajectory.
* **Persistent tank / recipe / fill fault suspected:** persistent deviation,
  supported unrequested-color evidence, a verified tank alert or independently
  calibrated abnormal fill. These observations suggest investigation, not an
  exact chemical root cause or automatic flush volume.
* **In-spec / clean production:** requires validated reference and camera
  evidence in addition to a color match. Experimental fits cannot authorize it.
* **Evidence pending:** no valid target, insufficient training evidence or an
  inconclusive cause. An exploratory color match may still have pending
  production status.

The timeline and filterable HTML dashboard keep all labels visible. Missing
targets and sensor rejects do not appear as zero color error. Recovery
attribution may use later observations and is marked retrospective.

Requested mL is not measured fill. A fill fault needs independently calibrated
`measured_fill_ml` with `fill_calibration_verified`; a tank fault needs verified
state evidence. No tank level, valve malfunction, chemical purity, expected
clearance volume or confidence interval is inferred from the static photograph.

**No pre-fill subtraction is applied.** Outside-tolerance or inconclusive vials
can receive a suggestion to inspect/capture the same vial before dispensing on
a repeat. That suggestion does not alter measured RGB, Lab, Delta E or the
trained prediction. The older subtraction experiment is preserved as historical
work and is not used in this pipeline.

## Outputs and available results

Each run writes a fresh directory under `outputs/operator_diagnostics/` with:

* notebook/input audit and missing-input reasons;
* tank colors, ROIs and image provenance;
* measured CSI vials, quality flags and measurement provenance;
* held-out predictions, coverage and train/test metrics;
* per-fold training/test identities and fitted transfer parameters;
* operator explanations, suggested actions, plots and filterable HTML report.

The CSI dataset and original tank image are absent from the current cloud
checkout. Consequently, no numerical CSI training/test performance is claimed
from that run. Upload/save the original tank image and make the CSI directory
available locally to execute the experiment. Missing CSI files never trigger an
October 29 evaluation or reuse an older successful result.

## Historical October 29 analysis

The saved 27-vial October 29 assessment remains under `analysis/10_29_26/` and
`docs/dataset_assessment_10_29_26.md`. Explicit historical reproduction is
available through `run_oct29_operator_workflow`; it is not the default operator
training/test workflow. Its 16 usable readings, missing T200 targets and earlier
Delta E values must not be presented as results for the CSI dataset.

The legacy contract adapters continue to abstain when validated model, camera
or measured-fill evidence is unavailable. This exploratory notebook does not
automatically enable production decisions in the API. Shared contracts and
their legacy thresholds remain unchanged.
