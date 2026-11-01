# Operator optical diagnostics

Open [`operator_diagnostic_system.ipynb`](../notebooks/operator_diagnostic_system.ipynb)
on `model_implementation`, or run:

```bash
uv sync --locked --group analysis
uv run --locked --group analysis python scripts/analyze_operator_diagnostics.py
```

The notebook/CLI use `changeover_guard.optical_workflow.run_operator_workflow`.
The previous whole-recipe transfer experiment remains accessible through
`diagnostic_workflow.run_operator_workflow`. Shared API contracts are unchanged.
The new experiment evaluates known recipes with different vials.

## Inputs

* Tank initialization: `data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg`.
  The original photograph initializes T100/T200/T400 endpoints, kept separately
  from empirical vial-camera calibration. A notebook upload control is provided.
* Train/test: `data/csi-vial-fill-dataset/vial_fill/labels/videoN_vial_recipes.csv`
  (also `vial_recipes.csv` for video 5), matched to `videos/vial_fill_video_N.mp4`.
* Blind inference: every unmatched MP4 in the same videos directory.

`--dataset-root` accepts wrapper/vial_fill/labels/videos paths with Windows or
POSIX separators. `--tank-image` overrides the photograph. Existence checks
precede VideoCapture; audits include total/decoded frame counts. October 29
production vials never substitute for CSI. T300 remains excluded.

Each CSV requires exactly vials 1–30, recipes 5–14 in contiguous three-vial
blocks, component volumes, total_ml, start_s/end_s and integer start_frame/end_frame.
Blank component volumes become zero. Nonnegative volumes, totals, intervals and
composition consistency are checked; frames and seconds must agree within 2s.

`SequenceConfig` exposes the liquid column/holder rectangles, reference geometry
and detector thresholds. Default liquid column `(295,270,32,127)` is x/y/w/h in
640×480 coordinates. Review against actual frames: rescaling cannot validate a
different pose. Sobel-Y finds candidate menisci; color crops exclude the top 20%
and bottom 15% of liquid height. Weak edges leave height unavailable and use the
configured column crop. Frame crop coordinates and raw thumbnails are exported.

## Model and evaluation

Five frames per vial span `[end_frame-45,end_frame-10]` within its labeled
interval. Twenty settled vials provide 100 candidate training frames; only valid
frames fit. Every vial and candidate frame remains in the output, including
missing-video rows. Rejected RGB values are retained; absent pixels stay missing.

Gates reject >15% channel pixels below 8 or channel mean below 5, >15% above 250,
spatial channel std >35, identical successive pixels, or timestamp error >2s.
The exact-zero difference gate is conservative and can flag a truly static scene.
Reasons and raw evidence are preserved for review.

Features concatenate Lab and `-log10(max(RGB,1)/245)`. This encoded-camera OD is
an optical descriptor, not calibrated chemical absorbance. No pre-fill subtraction
occurs. Pre-fill inspection is a follow-up suggestion only if a vial does not pass.

`VectorizedOpticalKNN` standardizes on training data, expands distances with matrix
operations and applies k=5 inverse-distance votes/fraction regression. Registry
centroids give each vial equal weight and retain calibration/clipping status.

Each settled test vial is excluded entirely from its scaler, model, registry and
training-derived OOD threshold. First vials never train. Fold artifacts record
exact training frames/vials, parameters and references. The final settled refit
is used for subsequent unlabeled videos only.

Accuracy/volume MAE include scored coverage and a confusion table. mL estimates
use requested totals and do not measure actual delivered volume. Ten first-vial
samples contain nine known changes; recipe position is not contamination truth.
True carryover recall is unavailable without independent labels. Instead report
transition proxy detection rate and three-vial decay, with differing LOOV
references disclosed. No numerical real-data performance is claimed here.

## Blind video processing

At approximately 5 FPS the state machine observes an absent-holder background,
holder entry, optical change, a low-motion plateau lasting 1.5s, and exit. It
tracks horizontal edges, chromatic variance and bottom-column OD. Starting
mid-fill without an empty scene is audited. Unsettled/truncated events remain
rejects. Sequential grabbing avoids repeated seeks. Runtime and whether the
15s target was met are reported; videos are never truncated to claim compliance.

Viterbi uses self-transition 0.70 and neutral emissions for rejected observations.
Targets are retrospective optical consensus, not known intended recipes. No
fixed three-vial blocks are imposed. A missing fit yields explicit pending states.
The height proxy requires three monotone observed requested-volume levels and
never extrapolates. Otherwise fractions remain available while mL stays missing.
This proxy cannot activate a verified >10% fill fault.

## Interpretation and rendering

Each row has VALID, SENSOR_CLIPPED, STALE_FRAME or UNSETTLED_MOTION observation
state. Missing source pixels use STALE_FRAME with explicit MISSING/DECODE reasons.
Causal states are SENSOR_REJECT, IN_SPEC_PASS, TRANSIENT_CARRYOVER, SYSTEMIC_FAULT
and EVIDENCE_PENDING for missing reference or inconclusive cause.

Carryover uses current/previous OD centroids, weight 0.05–0.85 and residual ≤0.08.
That weight is not measured residual-fluid percentage. Faults are investigation
hypotheses; in-spec means optical agreement. All production statuses remain
PENDING. Blind OOD queries cannot pass through nearest-recipe assignment alone.

The unified timeline preserves every index: ΔE/recipe spans/rejection rail,
fraction stacks/target outlines (unavailable predictions hatched), and actual RGB
patches (unavailable frames labeled). An empty sequence returns no figure; the
notebook shows the source audit instead of empty subplots.

## Evidence and limitations

Fresh `outputs/operator_diagnostics/<run>/` directories contain frame/vial CSVs,
metrics/confusion/decay, registry, fold parameters, final model, tank/height
initialization, detector events, source/runtime audits, raw ROI PNGs and timelines.
Notebook assertions verify 30 rows per labeled video, complete states and no
test-vial training overlap.

The real CSI files and original tank photo are absent from this cloud checkout.
Controlled fixtures validate software behavior, not real camera placement,
segmentation recall, recipe accuracy, runtime or chemical causes. Earlier
whole-recipe/October 29 analyses remain separate historical experiments.
