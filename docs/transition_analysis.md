# Vial color and recipe-transition analysis

The runnable workflow is in `notebooks/transition_analysis.ipynb`. It shares its
implementation with `scripts/analyze_transitions.py`; calculations do not live
only in notebook cells. It accepts both the original video-5 data and the new
`data/10_29_26/vial_filled/labels` collection. The newly committed CSVs were
audited: 27 vial labels, seven recipe runs, six changes and one recording
session. The subsequent image/video upload was analyzed using the aligned stills:
26 vials were sampled and 16 passed measurement quality checks. Camera controls
are not supplied, so color values and recovery thresholds remain exploratory.
See the [dataset assessment](dataset_assessment_10_29_26.md) and the committed
[`analysis/10_29_26`](../analysis/10_29_26/) snapshot for measured results.

## Run on your Windows checkout

From the repository root:

```powershell
uv sync --locked --group analysis
uv run --group analysis python scripts/analyze_transitions.py --write-config work/transition_config.json
```

The default configuration runs the committed collection without path edits.
The starter configuration detects `data/10_29_26/vial_filled/labels` and the files
`vial_recipes.csv` and `vial_fill_frame_labels.csv`. It also recognizes original
`data/vial_fill/labels/video5_vial_recipes.csv` and `video5_frame_labels.csv`.
If there is exactly one MP4 in the camera's `videos/` directory or camera folder,
it is selected automatically; otherwise the video path must be set explicitly.
All relative configuration paths resolve from the repository root. Use JSON
forward slashes for Windows absolute paths, or doubled backslashes.

```powershell
uv run --group analysis python scripts/analyze_transitions.py --config work/transition_config.json
```

Alternatively, open the notebook in VS Code/Jupyter, select the repository's
`.venv` kernel, edit the configuration cell, and run all cells. It loads
`work/transition_config.json` if present. Missing data is reported explicitly and
the notebook still completes. CLI exit codes are 0 for measured data, 2 for
invalid session inputs, and 3 for missing recordings/labels. Missing optional
controls do not prevent exploratory measurement.

`source_mode: auto` uses timestamped stills for the new wall-clock schema when
available. It uses the video for the original elapsed-second schema. Explicit
`stills` and `video` modes are also supported; old video-5 stills cannot silently
substitute for the known video clock. A wall-clock video-only source needs an
independently checked mapping (`video_clock_verified`), not an assumed equality.

Add one entry per **independent recording session**, with its own `session_id`,
`recipe_csv`, `frame_csv`, `video`, and optional `video_start`. Splitting one video
into many artificial sessions does not create independent experiments.

## Input schemas and alignment

Original recipes:

```csv
vial,recipe_id,t100_ml,t200_ml,t400_ml,start_s,end_s
```

New recipes (additional columns are preserved):

```csv
vial,recipe_id,type,t100_ml,t200_ml,t400_ml,total_ml,dispense_order,start_time,end_time,status,note
```

New frame labels include:

```csv
filename,timestamp,video_s,in_fill,vial,recipe_id,type,t100_ml,t200_ml,t400_ml,total_ml,dispense_order,color_purpose,status,note
```

`video_origin = timestamp - video_s`. All rows must agree within 0.1 seconds.
Wall-clock recipe times are converted to seconds relative to this origin;
midnight rollover is supported. With the supplied origin
`2026-10-30 02:02:55.390000`, the first vial's `2:04:48`–`2:05:17` window becomes
112.61–141.61 **wall-clock elapsed seconds**. These are not automatically MP4
playback seconds. Markdown `**` around pasted column names/cells and HTML
entities are normalized. Original elapsed-second labels do not need this origin.
If frame timing is unavailable, explicitly supply `video_start`; the workflow
never treats a wall-clock time as elapsed video seconds.

Blank component quantities mean an unused tank. Recipe totals, existing active
tank limits, unique vial identities, contiguous vial numbers, increasing time,
nonoverlapping windows and constant composition per recipe ID are checked.
An explicitly non-`ok` vial with missing times is retained as unobservable,
including vial 10's labeled blackout. It is never given an invented sampling
window. Operator errors also make requested throughput at and after that vial
unusable within the run; measured `dispensed_ml` is needed to recover that axis.
T300 remains out of service. Optional `dispensed_ml` provides measured total
throughput; otherwise volume axes are explicitly labeled requested-volume
proxies. Fractions still use recipe quantities; measured per-component delivery
accuracy needs a separate study. Optional `flush_ml` is recorded at recipe
boundaries; blank means **unknown**, not zero. A mid-run flush needs an explicit
history/session boundary. Flush/loading effects are not fitted from guesses.

For valid video clocks, full-vial frames are decoded at `end_s - 0.60`,
`end_s - 0.45`, and `end_s - 0.30`. Actual decoded times must match.
For the new timestamped stills, select the nearest same-vial/same-recipe image
to `end_s - 0.60`, within the last 2.5 seconds and before departure. Each still's
actual source time, filename, camera timestamp and SHA256 are retained.
One still provides no temporal repeatability estimate.

The committed collection also has a visually checked `clock_audit_csv` with
`filename,source_sha256,camera_timestamp`. It is automatically selected in the
starter config. Each entry must match the actual image hash; a burned-in camera
timestamp more than 0.5 seconds from the indexed time rejects that frame. This
catches a stale vial-9 image near the blackout. The audit and its clock contact
sheet are in `analysis/10_29_26`. New collections can supply a visually checked
audit in the same format; unverified source clocks cannot enable production or
qualify a carryover fit. Clock verification is separate from color repeatability.

This distinction matters in the committed data: the recordings last 1,956.3
container seconds, while the index spans 2,372 wall-clock seconds and the last
vial ends at 2,128.61. At playback second 140, the burned-in clock is already
02:05:45.608, rather than the label clock's 02:05:15.390. The workflow uses the
matching timestamped stills and refuses that uncorrected playback clock. It does
not use an estimated global scale/offset across missing camera intervals.
Review every recipe boundary in the saved alignment/ROI contact sheets before
setting `alignment_reviewed: true`. The old station calibration is only a
starting ROI, not proof that the new recording has the same camera geometry.

## Camera calibration and repeatability

The default liquid ROI is `[300,335,324,370]` at 640×480. Edit the `calibration`
configuration for the new camera. ROIs scale only with compatible aspect ratios
and must avoid the burned-in timestamp. RGB is decoded from BGR, shadows/glare
are excluded without rejecting clear liquid, and a robust median is converted
to CIELAB D65/2°. Spatial and short-window temporal spread are quality diagnostics,
not confidence intervals. Unusable rows remain in the sequence and break a
consecutive recovery streak. Non-`ok` recipe status is flagged for review.

Measure the same chart/controlled color references under locked settings, then
set `chart_csv` to:

```csv
patch_id,split,observed_r,observed_g,observed_b,reference_r,reference_g,reference_b
```

Use at least six training and six disjoint held-out patches, each split spanning
three color channels. `split` is `train` or `validation`. Reference colors must
be in the common sRGB domain, not packed OpenCV Lab. A 3×3 correction is fitted
in **linear** RGB and evaluated on the held-out patches using CIEDE2000. Its
fingerprint includes the chart, geometry, matrix, reference normalization and
review/settings configuration. A changed calibration invalidates old controls.
When `reference_roi` and `reference_rgb` are set, every frame must contain that
usable reference patch. Chart observations must undergo the same normalization
before fitting. A neutral patch corrects exposure/white balance approximately;
it cannot fix arbitrary changing camera processing.

After fitting the chart, use the emitted `vial_camera_calibration.json` and its
`calibration_id` for independently captured, corrected stable-vial measurements:

Set `calibration_file` to that JSON path on subsequent runs. Set the geometry,
review and camera-settings fields before collecting controls; changing them
changes the fingerprint and requires matching controls.

```csv
sample_id,session_id,capture_id,camera_id,calibration_id,r,g,b
```

Set `repeatability_csv` to that file. Each of at least five physical color
controls needs at least five distinct captures in each of two sessions. Do not
relabel adjacent production frames as independent stable controls. The worst
pairwise CIEDE2000 for each unchanged control must be **below 2.2/3 ≈ 0.733**,
as must the held-out chart p95 error. These conservative engineering budgets
reserve margin for a 2.2 decision; they are not a formal measurement-system
certification. The operator must also confirm alignment and lock settings or
use a reference patch. No single `enable_pass_fail` switch bypasses these checks.

## Clean targets and optical-density fit

Set `clean_controls_csv` to corrected measurements from independently verified
clean vial runs:

```csv
control_id,source_run_id,source_session_id,recipe_id,camera_id,calibration_id,is_clean_control,r,g,b,t100_ml,t200_ml,t400_ml
```

`is_clean_control` must be true. Source IDs must identify actual independent
clean runs, separate from the production runs being evaluated. Capture repeats
within one control run are aggregated before fitting so frames do not inflate
the sample size. Record at least two independently collected T100-only blanks,
and mixtures varying T200 and T400 fractions independently. Include enough
different mixtures to cover the expected recipe simplex. Set `baseline_id` to
the saved tank initialization ID for provenance when appropriate. Tank-camera
RGB/Lab is never used as an uncalibrated vial absorption coefficient.

The clean model fixes the measured T100 blank and fits nonnegative effective
absorbance vectors using NNLS:

```
A = -log(linear_rgb / measured_blank_linear_rgb)
A_clean = f200*K200 + f400*K400
predicted_sRGB = encode(measured_blank_linear_rgb * exp(-A_clean))
```

T100 has zero absorbance relative to its own blank. The fit uses recipe
fractions, not tank inventory or averages of Lab. Validation holds out **whole
recipes**, with the separately measured blank fixed. Predictions outside the
convex hull of measured fractions are marked unsupported. Coefficients,
calibration ID, baseline ID, training domain and validation errors are saved in
`clean_optical_model.json`; use `CleanOpticalModel.from_dict` and `.predict` to
reload it.

Without independent clean controls the workflow can compare to stable run
tails for exploration. A tail must contain at least two usable, stable vials.
Other-run tails are preferred; if only the same run is available, its target
vials are withheld from its curve. Every such target is labeled a surrogate,
never accepted as certified clean fluid and never used to fit the clean model
or authorize carryover fitting.

## Recovery, data sufficiency and carryover

A transition begins only when recipe identity changes within a recording.
Repeated vials of the same recipe retain their run and cumulative volume. The
initial run has unknown incoming history. Recovery requires **three consecutive
usable vials with CIEDE2000 strictly below 2.2**. Both the first vial in that
streak and its later confirmation point are reported, in vials, mL and seconds.
Missing observations, withheld targets and bad measurements break the streak.
If the session ends or the next recipe starts first, recovery is right-censored.
Censored transitions remain separate from completed numerical rankings; a
short observation window cannot be called a fast recovery.

The support report counts directed pairs, runs and independent collection
sessions. Minimum prerequisites for evaluating a **shared** decay are five
transitions from three sessions, each with four usable observations and an
independent clean reference. Three independently collected sessions of the same
directed pair are needed even to consider a pair-specific extension. These
counts are conservative identifiability checks, **not a power calculation or
proof of independence**. Adjacent frames do not increase these counts.

Only a shared single-exponential OD residual is currently implemented:

```
A_vial = A_clean + amplitude_transition * mean(exp(-V/tau), vial_volume_interval)
```

It integrates the residual over each collected vial, with a signed three-channel
OD vector per observed transition. `tau` is shared; no chemical residue
percentage is inferred. Whole sessions are held out. The first observed vial
initializes each held-out transition; subsequent vials are scored. Therefore
the model predicts **recovery after the first observation**, not contamination
in an unseen first vial. It must improve held-out OD error over both zero
carryover and constant-residual baselines by at least 10%, and its tau must not
sit at a search boundary. Otherwise parameters remain unsupported/null. Loading,
flush coefficients, adsorption and two-exponential effects are deliberately
left unestimated until independent experiments identify them.

After the camera gate passes, clean-model held-out error envelopes may be
reported. These are empirical validation errors, not calibrated confidence
intervals. A session bootstrap of tau requires both proven camera repeatability
and at least five independent sessions. Its 200 resamples keep whole sessions
together. Before these prerequisites, uncertainty fields remain null.

## Production decisions and integration boundary

`quality_decision` compares observed corrected color to **the clean optical
prediction**. It cannot compare against predicted contaminated fluid. It
requires the matching camera/calibration/threshold proof, at least five held-out
recipe folds from at least three control sessions, clean-model p95 error below
the same one-third-threshold budget, and a recipe inside the measured domain.
Only then does ΔE00 < 2.2 return `PASS`; equality or larger returns `FAIL`.
Otherwise it returns `PENDING`, with reasons and null uncertainty.

This workflow measures color. It does not invent a vial fill-volume sensor or
automatically promote an exploratory artifact into the existing operator API.
The existing `measure.vial`, `mixing.predict` and advice/UI contract remains
pending until calibrated volume measurement and validated deployment artifacts
are explicitly integrated. The notebook/CLI exposes the implemented measurement,
clean prediction, recovery fit and gated quality functions directly.

## Outputs

Each run gets a new directory under ignored `outputs/transition_analysis/`.
`latest.json` points to it. This prevents stale coefficients from appearing to
belong to a failed or partial run. Outputs include:

- `vial_color_measurements.csv`, including rejected rows, sample/decoded times,
  quality flags, calibrated RGB/Lab and gated production status.
- `recovery_curves.csv` and `transition_summary.csv`, including target sources,
  censoring, initial error and recovery/confirmation volume.
- `normalized_vial_labels.csv`, `recipe_run_inventory.csv` and
  `label_transition_inventory.csv`, available even without image/video bytes.
- `recipe_clean_targets.json` with independent-control/tail provenance.
- `clean_optical_model.json` and `carryover_model_parameters.json` with fit
  status, parameters or explicit nulls and grouped validation.
- Camera calibration, repeatability diagnostics, resolved configuration, input
  CSV hashes/video metadata and `analysis_report.json`.
- Full-vial ROI contact sheets, ΔE/vial/volume/Lab plots and a directed ranking
  heatmap retaining censored pairs.

Run `uv run --locked pytest` for software regression checks. Controlled-pixel,
generated-video and known-coefficient fixtures verify implementation behavior;
they provide no evidence about real-camera accuracy or physical recovery times.
