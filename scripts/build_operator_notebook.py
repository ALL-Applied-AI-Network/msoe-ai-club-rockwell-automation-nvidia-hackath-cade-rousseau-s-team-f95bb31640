"""Build the operator notebook from reviewable source (requires analysis group)."""

from pathlib import Path

import nbformat as nbf


def build(root: Path):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook.cells = [
        md("""# Changeover Guard — operator diagnostic system

This notebook investigates all **27 October 29 vials**, retaining sensor rejects,
missing references and uncertain causes. It compares overhead tank initialization
with same-camera, held-out-run vial references, then explains the available
recipe-sequence, camera-quality and recovery evidence to an operator.

**Kept:** the existing timestamp/hash audit, the stale vial-9 check, dark-channel
and uneven-patch rejection, blackout labels, requested-volume provenance and
recipe-run tracking from `transition_analysis.ipynb` and its reusable modules.

**Refactored:** both baseline methods use the same usable cohort, the same
threshold and whole-run holdouts. Same-run tails cannot grade themselves. Missing
T200 targets stay missing. Empty score groups print coverage and reasons rather
than an unexplained blank chart. The earlier pre-fill comparison mixed reference
rules and could select the previous filled vial; **no pre-fill subtraction is
performed here**. Pre-fill inspection is suggested only as operator follow-up.

Attribution is **retrospective and exploratory**: later vials may support a
recovery trend, stable tails are surrogate clean references, and color agreement
cannot prove chemical purity. No camera repeatability or calibrated vial-fill
measurements have been supplied. Production release remains `PENDING`.

Run from the repository with `uv sync --locked --group analysis`, then select its
Python environment in Jupyter. All outputs go into a fresh ignored directory;
the previous notebooks and their saved analyses are preserved."""),
        code("""from pathlib import Path

import pandas as pd
from IPython.display import HTML, Image, display

from changeover_guard.diagnostic_baselines import OCT29_TANK_ROIS
from changeover_guard.diagnostic_workflow import audit_notebooks, run_operator_workflow

ROOT = Path.cwd().resolve()
if not (ROOT / 'pyproject.toml').is_file():
    ROOT = ROOT.parent
assert (ROOT / 'pyproject.toml').is_file(), 'Start in the repository or notebooks directory.'
pd.set_option('display.max_colwidth', 160)
TANK_IMAGE = ROOT / 'data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg'
# Normalized rectangles are specific to the 4:3 photo supplied for this collection.
# Adjust these if ROI verification shows glare, hardware or a different view.
TANK_ROIS = dict(OCT29_TANK_ROIS)
display(pd.DataFrame(TANK_ROIS, index=['left', 'top', 'right', 'bottom']).T)
"""),
        md("""## 1. Audit all existing notebooks

The table below reads each `.ipynb` source, records its imports and source hash,
and documents reused behavior and limitations. The detailed October 29 assessment
is `docs/dataset_assessment_10_29_26.md`.

Important clock lesson: the roughly 30-second discrepancy observed at playback
second 140 is **not assumed to be a constant offset**. Wall labels and MP4 seconds
are different domains here; source selection uses audited timestamped stills.
The legacy adapters now measure color, compare against clean color and provide
investigation advice, but abstain without validated model/fill evidence."""),
        code("""audit = audit_notebooks(ROOT)
display(audit[['notebook', 'cells', 'code_cells', 'kept', 'improved_or_not_used']])
"""),
        md("""## 2. Supply the original overhead tank image

If the JPG exists at the configured path, it is used directly. Otherwise upload
the original image with the control below, then rerun from Section 3. Images
visible in chat are not automatically copied into the execution filesystem.
No generated image or hand-estimated color is substituted for this photograph.

Physical vessel order is T100, T200, **T300 out of service**, T400. The explicit
liquid ROIs cover T100/T200/T400 only. This image supplies color signatures, not
calibrated tank concentration or liquid-level measurements."""),
        code("""if not TANK_IMAGE.is_file():
    import ipywidgets as widgets

    from changeover_guard.baseline import decode_image

    tank_upload = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False)
    upload_status = widgets.Output()

    def save_tank_upload(change):
        if not tank_upload.value:
            return
        upload = tank_upload.value[0]
        payload = bytes(upload['content'])
        with upload_status:
            upload_status.clear_output()
            try:
                image = decode_image(payload)
                TANK_IMAGE.parent.mkdir(parents=True, exist_ok=True)
                TANK_IMAGE.write_bytes(payload)
                print('Saved original tank image:', TANK_IMAGE.relative_to(ROOT))
                print('Image dimensions:', image.shape[1], 'x', image.shape[0])
                print('Rerun Section 3 onward to compare both baseline methods.')
            except ValueError as exc:
                print('Image rejected:', exc)

    tank_upload.observe(save_tank_upload, names='value')
    display(tank_upload, upload_status)
    print('Approach A stays pending until the original image is saved/uploaded.')
else:
    print('Using:', TANK_IMAGE.relative_to(ROOT))
"""),
        md("""## 3. Re-measure the October 29 sequence and review exact ROIs

The workflow runs from raw labels/images, using the established quality gates.
It does not reload a prior successful fit. In the supplied collection, 26 images
exist for 27 vials; 16 readings pass the original checks. Rejects remain in the
sequence and do not receive numerical color scores.

The following figures show the tank liquid patches and all selected vial source
frames. Clock ages, image hashes and rejection flags are exported for review."""),
        code("""result = run_operator_workflow(ROOT, tank_image=TANK_IMAGE, tank_rois=TANK_ROIS)
OUTPUT = result['output_dir']
measured = result['measurement_result']
measurements = measured['measurements']
display(pd.Series(result['summary']))
display(pd.DataFrame(measured['report'].get('label_audits', [])))
display(pd.Series(measured['report'].get('measurement_quality_flags', {}), name='flagged_vials'))
display(Image(filename=str(OUTPUT / 'tank_roi_verification.png'), width=950))
for contact in sorted(OUTPUT.glob('alignment_roi_*.png')):
    display(Image(filename=str(contact), width=1150))
if not measurements.empty:
    display(measurements[['vial', 'recipe_id', 'run_id', 'r', 'g', 'b', 'usable',
                          'source_clock_verified', 'source_timestamp_age_s',
                          'quality_flags', 'volume_source']])
"""),
        md("""## 4. Controlled A vs B initialization experiment

**A — overhead tank camera:** convert pure-tank RGB to linear RGB, compute signed
effective optical density relative to T100, and combine by recipe volume
fractions. Pure recipes reproduce their measured tank endpoint. This candidate
has **no learned camera-transfer or path-length calibration**. Total mL effects
are unestimated; tank RGB is not silently certified as expected vial RGB.

**B — vial camera:** select stable two-vial tails, then build the target from
matching recipe/composition tails in **other runs only**. No self-target fallback
is allowed. Future runs can supply references, so this is retrospective analysis.

Both use the same 16 quality-eligible vials and **ΔE00 < 2.2**. Comparative metrics
use the **intersection of scored vial IDs**. All rows, including unscored ones,
remain in `baseline_scores.csv` with reasons and reference identities.

First changeover vials and the last two vials of runs are **proxy groups**, not
independent clean/contaminated labels. Evaluation late-vial membership is fixed
by position, not by low measured error. Balanced proxy accuracy is the average
of first-vial alert rate and late-vial match rate; it is not production accuracy.
Selection requires both proxy groups in common coverage. Otherwise B can be
used provisionally, and the notebook explicitly declares no head-to-head winner."""),
        code("""benchmark = result['benchmark']
display(benchmark['metrics'])
display(pd.Series(benchmark['selection']))
if not benchmark['scores'].empty:
    display(benchmark['scores'][['vial', 'recipe_id', 'method', 'trusted',
        'evaluation_role', 'reference_available', 'delta_e', 'paired_comparison',
        'reference_runs', 'reference_reason']])
comparison_plot = OUTPUT / 'baseline_comparison.png'
if comparison_plot.is_file():
    display(Image(filename=str(comparison_plot), width=1150))
"""),
        md("""## 5. Operator diagnostic and attribution engine

The engine inspects every label row in sequence, using the selected baseline.

| State | Required observation and operator interpretation |
|---|---|
| Measurement / sensor reject | Clipping, stale/unverified timing, uneven patch, missing frame or operator-error label. Reacquire trustworthy evidence before judging mixture. |
| Changeover carryover suspected | A recipe change, an identifiable previous-to-current color trajectory, and decreasing error across at least three consecutive usable run observations. This uses later observations and is a hypothesis. |
| Persistent tank / recipe / fill fault suspected | Persistent observed deviation without decay, an unrequested in-camera tank signature, a verified tank-state alert, or independently calibrated abnormal fill. Inspect hardware and recipe; no automatic root-cause proof. |
| In-spec / clean production | Within tolerance **and** validated independent reference and camera evidence. An exploratory match alone cannot release production. |
| Evidence pending | Missing reference, ambiguous trajectory or incomplete calibration. The report records the available color assessment and an explicit next investigation. |

Requested recipe mL is not a fill measurement. A static tank photograph does not
prove valve malfunction or abnormal tank level. No expected clearance volume is
invented. A Lab trajectory weight is a color-coordinate parameter, not a residual
chemical fraction. No confidence interval is reported without camera evidence.

**Pre-fill policy:** raw filled-vial colors are used throughout. If a vial is
outside tolerance, rejected or lacks sufficient evidence, the operator may be
advised to inspect/capture the same vial before dispensing. There is no automatic
pre-fill subtraction, normalization or correction in this notebook."""),
        code("""operator_report = result['operator_report']
if operator_report.empty:
    print('No measurements: inspect the missing-input audit before running diagnostics.')
else:
    display(operator_report[['vial', 'recipe_id', 'diagnostic_label', 'delta_e',
        'color_assessment', 'fill_status', 'production_status', 'explanation',
        'recommended_action', 'prefill_inspection_suggestion']])
    display(operator_report[['vial', 'run_id', 'run_decay_supported',
        'trajectory_supported', 'trajectory_residual_delta_e', 'prior_color_weight',
        'trend_usable_prefix_vials', 'evidence']])
"""),
        md("""## 6. All-vial timeline and filterable operator dashboard

An x at zero identifies a sensor reject; an empty square marks a missing
reference. Neither means zero color error. Missing scores create gaps in the
ΔE curve. The state strip keeps every vial
visible, including the blackout. The exported HTML dashboard can be opened
without Jupyter and filters explanations by attribution state."""),
        code("""timeline = OUTPUT / 'operator_timeline.png'
if timeline.is_file():
    display(Image(filename=str(timeline), width=1250))
dashboard = OUTPUT / 'operator_dashboard.html'
if dashboard.is_file():
    display(HTML(dashboard.read_text(encoding='utf-8')))
"""),
        md("""## 7. Evidence gates, exports and interpretation

The original analysis found about **14.94 and 15.23 ΔE00** at the T400 → T100
starts, vials 13 and 23. A high initial difference and later recovery suggest a
process worth investigating; cause attribution still requires trajectory and
reference evidence. A censored run is not assumed to have cleared.

Missing T200 references, clipped channels, short follow-up and one recording
session limit conclusions. The original constrained clean model and carryover
artifacts remain `insufficient_data`. This diagnostic engine does not manufacture
their coefficients, mixture purity, measured fill, a flush recommendation or an
unseen first-vial forecast. Production remains pending until the evidence gates
are met. No pre-fill measurements are used in numerical scoring.

Adding the overhead image reruns A and the paired comparison; it cannot on its
own establish camera transfer accuracy or independent clean ground truth."""),
        code("""display(pd.Series(result['summary']))
display(pd.Series(measured['repeatability']))
display(pd.Series(measured['carryover']))
display(measured['summary'])
display(pd.DataFrame({'artifact': sorted(p.name for p in OUTPUT.iterdir() if p.is_file())}))
print('Exports:', OUTPUT.relative_to(ROOT))
assert result['summary']['prefill_subtraction_applied'] is False
assert result['summary']['root_causes_confirmed'] == 0
if not operator_report.empty:
    assert len(operator_report) == len(measurements)
    assert operator_report['vial_key'].is_unique
    assert operator_report['production_status'].eq('PENDING').all()
"""),
    ]
    output = root / "notebooks/operator_diagnostic_system.ipynb"
    nbf.write(notebook, output)
    return output


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1]))
