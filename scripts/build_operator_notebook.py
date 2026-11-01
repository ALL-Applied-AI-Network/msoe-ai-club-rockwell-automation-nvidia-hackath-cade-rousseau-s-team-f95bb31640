"""Build the CSI optical k-NN and blind-sequence diagnostic notebook."""

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
        md("""# Changeover Guard — optical recipe learning and sequence diagnostics

The supplied tank photograph initializes T100/T200/T400 colors. Only CSI labels
and matching videos supply empirical vial-camera training data. October 29
production vials are never substituted.

Train on the second and third vials of each three-vial recipe block. Test settled
vials using **leave-one-vial-out**: exclude all five frames of the tested vial
from its scaler, model and reference. Directly test each block's first vial
against the settled calibration. This tests **known recipes**, not unseen recipes.

Rejected RGB values and raw patches remain visible. No pre-fill subtraction is
applied. Fixed 245 RGB normalization supplies an optical descriptor, not measured
empty-vial correction or calibrated chemical absorbance.

Install `uv sync --locked --group analysis`; select the repository `.venv` kernel."""),
        code("""from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from changeover_guard.diagnostic_baselines import OCT29_TANK_ROIS
from changeover_guard.optical_sequence import SequenceConfig
from changeover_guard.optical_workflow import run_operator_workflow

ROOT = Path.cwd().resolve()
if not (ROOT / 'pyproject.toml').is_file():
    ROOT = ROOT.parent
assert (ROOT / 'pyproject.toml').is_file(), 'Start in the repository or notebooks directory.'
TANK_IMAGE = ROOT / 'data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg'
CSI_DATASET = ROOT / 'data/csi-vial-fill-dataset/vial_fill/labels'
TANK_ROIS = dict(OCT29_TANK_ROIS)
# Review these rectangles against the actual camera image (640x480 reference).
CONFIG = SequenceConfig(liquid_column=(295, 270, 32, 127))
pd.set_option('display.max_colwidth', 160)
"""),
        md("""## 1. Initialize tank colors

Upload the original overhead JPG if absent. Tank endpoints are retained separately
from the vial-camera calibration registry: both camera views need not have equal
RGB. The k-NN fit uses CSI vial frames. T300 is excluded."""),
        code("""if not TANK_IMAGE.is_file():
    import ipywidgets as widgets

    from changeover_guard.baseline import decode_image

    tank_upload = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False)
    upload_status = widgets.Output()

    def save_tank_upload(change):
        if not tank_upload.value:
            return
        payload = bytes(tank_upload.value[0]['content'])
        with upload_status:
            upload_status.clear_output()
            try:
                decode_image(payload)
                TANK_IMAGE.parent.mkdir(parents=True, exist_ok=True)
                TANK_IMAGE.write_bytes(payload)
                print('Tank image saved. Rerun Section 2 onward.')
            except ValueError as exc:
                print('Image rejected:', exc)

    tank_upload.observe(save_tank_upload, names='value')
    display(tank_upload, upload_status)
else:
    print('Tank initialization:', TANK_IMAGE)
"""),
        md("""## 2. Sample, train, evaluate and detect

CSV inputs require vials 1–30, recipes 5–14 in three-vial blocks, component/total
volumes and explicit frame/second bounds. Blank component volumes become zero.
Five candidate frames span `end_frame-45` to `end_frame-10` inside each label.
Color crops exclude the top 20% and bottom 15% below a candidate meniscus.

Gates reject >15% channel pixels below 8 (or mean <5), >15% above 250, channel
spatial std >35, identical consecutive sampled pixels, or timestamp error >2s.
The identical-pixel gate is conservative and may flag a static scene. Rejected
RGBs remain observations; only accepted frames train. Missing pixels are never
imputed. There are **100 candidate training frames**, with actual counts below.

Every unmatched MP4 is sampled near 5 FPS. Detection requires an observed empty
scene, holder entry, optical change, a 1.5-second settled plateau and exit.
The audit measures runtime against the 15-second target without truncating videos."""),
        code("""result = run_operator_workflow(
    ROOT, tank_image=TANK_IMAGE, tank_rois=TANK_ROIS,
    dataset_root=CSI_DATASET, config=CONFIG,
)
OUTPUT = result['output_dir']
measured = result['measurement_result']
measurements = measured['measurements']
df_labeled_results = result['df_labeled_results']
display(pd.Series(result['summary']))
display(pd.DataFrame(measured['audit']))
for path in result['summary']['missing_inputs']:
    print('Missing input:', path)
tank_plot = OUTPUT / 'tank_roi_verification.png'
if tank_plot.is_file():
    display(Image(filename=str(tank_plot), width=950))
if not measurements.empty:
    display(measurements[['vial_key', 'recipe_id', 'r', 'g', 'b', 'usable',
        'ObservationState', 'quality_flags', 'quality_details', 'decoded_frames',
        'source_clock_verified']])
"""),
        md("""## 3. Vial-disjoint metrics and locked calibration

The vectorized k-NN uses standardized Lab + OD features, k=5 and inverse-distance
votes/volume fractions. Registry centroids give each contributing vial equal
weight. Settled test vials are excluded from their scaler, fit, centroid and OOD
threshold. The final model uses all accepted settled frames for blind videos only.

Accuracy and volume MAE report scored coverage. mL estimates multiply predicted
fractions by requested totals; they are **not measured delivery**. There are ten
first-vial samples but nine known recipe changes. True carryover recall needs
independent contamination labels; report a proxy detection rate and block decay
instead. Decay uses vial-disjoint references which may differ between folds.
Clipped recipes retain warnings and raw evidence, never untrusted training frames."""),
        code("""benchmark = result['benchmark']
display(benchmark['metrics'])
print('Recipe confusion matrix: actual rows, predicted columns; rejected vials counted in coverage.')
display(benchmark['confusion'])
display(benchmark['decay'])
display(pd.DataFrame.from_dict(benchmark['registry'], orient='index'))
display(pd.DataFrame([{
    'test_vial': f['test_vial_keys'][0], 'partition': f['partition'],
    'training_vials': len(f['train_vial_keys']), 'training_frames': len(f['train_frame_keys'])
} for f in benchmark['folds']]))
"""),
        md("""## 4. Labeled timeline — every vial remains visible

Top: ΔE00, recipe spans, tolerance 2.2, causal markers and annotated rejection rail.
Middle: estimated fractions and target outlines; hatching means unavailable.
Bottom: actual RGB crops, including rejects; missing decoded frames are labeled.

`IN_SPEC_PASS` means an optical match, with production still pending.
`TRANSIENT_CARRYOVER` uses the prior/current OD segment, weight 0.05–0.85 and
residual ≤ CONFIG.tau_mix. Its optical weight is not a measured residual-fluid
percentage. `SYSTEMIC_FAULT` is an investigation hypothesis. `EVIDENCE_PENDING`
covers missing calibration/inconclusive causes. Rejected frames get `SENSOR_REJECT`.
Pre-fill inspection is a follow-up suggestion only when a vial does not pass."""),
        code("""if not df_labeled_results.empty:
    display(df_labeled_results[['vial_key', 'recipe_id', 'predicted_recipe_id',
        'ObservationState', 'Causal_State', 'delta_e', 'alpha', 'mixing_residual',
        'operator_action', 'prefill_inspection_suggestion']])
    for path in sorted(OUTPUT.glob('labeled_video*_diagnostics.png')):
        display(Image(filename=str(path), width=1400))
else:
    print('No labeled sequence loaded. Resolve listed inputs; no empty chart is rendered.')
"""),
        md("""## 5. Blind video inference

Viterbi smoothing uses self-transition 0.70 and neutral emissions for rejects.
Targets are retrospective inferred recipes; the intended recipes remain unknown.
No fixed three-vial grouping is imposed on detections. Unsettled/truncated events
remain explicit rejects. Review source counts and detections before interpretation.

Height calibration needs three monotone observed height/requested-volume levels
and permits no extrapolation. Otherwise report fractions only. This unverified
proxy cannot activate a >10% verified-fill fault. A persistent valve error that
resembles another recipe needs independent intended-recipe evidence."""),
        code("""display(pd.DataFrame(result['unlabeled_audit']))
display(pd.Series(result['height_calibration']))
for video_name, report in result['unlabeled_results'].items():
    print(video_name, ':', len(report), 'detected vials')
    if report.empty:
        print('No events detected; inspect source audit and detector ROI settings.')
        continue
    display(report[['vial_idx', 'recipe_id', 'predicted_recipe_id', 'Causal_State',
        'delta_e', 'estimated_total_ml', 'volume_basis', 'operator_action']])
    display(Image(filename=str(OUTPUT / f'{video_name}_diagnostics.png'), width=1400))
"""),
        md("""## 6. Self-audit and exports

Frame/vial CSVs retain every candidate observation. JSON records include fold
memberships, scalers/models, registry, frame counts, configuration, timing and
height calibration. Raw ROI PNGs retain failed measurements. Absent real data
cannot establish model accuracy, detection coverage, physical causes or runtime."""),
        code("""assert result['summary']['prefill_subtraction_applied'] is False
assert result['summary']['october29_vials_used_for_training_or_testing'] is False
if not df_labeled_results.empty:
    if df_labeled_results['video_id'].nunique() == 1:
        assert len(df_labeled_results) == 30
    for _, sequence in df_labeled_results.groupby('video_id'):
        assert len(sequence) == 30
        assert sequence['sequence_index'].tolist() == list(range(1, 31))
    assert df_labeled_results['Causal_State'].isna().sum() == 0
    assert df_labeled_results['ObservationState'].isin(
        ['VALID', 'SENSOR_CLIPPED', 'STALE_FRAME', 'UNSETTLED_MOTION']
    ).all()
    assert df_labeled_results['production_status'].eq('PENDING').all()
    for fold in benchmark['folds']:
        assert not set(fold['train_vial_keys']) & set(fold['test_vial_keys'])
    print('All labeled rows and states retained; tested vials excluded from training.')
else:
    print('Dataset assertions not evaluated: labeled inputs absent.')
for report in result['unlabeled_results'].values():
    if not report.empty:
        assert report['Causal_State'].notna().all()
        assert report['sequence_index'].tolist() == list(range(1, len(report) + 1))
display(pd.DataFrame({'artifact': sorted(p.name for p in OUTPUT.iterdir())}))
print('Exports:', OUTPUT)
"""),
    ]
    output = root / "notebooks/operator_diagnostic_system.ipynb"
    nbf.write(notebook, output)
    return output


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1]))
