"""Photo initialization followed by CSI training/testing and operator diagnostics."""

import ast
import base64
import hashlib
import html
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from changeover_guard.diagnostic_baselines import (
    METHODS,
    compare_baselines,
    extract_tank_baselines,
)
from changeover_guard.model_workflow import default_config, run_workflow, write_json
from changeover_guard.operator_diagnostics import STATE_LABELS, diagnose_vials


def audit_notebooks(root: Path) -> pd.DataFrame:
    """Read every notebook source, not its base64 image outputs."""
    lessons = {
        "transition_analysis.ipynb": (
            "Reuse clock/hash audit, rejected rows, same-recipe runs, camera and model evidence gates.",
            "Hold out the whole evaluated run; never substitute the same run's tail for an independent target.",
        ),
        "paired_empty_vial_transition_analysis.ipynb": (
            "Reuse visual ROI/source review and explicit missing-reference reporting.",
            "No pre-fill subtraction. Use identical holdouts and proxy groups; inspect pre-fill only as operator follow-up.",
        ),
        "operator_diagnostic_system.ipynb": (
            "Use the October 29 photograph solely to initialize active-tank colors.",
            "Train/test on CSI whole-recipe holdouts; no October 29 vial fallback or pre-fill subtraction.",
        ),
    }
    rows = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        sources = ["".join(cell.get("source", [])) for cell in cells]
        imports = set()
        for cell, source in zip(cells, sources):
            if cell.get("cell_type") != "code":
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue  # notebooks may contain magics
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
                elif isinstance(node, ast.Import):
                    imports.update(name.name for name in node.names)
        kept, changed = lessons.get(
            path.name,
            (
                "Source inspected; imported modules recorded.",
                "No changes to this notebook.",
            ),
        )
        rows.append(
            {
                "notebook": str(path.relative_to(root)),
                "cells": len(cells),
                "code_cells": sum(c.get("cell_type") == "code" for c in cells),
                "source_sha256": hashlib.sha256(
                    "\n".join(sources).encode()
                ).hexdigest(),
                "imports": ", ".join(sorted(imports)),
                "kept": kept,
                "improved_or_not_used": changed,
            }
        )
    return pd.DataFrame(rows)


def plot_tank_roi(tank: dict, output: Path):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axis = plt.subplots(figsize=(10, 7))
    if tank["status"] != "measured":
        axis.text(
            0.5,
            0.5,
            "Approach A pending: original overhead JPG is missing.\n"
            "Save/upload the supplied image and rerun the analysis.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        image = cv2.imread(tank["path"])
        axis.imshow(image[:, :, ::-1])
        for tank_id, record in tank["tanks"].items():
            x1, y1, x2, y2 = record["roi_pixels"]
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2,
                )
            )
            axis.text(x1, y1 - 8, tank_id, color="yellow", backgroundcolor="black")
        axis.set_title(
            "Explicit tank liquid ROIs — review before interpreting Approach A; T300 excluded"
        )
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output / "tank_roi_verification.png", dpi=130)
    plt.close(fig)


def plot_benchmark(measurements, scores, output: Path, threshold=2.2):
    import matplotlib.pyplot as plt

    runs = list(measurements.groupby("run_id", sort=False))
    methods = list(scores.method.unique()) if not scores.empty else list(METHODS)
    fig, axes = plt.subplots(
        len(runs),
        len(methods),
        figsize=(6 * len(methods), max(5, len(runs) * 2.3)),
        squeeze=False,
        layout="constrained",
    )
    for row_index, (run_id, run) in enumerate(runs):
        for col, method in enumerate(methods):
            axis = axes[row_index, col]
            subset = scores[scores.run_id.eq(run_id) & scores.method.eq(method)]
            axis.plot(subset.vial, subset.delta_e, "o-", label=method)
            missing = subset[subset.delta_e.isna()]
            axis.scatter(
                missing.vial,
                np.zeros(len(missing)),
                marker="x",
                color="gray",
                label="not scored (reject / reference missing)",
            )
            if subset.delta_e.notna().sum() == 0:
                axis.text(
                    0.5,
                    0.55,
                    "No score: input / target unavailable",
                    ha="center",
                    transform=axis.transAxes,
                    fontsize=9,
                )
            axis.axhline(threshold, color="firebrick", linestyle="--")
            axis.set(
                title=f"{run_id} / recipe {int(run.recipe_id.iloc[0])}",
                ylabel="ΔE00 (exploratory)",
                xlabel="Vial ID",
            )
            axis.set_xticks(run.vial)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.legend(fontsize=7)
    holdout = "whole-recipe holdout" if "fold_id" in scores else "whole-run holdout"
    fig.suptitle(
        f"Baseline comparison — same usable rows, {holdout}, identical threshold"
    )
    fig.savefig(output / "baseline_comparison.png", dpi=120)
    plt.close(fig)


def plot_operator_timeline(report, output: Path):
    import matplotlib.pyplot as plt

    report = report.copy()
    report["plot_x"] = report.get("inspection_index", report.vial)
    colors = {
        "measurement_sensor_reject": "#b22c2c",
        "expected_changeover_carryover": "#b76e00",
        "persistent_tank_or_recipe_fault": "#7431a8",
        "in_spec_clean_production": "#168149",
        "evidence_pending": "#516473",
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(17, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2]},
        layout="constrained",
    )
    for index, (_, run) in enumerate(report.groupby("run_id", sort=False)):
        axes[0].plot(
            run.plot_x,
            run.delta_e,
            "o-",
            color="#226da2",
            label="ΔE to selected held-out reference" if index == 0 else None,
        )
    axes[0].axhline(
        report.threshold.iloc[0],
        linestyle="--",
        color="firebrick",
        label="Exploratory tolerance",
    )
    for state, subset in report.groupby("diagnostic_state"):
        scored = (
            subset
            if state == "measurement_sensor_reject"
            else subset[subset.delta_e.notna()]
        )
        axes[0].scatter(
            scored.plot_x,
            scored.delta_e.fillna(0),
            color=colors[state],
            marker="x" if state == "measurement_sensor_reject" else "o",
            s=65,
            label=STATE_LABELS[state],
        )
    no_reference = report[report.trusted_color & report.delta_e.isna()]
    axes[0].scatter(
        no_reference.plot_x,
        np.zeros(len(no_reference)),
        marker="s",
        facecolors="none",
        edgecolors="#516473",
        s=65,
        label="Reference unavailable (unscored)",
    )
    for _, run in report.groupby("run_id", sort=False):
        start = run.plot_x.min() - 0.5
        axes[0].axvline(start, color="gray", linewidth=0.7)
        axes[0].text(
            start + 0.15,
            1,
            f"R{int(run.recipe_id.iloc[0])}",
            transform=axes[0].get_xaxis_transform(),
            fontsize=9,
        )
    for _, row in report.iterrows():
        axes[1].bar(row.plot_x, 1, color=colors[row.diagnostic_state], alpha=0.18)
        short = {
            "measurement_sensor_reject": "SENSOR\nREJECT",
            "expected_changeover_carryover": "CARRYOVER\nSUSPECTED",
            "persistent_tank_or_recipe_fault": "SYSTEMIC\nSUSPECTED",
            "in_spec_clean_production": "IN SPEC",
            "evidence_pending": "PENDING",
        }[row.diagnostic_state]
        axes[1].text(
            row.plot_x, 0.5, short, rotation=90, ha="center", va="center", fontsize=7
        )
    axes[0].set_ylabel("ΔE00; x / empty square at zero = unscored")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].set(
        xticks=report.plot_x,
        yticks=[],
        xlabel="Inspection sequence — all label rows retained",
    )
    fig.suptitle(
        "Operator timeline — retrospective hypotheses; production remains PENDING without validation"
    )
    fig.savefig(output / "operator_timeline.png", dpi=130)
    plt.close(fig)


def dashboard_html(report, output: Path) -> str:
    """Standalone filterable report; all data text escaped, no external assets."""
    fields = [
        "inspection_index",
        "vial_key",
        "vial",
        "recipe_id",
        "diagnostic_label",
        "delta_e",
        "color_assessment",
        "fill_status",
        "production_status",
        "explanation",
        "recommended_action",
        "prefill_inspection_suggestion",
    ]
    labels = "".join(
        f'<option value="{html.escape(state)}">{html.escape(label)}</option>'
        for state, label in STATE_LABELS.items()
    )
    headers = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    rows = []
    for _, row in report.iterrows():
        cells = "".join(
            f"<td>{html.escape('' if pd.isna(row[field]) else str(row[field]))}</td>"
            for field in fields
        )
        rows.append(
            f'<tr data-state="{html.escape(row.diagnostic_state)}">{cells}</tr>'
        )
    image = output / "operator_timeline.png"
    timeline = (
        ""
        if not image.is_file()
        else (
            '<img style="max-width:100%" alt="All vial timeline" src="data:image/png;base64,'
            + base64.b64encode(image.read_bytes()).decode()
            + '">'
        )
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Changeover Guard diagnostics</title>
<style>body{{font:14px system-ui;margin:24px;color:#162936}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccd4d9;padding:8px;text-align:left;vertical-align:top}}th{{background:#e8eef2}}
select{{padding:8px;margin-bottom:15px}}</style></head><body>
<h1>Operator root-cause report</h1><p>Retrospective evidence and hypotheses. No automatic chemical release,
flush volume or recipe reorder. Pre-fill images are inspection suggestions only; no subtraction is applied.</p>
{timeline}<p>Filter attribution: <select id="diagnostic-state"><option value="">All vials</option>{labels}</select></p>
<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<script>document.getElementById('diagnostic-state').addEventListener('change',function(){{
document.querySelectorAll('tbody tr').forEach(row=>{{row.hidden=this.value!==''&&row.dataset.state!==this.value;}});
}});</script></body></html>"""


def run_oct29_operator_workflow(
    root=".", *, tank_image=None, tank_rois=None, plots=True
) -> dict:
    """Explicit historical October 29 audit, not the default CSI train/test flow."""
    root = Path(root).resolve()
    config = default_config(root)
    # This notebook targets October 29 specifically, not whichever dataset happens to exist.
    session = config["sessions"][0]
    camera = root / "data/10_29_26/vial_filled"
    session.update(
        session_id="10_29_26",
        recipe_csv=str(camera / "labels/vial_recipes.csv"),
        frame_csv=str(camera / "labels/vial_fill_frame_labels.csv"),
        source_mode="stills",
        clock_audit_csv=str(root / "analysis/10_29_26/camera_clock_audit.csv"),
    )
    config["output_dir"] = "outputs/operator_diagnostics"
    result = run_workflow(config, root, plots=plots)
    output = result["output_dir"]
    audit = audit_notebooks(root)
    audit.to_csv(output / "notebook_audit.csv", index=False)
    tank_path = (
        Path(tank_image)
        if tank_image
        else root / "data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg"
    )
    if not tank_path.is_absolute():
        tank_path = root / tank_path
    tank = extract_tank_baselines(tank_path, tank_rois)
    measurements = result["measurements"]
    if measurements.empty:
        benchmark = {
            "scores": pd.DataFrame(),
            "metrics": pd.DataFrame(),
            "tails": [],
            "selection": {
                "primary_method": None,
                "status": "missing_measurements",
                "threshold": 2.2,
            },
        }
        report = pd.DataFrame()
    else:
        benchmark = compare_baselines(measurements, tank, config["threshold"])
        report = diagnose_vials(
            measurements, benchmark, camera_proven=result["repeatability"]["proven"]
        )
        if plots:
            plot_benchmark(
                measurements, benchmark["scores"], output, config["threshold"]
            )
            plot_operator_timeline(report, output)
    if plots:
        plot_tank_roi(tank, output)
    for name, table in [
        ("baseline_scores", benchmark["scores"]),
        ("baseline_metrics", benchmark["metrics"]),
        ("operator_root_cause_report", report),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    tank_record = dict(tank)
    tank_record["path"] = (
        str(tank_path.relative_to(root))
        if tank_path.is_relative_to(root)
        else str(tank_path)
    )
    write_json(output / "tank_initialization.json", tank_record)
    write_json(output / "baseline_selection.json", benchmark["selection"])
    write_json(output / "baseline_tail_provenance.json", benchmark["tails"])
    write_json(output / "operator_root_cause_report.json", report.to_dict("records"))
    summary = {
        "status": "measured" if not measurements.empty else "missing_data",
        "vials": len(report),
        "quality_eligible_vials": int(report.trusted_color.sum()) if len(report) else 0,
        "diagnostic_states": report.diagnostic_state.value_counts().to_dict()
        if len(report)
        else {},
        "color_assessments": report.color_assessment.value_counts().to_dict()
        if len(report)
        else {},
        "tank_image_status": tank["status"],
        "selection": benchmark["selection"],
        "camera_repeatability_proven": result["repeatability"]["proven"],
        "measured_fill_available": bool(report.fill_status.ne("not_measured").any())
        if len(report)
        else False,
        "prefill_subtraction_applied": False,
        "root_causes_confirmed": 0,
        "production_decisions_enabled": False,
        "scope": "retrospective operator investigation, not live chemical quality certification",
        "unavailable_evidence": [
            "Independent certified clean controls",
            "Calibrated vial fill measurements",
            "Independent-session validation",
            "Tank level / valve telemetry",
        ],
    }
    write_json(output / "operator_diagnostic_summary.json", summary)
    if not report.empty:
        (output / "operator_dashboard.html").write_text(
            dashboard_html(report, output), encoding="utf-8"
        )
    return {
        "measurement_result": result,
        "audit": audit,
        "tank": tank,
        "benchmark": benchmark,
        "operator_report": report,
        "summary": summary,
        "output_dir": output,
    }


def plot_csi_sources(measurements: pd.DataFrame, output: Path):
    """Show the actual representative filled frames used for CSI measurements."""
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    examples = measurements[measurements.stable_t_measure.notna()]
    for page, offset in enumerate(range(0, len(examples), 16), 1):
        subset = examples.iloc[offset : offset + 16]
        fig, axes = plt.subplots(4, 4, figsize=(14, 11), layout="constrained")
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, row in zip(axes.flat, subset.itertuples()):
            cap = cv2.VideoCapture(str(row.source_video))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.source_video_frame_index))
                ok, frame = cap.read()
                decoded_index = round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                decoded_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                clock_ok = (
                    decoded_index == int(row.source_video_frame_index)
                    and abs(decoded_s - row.stable_decoded_t_measure)
                    <= 1 / row.source_fps + 1e-6
                )
            finally:
                cap.release()
            if not ok or not clock_ok:
                axis.text(
                    0.5, 0.5, "Source frame unavailable / decode mismatch", ha="center"
                )
                continue
            axis.imshow(frame[:, :, ::-1])
            box = (
                json.loads(row.roi_pixels)
                if isinstance(row.roi_pixels, str)
                else row.roi_pixels
            )
            x1, y1, x2, y2 = box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1.5,
                )
            )
            axis.set_title(
                f"{row.session_id} / vial {row.vial} / R{row.recipe_id}\n"
                f"FILLED t={row.stable_t_measure:.2f}s",
                fontsize=8,
            )
        fig.suptitle("CSI measured filled-vial frames — no pre-fill subtraction")
        fig.savefig(output / f"csi_vial_roi_{page:02d}.png", dpi=120)
        plt.close(fig)


def run_operator_workflow(
    root=".",
    *,
    tank_image=None,
    tank_rois=None,
    dataset_root=None,
    plots=True,
    sample_stride_s=1.0,
) -> dict:
    """Initialize from the October photo, train/test only on the CSI recordings.

    No October 29 vial labels or readings are accessed by this default workflow.
    Missing CSI data remains missing rather than invoking the historical audit.
    """
    from changeover_guard.csi_measurements import load_csi_measurements
    from changeover_guard.csi_training import train_test_csi

    root = Path(root).resolve()
    dataset = (
        Path(dataset_root)
        if dataset_root is not None
        else root / "data/csi-vial-fill-dataset"
    )
    dataset = dataset if dataset.is_absolute() else root / dataset
    tank_path = (
        Path(tank_image)
        if tank_image is not None
        else root / "data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg"
    )
    tank_path = tank_path if tank_path.is_absolute() else root / tank_path
    output = (
        root
        / "outputs/operator_diagnostics"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False)
    # Initialization is independent of—and precedes—reading CSI training labels.
    tank = extract_tank_baselines(tank_path, tank_rois)
    measured = load_csi_measurements(dataset, sample_stride_s=sample_stride_s)
    measurements = measured["measurements"]
    benchmark = train_test_csi(measurements, tank, threshold=2.2)
    report = (
        diagnose_vials(measurements, benchmark)
        if not measurements.empty
        else pd.DataFrame()
    )
    audit = audit_notebooks(root)
    for filename, table in [
        ("notebook_audit.csv", audit),
        ("csi_vial_measurements.csv", measurements),
        ("baseline_scores.csv", benchmark["scores"]),
        ("baseline_metrics.csv", benchmark["metrics"]),
        ("operator_root_cause_report.csv", report),
    ]:
        table.to_csv(output / filename, index=False)
    write_json(output / "csi_measurement_audit.json", measured["audit"])
    write_json(output / "training_folds.json", benchmark["folds"])
    benchmark["training_metrics"].to_csv(output / "training_metrics.csv", index=False)
    write_json(output / "csi_trained_model.json", benchmark["final_model"])
    write_json(output / "baseline_selection.json", benchmark["selection"])
    write_json(output / "operator_root_cause_report.json", report.to_dict("records"))
    tank_record = dict(tank)
    tank_record["path"] = (
        str(tank_path.relative_to(root))
        if tank_path.is_relative_to(root)
        else str(tank_path)
    )
    write_json(output / "tank_initialization.json", tank_record)
    decoded_recording = any(
        item.get("status") == "measured" for item in measured["audit"]
    )
    stage_status = (
        "missing_data"
        if measurements.empty or not decoded_recording
        else "partial_data"
        if measured["missing_inputs"]
        else "measured"
    )
    summary = {
        "status": stage_status,
        "tank_initialization_source": tank_record["path"],
        "tank_image_status": tank["status"],
        "training_and_test_dataset": str(dataset.relative_to(root))
        if dataset.is_relative_to(root)
        else str(dataset),
        "october29_vials_used_for_training_or_testing": False,
        "vials": len(measurements),
        "labeled_vials": len(measurements),
        "selection": benchmark["selection"],
        "quality_eligible_vials": int(report.trusted_color.sum()) if len(report) else 0,
        "diagnostic_states": report.diagnostic_state.value_counts().to_dict()
        if len(report)
        else {},
        "color_assessments": report.color_assessment.value_counts().to_dict()
        if len(report)
        else {},
        "missing_inputs": measured["missing_inputs"]
        + ([] if tank["status"] == "measured" else [tank_record["path"]]),
        "holdout": "leave one entire recipe out; every vial and session of the tested recipe excluded from training",
        "camera_repeatability_proven": False,
        "production_decisions_enabled": False,
        "prefill_subtraction_applied": False,
        "root_causes_confirmed": 0,
        "scope": "CSI held-out prediction errors and retrospective diagnostic hypotheses; late vials are clean proxies only",
    }
    write_json(output / "operator_diagnostic_summary.json", summary)
    write_json(
        output.parent / "latest.json",
        {"output_dir": str(output), "status": summary["status"]},
    )
    if plots:
        plot_tank_roi(tank, output)
        if not report.empty:
            plot_benchmark(measurements, benchmark["scores"], output)
            plot_operator_timeline(report, output)
            plot_csi_sources(measurements, output)
    if not report.empty:
        (output / "operator_dashboard.html").write_text(
            dashboard_html(report, output), encoding="utf-8"
        )
    return {
        "measurement_result": measured,
        "audit": audit,
        "tank": tank,
        "benchmark": benchmark,
        "operator_report": report,
        "summary": summary,
        "output_dir": output,
    }
