"""Photo initialization and supervised/blind CSI optical diagnostic workflow."""

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise

import cv2
import numpy as np

from changeover_guard.diagnostic_baselines import extract_tank_baselines
from changeover_guard.diagnostic_workflow import plot_tank_roi
from changeover_guard.model_workflow import write_json
from changeover_guard.optical_evaluation import (
    evaluate_labeled,
    evaluate_unlabeled,
    fit_height_calibration,
)
from changeover_guard.optical_sequence import (
    SequenceConfig,
    UnlabeledVideoSegmenter,
    camera_directory,
    load_labeled_sequences,
    portable_path,
)


def render_run_diagnostics(df_run, rois_dict, title, *, tolerance=2.2):
    """Every labeled/detected event occupies all three tracks, including rejects.

    With no events there is no timeline to draw; return None and let the caller
    display the explicit source audit. Never create a blank placeholder figure.
    """
    if df_run.empty:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    run = df_run.sort_values("sequence_index").copy()
    x = run.sequence_index.to_numpy(int)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.5, 1.2]},
        layout="constrained",
    )
    top, volumes, strip = axes
    de = run.delta_e.to_numpy(float)
    valid = np.isfinite(de) & run.Causal_State.ne("SENSOR_REJECT").to_numpy()
    maximum = max(5.0, float(np.max(de[valid])) if valid.any() else 0, tolerance * 2)
    rail = maximum * 1.22
    top.plot(x, np.where(valid, de, np.nan), color="0.4", lw=1, zorder=2)
    top.axhline(tolerance, color="black", ls="--", lw=1, label=f"ΔE00 = {tolerance}")
    styles = {
        "IN_SPEC_PASS": ("#16803b", "o"),
        "TRANSIENT_CARRYOVER": ("#db9b00", "^"),
        "SYSTEMIC_FAULT": ("#bf2537", "D"),
        "EVIDENCE_PENDING": ("#6c578d", "s"),
    }
    for state, (color, marker) in styles.items():
        mask = run.Causal_State.eq(state).to_numpy() & valid
        if mask.any():
            top.scatter(
                x[mask], de[mask], c=color, marker=marker, s=65, label=state, zorder=4
            )
    rejected = ~valid
    if rejected.any():
        top.scatter(
            x[rejected],
            np.full(rejected.sum(), rail),
            c="crimson",
            marker="x",
            s=160,
            linewidths=3,
            label="SENSOR REJECT / REFERENCE PENDING",
            zorder=5,
        )
        for i in np.flatnonzero(rejected):
            row = run.iloc[i]
            reason = (
                row.quality_details
                if row.Causal_State == "SENSOR_REJECT"
                else "Reference unavailable"
            )
            top.annotate(
                str(reason)[:100],
                (x[i], rail),
                xytext=(0, -9),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="center",
                fontsize=6,
                color="crimson",
            )
    top.set_ylim(-0.3, rail * 1.28)
    top.set_ylabel("ΔE00 to active recipe\n(raw filled-vial color)")
    top.set_title(
        title
        + "\nOptical diagnostics; causal labels are hypotheses, production status PENDING",
        fontsize=12,
    )
    top.grid(axis="y", alpha=0.2)
    top.legend(loc="upper left", fontsize=7, ncol=3)
    recipes = run.recipe_id.astype(str).to_numpy()
    boundaries = np.r_[0, np.flatnonzero(recipes[1:] != recipes[:-1]) + 1, len(run)]
    for block, (start, stop) in enumerate(pairwise(boundaries)):
        for axis in axes:
            axis.axvspan(
                x[start] - 0.5,
                x[stop - 1] + 0.5,
                color="steelblue" if block % 2 else "gold",
                alpha=0.07,
            )
        row = run.iloc[start]
        ratio = (
            "/".join(f"{row[c]:g}" for c in ("t100_ml", "t200_ml", "t400_ml"))
            if "t100_ml" in run
            else "inferred target"
        )
        top.text(
            (x[start] + x[stop - 1]) / 2,
            rail * 1.19,
            f"R{recipes[start]}: {ratio}",
            ha="center",
            va="center",
            fontsize=7,
        )
    # Draw decay arrows only for observed, consecutive decreases, never inferred gaps.
    for i in range(len(run) - 1):
        if (
            run.iloc[i].Causal_State == "TRANSIENT_CARRYOVER"
            and valid[i : i + 2].all()
            and recipes[i] == recipes[i + 1]
            and de[i + 1] < de[i]
        ):
            top.annotate(
                "",
                xy=(x[i + 1], de[i + 1]),
                xytext=(x[i], de[i]),
                arrowprops={"arrowstyle": "->", "color": "#db9b00"},
            )
    bottom = np.zeros(len(run))
    pred = run[["pred_f100", "pred_f200", "pred_f400"]].to_numpy(float)
    available = np.isfinite(pred).all(axis=1)
    colors = ["#39c5d8", "#f0b13a", "#ae4bb9"]
    for j, tank in enumerate(("T100", "T200", "T400")):
        height = np.where(available, pred[:, j], 0)
        volumes.bar(x, height, bottom=bottom, color=colors[j], width=0.66, label=tank)
        bottom += height
    for i, row in enumerate(run.to_dict("records")):
        if not available[i]:
            volumes.bar(
                x[i], 1, width=0.66, facecolor="none", edgecolor="crimson", hatch="///"
            )
            volumes.text(
                x[i], 0.5, "N/A", rotation=90, ha="center", va="center", fontsize=6
            )
        target = row.get("target_fractions")
        if target is None and all(k in row for k in ("f100", "f200", "f400")):
            target = [row[k] for k in ("f100", "f200", "f400")]
        if isinstance(target, (list, tuple, np.ndarray)) and np.isfinite(target).all():
            base = 0
            for fraction in target:
                volumes.bar(
                    x[i] + 0.35,
                    fraction,
                    bottom=base,
                    width=0.16,
                    facecolor="none",
                    edgecolor="black",
                    lw=0.7,
                )
                base += fraction
        thumbnail = rois_dict.get(row["vial_key"])
        if thumbnail is not None:
            strip.add_artist(
                AnnotationBbox(
                    OffsetImage(thumbnail, zoom=0.85),
                    (x[i], 0.5),
                    frameon=True,
                    bboxprops={"edgecolor": "crimson" if rejected[i] else "0.4"},
                )
            )
        else:
            strip.text(
                x[i],
                0.5,
                "No decoded\nframe",
                rotation=90,
                ha="center",
                va="center",
                fontsize=6,
                color="crimson",
            )
    volumes.set_ylim(0, 1.08)
    volumes.set_ylabel("Estimated fractions\noutline = target")
    volumes.legend(loc="upper left", fontsize=7, ncol=3)
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.set_ylabel("Actual RGB ROI\n(rejects retained)")
    strip.set_xlabel("Vial sequence index")
    strip.set_xticks(x)
    strip.set_xlim(x.min() - 0.65, x.max() + 0.65)
    return fig


def run_operator_workflow(
    root=".",
    *,
    tank_image=None,
    tank_rois=None,
    dataset_root=None,
    config=None,
    plots=True,
    include_unlabeled=True,
):
    """Specification workflow; the older whole-recipe audit remains separate."""
    root = portable_path(root)
    config = config or SequenceConfig()
    dataset = portable_path(dataset_root or "data/csi-vial-fill-dataset", root)
    tank_path = portable_path(
        tank_image or "data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg", root
    )
    output = (
        root
        / "outputs/operator_diagnostics"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    )
    output.mkdir(parents=True)
    tank = extract_tank_baselines(tank_path, tank_rois)
    measured = load_labeled_sequences(dataset, config)
    benchmark = evaluate_labeled(measured["measurements"], measured["frames"], config)
    height = fit_height_calibration(measured["measurements"])
    labeled = benchmark["results"]
    unlabeled, audits = {}, []
    all_rois = dict(measured["rois"])
    camera = camera_directory(dataset)
    known = {p.resolve() for p in measured["labeled_video_paths"]}
    videos = sorted(
        p
        for p in (camera / "videos").glob("*")
        if p.suffix.lower() == ".mp4" and p.resolve() not in known
    )
    if include_unlabeled:
        for path in videos:
            segmented = UnlabeledVideoSegmenter(config).segment(path)
            report = evaluate_unlabeled(segmented["events"], benchmark, height, config)
            unlabeled[path.stem] = report
            audits.append(segmented["audit"])
            all_rois.update(segmented["rois"])
            segmented["events"].to_csv(output / f"{path.stem}_events.csv", index=False)
            report.to_csv(output / f"{path.stem}_results.csv", index=False)
    for name, table in [
        ("csi_vial_measurements", measured["measurements"]),
        ("csi_frame_measurements", measured["frames"]),
        ("labeled_results", labeled),
        ("evaluation_metrics", benchmark["metrics"]),
        ("block_decay", benchmark["decay"]),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    benchmark["confusion"].to_csv(
        output / "recipe_confusion_matrix.csv", index_label="actual_recipe"
    )
    for name, data in [
        ("calibration_registry", benchmark["registry"]),
        ("training_folds", benchmark["folds"]),
        (
            "optical_knn",
            benchmark["model"].to_dict()
            if benchmark["model"] is not None
            else {"status": "unavailable"},
        ),
        ("height_calibration", height),
        ("tank_initialization", tank),
        ("csi_measurement_audit", measured["audit"]),
        ("unlabeled_video_audit", audits),
        ("sequence_config", asdict(config)),
    ]:
        write_json(output / f"{name}.json", data)
    if all_rois:
        roi_dir = output / "raw_rois"
        roi_dir.mkdir()
        for key, rgb in all_rois.items():
            cv2.imwrite(
                str(roi_dir / (key.replace("/", "_") + ".png")), rgb[:, :, ::-1]
            )
    if plots:
        import matplotlib.pyplot as plt

        if tank["status"] == "measured":
            plot_tank_roi(tank, output)
        if not labeled.empty:
            for video_id, group in labeled.groupby("video_id"):
                figure = render_run_diagnostics(
                    group,
                    all_rois,
                    f"CSI video {video_id} — vial-disjoint evaluation",
                    tolerance=config.tolerance,
                )
                figure.savefig(
                    output / f"labeled_video{video_id}_diagnostics.png", dpi=150
                )
                plt.close(figure)
        for name, report in unlabeled.items():
            figure = render_run_diagnostics(
                report,
                all_rois,
                f"{name} — inferred run targets",
                tolerance=config.tolerance,
            )
            if figure is not None:
                figure.savefig(output / f"{name}_diagnostics.png", dpi=150)
                plt.close(figure)
    valid_training = measured["frames"]
    training_count = (
        int((valid_training.usable & valid_training.vial_in_run.isin([2, 3])).sum())
        if len(valid_training)
        else 0
    )
    summary = {
        "status": "missing_data"
        if labeled.empty
        else "partial_data"
        if measured["missing_inputs"]
        else "evaluated",
        "labeled_vials": len(labeled),
        "training_instances": training_count,
        "requested_training_instances": 100 * len(measured["audit"]),
        "usable_vials": int(labeled.usable.sum()) if len(labeled) else 0,
        "training_status": "fitted"
        if benchmark["model"] is not None
        else "no_valid_training_frames",
        "unlabeled_videos_discovered": len(videos),
        "unlabeled_videos_processed": len(audits),
        "unlabeled_detected_vials": sum(len(table) for table in unlabeled.values()),
        "missing_inputs": measured["missing_inputs"]
        + ([] if tank["status"] == "measured" else [str(tank_path)]),
        "tank_image_status": tank["status"],
        "tank_initialization_role": "tank-camera endpoints; empirical vial k-NN learns only CSI frames",
        "evaluation_scope": "within-known-recipe settled LOOV and direct first-vial changeover tests; not unseen-recipe generalization",
        "prefill_subtraction_applied": False,
        "october29_vials_used_for_training_or_testing": False,
        "production_decisions_enabled": False,
        "root_causes_confirmed": 0,
        "runtime_target": "15s per unlabeled video; observed timings in audit, never truncate to claim compliance",
    }
    write_json(output / "operator_diagnostic_summary.json", summary)
    write_json(
        output.parent / "latest.json",
        {"output_dir": str(output), "status": summary["status"]},
    )
    return {
        "summary": summary,
        "output_dir": output,
        "tank": tank,
        "measurement_result": measured,
        "benchmark": benchmark,
        "operator_report": labeled,
        "df_labeled_results": labeled,
        "unlabeled_results": unlabeled,
        "unlabeled_audit": audits,
        "height_calibration": height,
        "rois": all_rois,
    }
