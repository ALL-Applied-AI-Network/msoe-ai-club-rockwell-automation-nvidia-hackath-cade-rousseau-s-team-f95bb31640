"""Reproducible analysis entry point shared by the notebook and CLI."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from changeover_guard.contracts import Recipe
from changeover_guard.optical_model import (
    fit_carryover_decay,
    fit_clean_optical,
    quality_decision,
)
from changeover_guard.transition_analysis import (
    ML_COLUMNS,
    label_inventory,
    load_session_labels,
    measure_timestamped_stills,
    measure_video,
    read_label_csv,
    recovery_curves,
    transition_support,
    validate_clean_controls,
    validate_frame_labels,
    validate_vials,
)
from changeover_guard.vial_color import (
    VialCalibration,
    fit_camera_chart,
    repeatability_report,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value):
    path.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _path(base: Path, value) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _find_file(directory: Path, candidates: list[str]) -> Path:
    return next(
        (directory / name for name in candidates if (directory / name).is_file()),
        directory / candidates[0],
    )


def default_config(root: str | Path = ".") -> dict:
    root = Path(root)
    directories = [
        root / "data" / "10_29_26" / "vial_filled" / "labels",
        root / "data" / "vial_filled" / "labels",
        root / "data" / "vial_fill" / "labels",
    ]
    labels = next((p for p in directories if p.is_dir()), directories[0])
    recipes = _find_file(
        labels, ["vial_recipes.csv", "video5_vial_recipes.csv", "vial_recipes"]
    )
    frames = _find_file(
        labels,
        [
            "vial_fill_frame_labels.csv",
            "video5_frame_labels.csv",
            "vial_fill_frame_labels",
        ],
    )
    videos = sorted(labels.parent.glob("videos/*.mp4")) + sorted(
        labels.parent.glob("*.mp4")
    )
    index_path = labels.parent / "image_index.csv"
    video_id = 5
    if index_path.is_file():
        ids = pd.read_csv(index_path, usecols=["video_id"]).video_id.dropna().unique()
        if len(ids) == 1:
            video_id = int(ids[0])
    video = (
        videos[0]
        if len(videos) == 1
        else labels.parent / "videos" / f"vial_fill_video_{video_id}.mp4"
    )
    return {
        "output_dir": "outputs/transition_analysis",
        "threshold": 2.2,
        "consecutive": 3,
        # The starter is editable; immutable saved calibration files retain
        # their fingerprint and reject edits that do not match it.
        "calibration": {
            k: v
            for k, v in VialCalibration().to_dict().items()
            if k != "calibration_id"
        },
        "calibration_file": None,
        "chart_csv": None,
        "repeatability_csv": None,
        "clean_controls_csv": None,
        "baseline_id": None,
        "sessions": [
            {
                "session_id": "10_29_26" if "10_29_26" in labels.parts else "video5",
                "recipe_csv": str(recipes.resolve()),
                "frame_csv": str(frames.resolve()),
                "video": str(video.resolve()),
                "video_start": None,
                "source_mode": "auto",
                "video_clock_verified": False,
                "clock_audit_csv": str(
                    (root / "analysis/10_29_26/camera_clock_audit.csv").resolve()
                )
                if (root / "analysis/10_29_26/camera_clock_audit.csv").is_file()
                else None,
            }
        ],
    }


def _provenance(paths: list[Path]) -> list[dict]:
    entries = []
    for path in dict.fromkeys(paths):
        if path is None or not path.is_file():
            continue
        item = {"path": str(path.resolve()), "bytes": path.stat().st_size}
        if path.suffix.lower() != ".mp4":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            item["modified_ns"] = path.stat().st_mtime_ns
        entries.append(item)
    return entries


def run_workflow(config: dict, base_dir: str | Path = ".", plots=True) -> dict:
    """Run on real inputs only; missing data yields a complete, explicit status.

    Every invocation writes a distinct run directory so failed or partial runs
    cannot reuse stale model artifacts. latest.json points to the current run.
    """
    base = Path(base_dir).resolve()
    output_root = _path(base, config.get("output_dir", "outputs/transition_analysis"))
    run_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = output_root / run_name
    output.mkdir(parents=True, exist_ok=False)
    calibration = VialCalibration.from_dict(config.get("calibration", {}))
    inputs, missing, errors, audits = [], [], [], []
    calibration_path = _path(base, config.get("calibration_file"))
    if calibration_path is not None:
        if not calibration_path.is_file():
            raise ValueError(
                f"configured camera calibration is missing: {calibration_path}"
            )
        calibration = VialCalibration.from_dict(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
        inputs.append(calibration_path)
    threshold = float(config.get("threshold", 2.2))
    consecutive = int(config.get("consecutive", 3))
    if not np.isfinite(threshold) or threshold <= 0 or consecutive <= 0:
        raise ValueError("threshold and consecutive must be positive")
    chart_path = _path(base, config.get("chart_csv"))
    if chart_path is not None:
        inputs.append(chart_path)
        if chart_path.is_file():
            calibration = fit_camera_chart(read_label_csv(chart_path), calibration)
        else:
            missing.append(str(chart_path))
    write_json(output / "vial_camera_calibration.json", calibration.to_dict())
    repeats = None
    repeat_path = _path(base, config.get("repeatability_csv"))
    if repeat_path is not None:
        inputs.append(repeat_path)
        if repeat_path.is_file():
            repeats = read_label_csv(repeat_path)
        else:
            missing.append(str(repeat_path))
    repeatability = repeatability_report(repeats, calibration, threshold)
    controls = None
    control_path = _path(base, config.get("clean_controls_csv"))
    if control_path is not None:
        inputs.append(control_path)
        if control_path.is_file():
            controls = validate_clean_controls(
                read_label_csv(control_path), calibration
            )
        else:
            missing.append(str(control_path))
    model, clean_status = (
        None,
        {
            "status": "insufficient_data",
            "reason": "Independent clean-vial controls are missing.",
            "model": None,
        },
    )
    if controls is not None:
        try:
            model = fit_clean_optical(controls, calibration, config.get("baseline_id"))
            if (
                repeatability["proven"]
                and model.validation["independent_recipe_folds"] >= 5
                and model.validation["control_sessions"] >= 3
            ):
                model.validation["uncertainty_delta_e"] = {
                    "held_out_prediction_error_p95": model.validation["p95_delta_e"],
                    "interpretation": "Empirical validation error envelope, not a calibrated confidence interval.",
                }
            clean_status = {
                "status": "fitted",
                "reason": None,
                "model": model.to_dict(),
            }
        except ValueError as exc:
            clean_status["reason"] = str(exc)
    measured = []
    labels_by_session = []
    seen = set()
    sessions = config.get("sessions", [])
    if not sessions:
        missing.append("No collection sessions configured.")
    for specification in sessions:
        session_id = str(specification["session_id"])
        if session_id in seen:
            raise ValueError("session_id must identify exactly one collection video")
        seen.add(session_id)
        recipe_path = _path(base, specification.get("recipe_csv"))
        frame_path = _path(base, specification.get("frame_csv"))
        video_path = _path(base, specification.get("video"))
        inputs.extend(
            [p for p in (recipe_path, frame_path, video_path) if p is not None]
        )
        if recipe_path is not None:
            inputs.append(recipe_path.parent.parent / "image_index.csv")
        if recipe_path is None or not recipe_path.is_file():
            missing.append(
                str(recipe_path)
                if recipe_path
                else f"{session_id}: recipe_csv not configured"
            )
            continue
        try:
            labels, timing = load_session_labels(
                recipe_path, frame_path, session_id, specification.get("video_start")
            )
            labels_by_session.append(labels)
            frame_audit = (
                validate_frame_labels(frame_path, labels)
                if frame_path is not None
                else {"status": "missing"}
            )
            source_audit = {
                "session_id": session_id,
                "timing": timing,
                "frame_labels": frame_audit,
            }
            audits.append(source_audit)
            source_mode = specification.get("source_mode", "auto")
            if source_mode not in ("auto", "stills", "video"):
                raise ValueError("source_mode must be auto, stills or video")
            if video_path is not None and video_path.is_file():
                cap = cv2.VideoCapture(str(video_path))
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
                finally:
                    cap.release()
                source_audit["video_container_duration_s"] = duration
                source_audit["last_labeled_end_s"] = float(labels.end_s.max())
                source_audit["label_clock_exceeds_container"] = bool(
                    labels.end_s.max() > duration
                )
            use_stills = source_mode == "stills" or (
                source_mode == "auto"
                and timing["schema"] == "wall_clock"
                and frame_audit.get("available_images", 0) > 0
            )
            contact_dir = output / "contact_frames" / f"session_{len(seen)}"
            if use_stills:
                if timing["schema"] != "wall_clock":
                    raise ValueError(
                        "stills require the explicit wall-clock/video_s alignment; legacy video5 stills are not interchangeable"
                    )
                source_audit["measurement_source"] = "timestamped_stills"
                clock_path = _path(base, specification.get("clock_audit_csv"))
                clock_audit = None
                if clock_path is not None:
                    inputs.append(clock_path)
                    if clock_path.is_file():
                        clock_audit = read_label_csv(clock_path)
                    else:
                        missing.append(str(clock_path))
                measured.append(
                    measure_timestamped_stills(
                        labels,
                        frame_path,
                        calibration,
                        contact_dir,
                        clock_audit=clock_audit,
                    )
                )
                continue
            if video_path is None or not video_path.is_file():
                missing.append(
                    str(video_path)
                    if video_path
                    else f"{session_id}: video not configured"
                )
                continue
            if source_audit.get("label_clock_exceeds_container"):
                raise ValueError(
                    "label clock exceeds video container time; use aligned timestamped stills or explicitly re-timed video labels"
                )
            if timing["schema"] == "wall_clock" and not specification.get(
                "video_clock_verified", False
            ):
                raise ValueError(
                    "wall-clock labels need a verified mapping to video container time; use timestamped stills or explicitly aligned elapsed-second labels"
                )
            source_audit["measurement_source"] = "video_container_time"
            measured.append(
                measure_video(
                    labels,
                    video_path,
                    calibration,
                    contact_dir,
                )
            )
        except (ValueError, OSError) as exc:
            errors.append({"session_id": session_id, "error": str(exc)})
    # Check recipe identity across sessions too; a reused ID with a new
    # composition cannot silently become a same-recipe continuation/target.
    if labels_by_session:
        all_labels = pd.concat(labels_by_session, ignore_index=True)
        validate_vials(all_labels)
        inventory, run_inventory, transition_inventory = label_inventory(all_labels)
    else:
        all_labels, run_inventory, transition_inventory = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        inventory = {
            "vials": 0,
            "sessions": 0,
            "recipe_runs": 0,
            "transitions": 0,
        }
    measurements = (
        pd.concat(measured, ignore_index=True) if measured else pd.DataFrame()
    )
    if controls is not None and not measurements.empty:
        production_runs = set(measurements.run_id)
        if controls.source_run_id.isin(production_runs).any():
            raise ValueError(
                "clean controls must come from separate, independently verified clean runs"
            )
    if not measurements.empty:
        measurements["production_status"] = "PENDING"
        measurements["production_delta_e_to_clean"] = np.nan
        measurements["prediction_error_p95_delta_e"] = np.nan
        if model is not None:
            for index, row in measurements[
                measurements.usable & measurements.source_clock_verified
            ].iterrows():
                recipe = Recipe(
                    int(row.recipe_id),
                    {c[:4].upper(): float(row[c]) for c in ML_COLUMNS if row[c] > 0},
                )
                decision = quality_decision(
                    row[["r", "g", "b"]].to_numpy(float),
                    recipe,
                    model,
                    calibration,
                    repeatability,
                    threshold,
                )
                measurements.loc[index, "production_status"] = decision["status"]
                measurements.loc[index, "production_delta_e_to_clean"] = decision[
                    "delta_e_to_clean"
                ]
                if decision["prediction_error_p95_delta_e"] is not None:
                    measurements.loc[index, "prediction_error_p95_delta_e"] = decision[
                        "prediction_error_p95_delta_e"
                    ]
    curves, summary, targets = recovery_curves(
        measurements, controls, threshold, consecutive
    )
    support = transition_support(
        summary if not summary.empty else transition_inventory, curves
    )
    decay = fit_carryover_decay(
        curves,
        support,
        model.blank_rgb if model is not None else [255, 255, 255],
        calibration=calibration,
        repeatability=repeatability,
        threshold=threshold,
    )
    decay.update(
        {
            "calibration_id": calibration.calibration_id,
            "blank_source": "measured_T100_controls"
            if model is not None
            else "ratio_cancels_reference",
        }
    )
    # OD(obs)-OD(target) cancels the blank exactly, so recovery fitting can
    # proceed with independent targets even if clean-mixture fitting is rank deficient.
    status = (
        "invalid_inputs"
        if errors
        else ("missing_data" if measurements.empty else "measured")
    )
    report = {
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output),
        "missing_inputs": missing,
        "errors": errors,
        "input_provenance": _provenance(inputs),
        "label_audits": audits,
        "measurement_rows": len(measurements),
        "measured_vials": 0
        if measurements.empty
        else int(measurements.sample_frames.gt(0).sum()),
        "label_inventory": inventory,
        "usable_vials": 0 if measurements.empty else int(measurements.usable.sum()),
        "source_clock_verified_vials": 0
        if measurements.empty
        else int(measurements.source_clock_verified.sum()),
        "measurement_quality_flags": {}
        if measurements.empty
        else measurements.quality_flags.str.split(";")
        .explode()
        .loc[lambda s: s.ne("")]
        .value_counts()
        .to_dict(),
        "camera_repeatability": repeatability,
        "transition_support": support,
        "clean_model_status": clean_status["status"],
        "clean_model_reason": clean_status["reason"],
        "carryover_status": decay["status"],
        "production_decisions_enabled": False
        if measurements.empty
        else bool(measurements.production_status.isin(["PASS", "FAIL"]).any()),
        "limitations": [
            "Observed curve thresholds are exploratory until camera and clean-model validation pass.",
            "Stable run tails are surrogate targets, never certified clean controls.",
            "Censored transitions are reported separately from completed recoveries.",
            "Requested mL are throughput proxies unless measured dispensed_ml is supplied.",
            "Tank baseline colors are not assumed equal to vial-camera colors.",
            "A single session cannot establish general carryover behavior.",
        ],
    }
    for filename, table, columns in [
        (
            "normalized_vial_labels.csv",
            all_labels,
            [
                "session_id",
                "vial",
                "recipe_id",
                "start_s",
                "end_s",
                "timing_observable",
            ],
        ),
        (
            "recipe_run_inventory.csv",
            run_inventory,
            ["run_id", "recipe_id", "labeled_vials", "timed_vials"],
        ),
        (
            "label_transition_inventory.csv",
            transition_inventory,
            ["transition_id", "previous_recipe_id", "next_recipe_id", "session_id"],
        ),
        (
            "vial_color_measurements.csv",
            measurements,
            ["vial_key", "recipe_id", "r", "g", "b", "usable", "production_status"],
        ),
        (
            "recovery_curves.csv",
            curves,
            [
                "transition_id",
                "vial_in_run",
                "cumulative_ml",
                "delta_e_to_clean",
                "target_source",
            ],
        ),
        (
            "transition_summary.csv",
            summary,
            [
                "transition_id",
                "right_censored",
                "vials_to_clear",
                "volume_to_clear_ml",
                "target_source",
            ],
        ),
    ]:
        (table if not table.empty else pd.DataFrame(columns=columns)).to_csv(
            output / filename, index=False
        )
    write_json(output / "recipe_clean_targets.json", targets)
    write_json(output / "clean_optical_model.json", clean_status)
    write_json(output / "carryover_model_parameters.json", decay)
    write_json(output / "camera_repeatability.json", repeatability)
    write_json(output / "analysis_report.json", report)
    write_json(output / "resolved_config.json", config)
    write_json(
        output_root / "latest.json", {"output_dir": str(output), "status": status}
    )
    if plots and not curves.empty:
        plot_recovery(curves, summary, output, threshold)
    if plots and not measurements.empty:
        make_contact_sheets(measurements, output)
    return {
        "report": report,
        "measurements": measurements,
        "label_runs": run_inventory,
        "label_transitions": transition_inventory,
        "curves": curves,
        "summary": summary,
        "targets": targets,
        "support": support,
        "clean_model": model,
        "carryover": decay,
        "calibration": calibration,
        "repeatability": repeatability,
        "output_dir": output,
    }


def plot_recovery(
    curves: pd.DataFrame, summary: pd.DataFrame, output: Path, threshold=2.2
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "recovery_plots"
    directory.mkdir(exist_ok=True)
    for number, (transition, rows) in enumerate(
        curves.groupby("transition_id", sort=False)
    ):
        volume_label = (
            "Cumulative requested volume (mL)"
            if rows.volume_source.eq("requested_proxy").all()
            else "Cumulative measured volume (mL)"
        )
        target_label = (
            "CIEDE2000 to surrogate target"
            if rows.target_source.str.contains("surrogate").any()
            else "CIEDE2000 to clean reference"
        )
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        axes[0, 0].plot(rows.vial_in_run, rows.delta_e_to_clean, "o-")
        axes[0, 0].set(xlabel="Vial since recipe change", ylabel=target_label)
        volume_axis = rows.cumulative_ml.where(rows.volume_usable)
        axes[0, 1].plot(volume_axis, rows.delta_e_to_clean, "o-")
        axes[0, 1].set(xlabel=volume_label, ylabel="CIEDE2000")
        for axis in axes[0]:
            axis.axhline(
                threshold,
                color="firebrick",
                linestyle="--",
                label=f"Exploratory {threshold:g} threshold",
            )
            axis.legend(fontsize=8)
        for c in ["L", "a", "lab_b"]:
            if c in rows:
                axes[1, 0].plot(volume_axis, rows[c], "o-", label=c)
        axes[1, 0].set(xlabel=volume_label, ylabel="CIELAB")
        axes[1, 0].legend()
        swatches = rows[["r", "g", "b"]].fillna(128).to_numpy() / 255
        axes[1, 1].imshow(
            swatches[None, :, :], aspect="auto", extent=(0.5, len(rows) + 0.5, 0, 1)
        )
        for i, usable in enumerate(rows.usable, 1):
            if not usable:
                axes[1, 1].text(i, 0.5, "X", color="red", ha="center")
        axes[1, 1].set(xlabel="Vial since change (X = unusable)", yticks=[])
        record = summary[summary.transition_id.eq(transition)].iloc[0]
        annotation = (
            "right-censored"
            if record.right_censored
            else f"recovered at vial {record.vials_to_clear:g}"
        )
        fig.suptitle(
            f"{record.previous_recipe_id} → {record.next_recipe_id}: {transition}\n{record.target_source}; {annotation}"
        )
        fig.savefig(directory / f"transition_{number + 1:03d}.png", dpi=150)
        plt.close(fig)
    recipes = sorted(set(summary.previous_recipe_id) | set(summary.next_recipe_id))
    matrix = np.full((len(recipes), len(recipes)), np.nan)
    labels = {}
    for pair, group in summary.groupby(["previous_recipe_id", "next_recipe_id"]):
        i, j = recipes.index(pair[0]), recipes.index(pair[1])
        if group.right_censored.any():
            labels[i, j] = "censored"
        elif group.volume_to_clear_ml.isna().any():
            labels[i, j] = "unknown\nvolume"
        else:
            matrix[i, j] = group.volume_to_clear_ml.median()
            labels[i, j] = f"{matrix[i, j]:g}"
    fig, axis = plt.subplots(
        figsize=(max(5, len(recipes) * 0.6), max(4, len(recipes) * 0.6))
    )
    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")
    im = axis.imshow(matrix, cmap=cmap)
    for (i, j), label in labels.items():
        axis.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="black" if label in ("censored", "unknown\nvolume") else "white",
        )
    axis.set(
        xticks=range(len(recipes)),
        xticklabels=recipes,
        yticks=range(len(recipes)),
        yticklabels=recipes,
        xlabel="Next recipe",
        ylabel="Previous recipe",
        title="Observed recovery volume (mL); censored pairs kept separate",
    )
    fig.colorbar(im, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "transition_ranking_heatmap.png", dpi=150)
    plt.close(fig)


def make_contact_sheets(measurements: pd.DataFrame, output: Path):
    from PIL import Image, ImageDraw

    for number, (_, rows) in enumerate(
        measurements.groupby("session_id", sort=False), 1
    ):
        paths = [Path(p) for p in rows.contact_frame.dropna()]
        for page in range(0, len(paths), 24):
            subset = paths[page : page + 24]
            if not subset:
                continue
            sheet = Image.new("RGB", (4 * 320, ((len(subset) + 3) // 4) * 260), "white")
            draw = ImageDraw.Draw(sheet)
            for index, path in enumerate(subset):
                with Image.open(path) as source:
                    source.thumbnail((320, 240))
                    x, y = (index % 4) * 320, (index // 4) * 260
                    sheet.paste(source, (x, y))
                    draw.text((x + 4, y + 242), path.stem, fill="black")
            sheet.save(output / f"alignment_roi_session_{number}_{page // 24 + 1}.png")
