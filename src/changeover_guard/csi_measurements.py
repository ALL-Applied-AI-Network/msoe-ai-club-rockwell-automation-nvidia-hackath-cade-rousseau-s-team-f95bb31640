"""Measure labeled CSI fills without ordinal cycle matching or pre-fill subtraction.

Only frames inside a vial's own ``start_s <= video_time < end_s`` interval are
eligible. These are container/frame-index seconds, not October 29 wall-clock
seconds. A stable window cannot bridge a rejected, missing or noncontiguous
sample. Missing videos preserve label rows as rejects instead of substituting
the October 29 production collection.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from changeover_guard.transition_analysis import (
    ML_COLUMNS,
    read_label_csv,
    validate_vials,
)
from changeover_guard.vial_color import color_errors, lab_values
from color_monitor import config as monitor_config
from color_monitor.color import bgr_to_lab, crop, de00, measure_frame

MEASUREMENT_COLUMNS = [
    "session_id",
    "vial",
    "vial_key",
    "recipe_id",
    "run_id",
    "vial_in_run",
    "start_s",
    "end_s",
    *ML_COLUMNS,
    "requested_ml",
    "dispensed_ml",
    "cumulative_ml",
    "volume_start_ml",
    "volume_source",
    "volume_usable",
    "is_transition",
    "r",
    "g",
    "b",
    "L",
    "a",
    "lab_b",
    "usable",
    "source_clock_verified",
    "quality_flags",
    "stable_t_start",
    "stable_t_end",
    "stable_t_measure",
    "source_video",
    "source_image",
    "source_timestamp_age_s",
    "sample_frames",
    "measurement_source",
    "source_clock",
    "roi_pixels",
]


def _camera_directory(dataset_root: Path) -> Path:
    if dataset_root.name == "labels":
        return dataset_root.parent
    if dataset_root.name == "vial_fill":
        return dataset_root
    return dataset_root / "vial_fill"


def _recipe_files(camera: Path) -> list[tuple[int, Path]]:
    recipes = []
    for path in sorted((camera / "labels").glob("*.csv")):
        match = re.fullmatch(r"video(\d+)_vial_recipes\.csv", path.name)
        if match:
            recipes.append((int(match.group(1)), path))
        elif path.name == "vial_recipes.csv":
            recipes.append((5, path))
    ids = [video_id for video_id, _ in recipes]
    if len(ids) != len(set(ids)):
        raise ValueError("multiple recipe label files identify the same CSI video")
    return recipes


def _validated_labels(path: Path, video_id: int) -> pd.DataFrame:
    labels = read_label_csv(path)
    if not {"start_s", "end_s"}.issubset(labels):
        raise ValueError("CSI labels require explicit start_s/end_s video seconds")
    labels = validate_vials(labels, session_id=f"csi/video{video_id}")
    # File identity defines the recording; do not accept mislabeled sessions.
    labels["session_id"] = f"csi/video{video_id}"
    labels = validate_vials(labels, session_id=f"csi/video{video_id}")
    if "dispense_order" in labels:
        for row in labels.itertuples():
            expected = ">".join(
                tank
                for tank in ("T200", "T400", "T100")
                if getattr(row, f"{tank.lower()}_ml") > 0
            )
            if str(row.dispense_order).replace(" ", "") != expected:
                raise ValueError(
                    f"vial {row.vial}: dispense_order differs from requested tanks"
                )
    return labels


def _blank_measurement(label, video_id: int, video: Path) -> dict:
    x, y, width, height = monitor_config.LIQUID_ROI
    return {
        **label.to_dict(),
        "video_id": video_id,
        "camera_id": "csi_vial_fill",
        "calibration_id": "csi_raw_rgb_roi_v1",
        "r": np.nan,
        "g": np.nan,
        "b": np.nan,
        "L": np.nan,
        "a": np.nan,
        "lab_b": np.nan,
        "usable": False,
        "source_clock_verified": False,
        "quality_flags": "",
        "stable_t_start": np.nan,
        "stable_t_end": np.nan,
        "stable_t_measure": np.nan,
        "stable_decoded_t_measure": np.nan,
        "source_video_frame_index": np.nan,
        "source_fps": np.nan,
        "stable_n_frames": 0,
        "temporal_delta_e_max": np.nan,
        "spatial_delta_e_p95": np.nan,
        "source_video": str(video.resolve()),
        "source_image": None,
        "source_timestamp_age_s": np.nan,
        "source_sha256": None,
        "source_clock": "video_frame_index_seconds",
        "sample_times_s": [],
        "decoded_times_s": [],
        "sample_frames": 0,
        "measurement_source": "csi_labeled_video_stable_window",
        "roi_pixels": json.dumps([x, y, x + width, y + height]),
        "measured_fill_ml": np.nan,
        "fill_calibration_verified": False,
        "measurement_notes": "No pre-fill image is measured or subtracted; requested mL is not measured fill.",
    }


def _read_sample(cap, frame_index: int, fps: float) -> dict:
    requested_s = frame_index / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = cap.read()
    if not ok:
        return {"t_s": requested_s, "usable": False, "flags": ["missing_video_frame"]}
    actual_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    actual_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    clock_ok = (
        np.isfinite(actual_frame)
        and abs(actual_frame - frame_index) < 0.5
        and np.isfinite(actual_s)
        and abs(actual_s - requested_s) <= max(0.002, 1.5 / fps)
    )
    flags = [] if clock_ok else ["decoded_time_mismatch"]
    result = {
        "t_s": requested_s,
        "decoded_s": actual_s,
        "decoded_frame_index": actual_frame,
        "clock_ok": bool(clock_ok),
        "flags": flags,
        "usable": False,
    }
    if image.shape[:2] != (480, 640):
        flags.append("uncalibrated_image_geometry")
        return result
    measured = measure_frame(image)
    result.update(measured)
    x, y, width, height = monitor_config.LIQUID_ROI
    roi = crop(image, (x, y, width, height))
    pixel_lab = bgr_to_lab(roi).reshape(-1, 3)
    valid = (pixel_lab[:, 0] > monitor_config.MIN_L) & (
        pixel_lab[:, 0] < monitor_config.MAX_L
    )
    if valid.sum() < monitor_config.MIN_VALID_PIXELS:
        flags.append("insufficient_liquid_pixels")
        return result
    rgb_pixels = roi[:, :, ::-1].reshape(-1, 3)[valid]
    rgb = np.median(rgb_pixels, axis=0).astype(float)
    result["rgb"] = rgb
    result["spatial_delta_e_p95"] = float(
        np.percentile(color_errors(rgb_pixels, rgb), 95)
    )
    if np.mean(rgb_pixels <= 1) > 0.1:
        flags.append("clipped_dark_channels")
    if result["spatial_delta_e_p95"] > 10:
        flags.append("uneven_liquid_patch")
    if not measured["carrier_present"]:
        flags.append("carrier_not_present")
    if not np.isfinite([measured[c] for c in ("L", "a", "b")]).all():
        flags.append("color_not_observed")
    result["usable"] = not flags
    return result


def _last_stable_window(
    samples: list[dict], *, stride_s: float, min_frames: int, max_step_delta_e: float
) -> list[dict]:
    """Last low-change contiguous window, without an empty/bright-vial heuristic."""
    best, current = [], []
    for sample in samples:
        if not sample["usable"]:
            if len(current) >= min_frames:
                best = current
            current = []
            continue
        if current:
            previous = current[-1]
            contiguous = sample["t_s"] - previous["t_s"] <= stride_s * 1.5
            step = float(
                de00(
                    [sample[c] for c in ("L", "a", "b")],
                    [previous[c] for c in ("L", "a", "b")],
                )[0]
            )
            if not contiguous or step > max_step_delta_e:
                if len(current) >= min_frames:
                    best = current
                current = []
        current.append(sample)
    if len(current) >= min_frames:
        best = current
    return best


def load_csi_measurements(
    dataset_root, *, sample_stride_s=1.0, min_stable_frames=5, max_step_delta_e=1.5
) -> dict:
    """Return ``measurements``, per-video ``audit`` and ``missing_inputs``.

    ``dataset_root`` may identify the CSI wrapper, its vial_fill directory, or
    its labels directory. Invalid recipes/times raise ValueError rather than
    silently training on a partial invalid catalog. Missing inputs are explicit.
    Adjacent frame samples are not independent training/testing observations.
    """
    if not np.isfinite(sample_stride_s) or sample_stride_s <= 0:
        raise ValueError("sample_stride_s must be finite and positive")
    if not isinstance(min_stable_frames, int) or min_stable_frames < 2:
        raise ValueError("min_stable_frames must be an integer of at least two")
    if not np.isfinite(max_step_delta_e) or max_step_delta_e <= 0:
        raise ValueError("max_step_delta_e must be finite and positive")
    camera = _camera_directory(Path(dataset_root).resolve())
    files = _recipe_files(camera)
    missing, audits, rows = [], [], []
    if not files:
        missing.append(str(camera / "labels" / "video*_vial_recipes.csv"))
    for video_id, recipe_path in files:
        labels = _validated_labels(recipe_path, video_id)
        video = camera / "videos" / f"vial_fill_video_{video_id}.mp4"
        audit = {
            "video_id": video_id,
            "session_id": f"csi/video{video_id}",
            "recipe_csv": str(recipe_path),
            "source_video": str(video),
            "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            "label_rows": len(labels),
            "clock_domain": "elapsed_video_frame_index_seconds",
            "pairing": "explicit labeled start_s/end_s; no ordinal cycle matching",
            "prefill_subtraction_applied": False,
        }
        cap = cv2.VideoCapture(str(video)) if video.is_file() else None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            missing.append(str(video))
            audit["status"] = "missing_video"
            for _, label in labels.iterrows():
                record = _blank_measurement(label, video_id, video)
                record["quality_flags"] = "missing_video"
                rows.append(record)
            audits.append(audit)
            continue
        try:
            fps, n_frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if (
                not np.isfinite(fps)
                or fps <= 0
                or not np.isfinite(n_frames)
                or n_frames <= 0
            ):
                raise ValueError(f"video {video_id}: invalid frame-rate/count metadata")
            duration_s = n_frames / fps
            step_frames = max(1, round(sample_stride_s * fps))
            effective_stride_s = step_frames / fps
            audit.update(
                status="measured",
                fps=float(fps),
                duration_s=float(duration_s),
                sample_stride_s=effective_stride_s,
                video_bytes=video.stat().st_size,
                video_modified_ns=video.stat().st_mtime_ns,
                stable_window_max_end_gap_s=max(2.0, 2 * effective_stride_s),
            )
            for _, label in labels.iterrows():
                record = _blank_measurement(label, video_id, video)
                record["source_fps"] = float(fps)
                label_flags = []
                if "status" in label and str(label.status).lower() != "ok":
                    label_flags.append("label_status_not_ok")
                if not label.timing_observable:
                    record["quality_flags"] = ";".join(
                        label_flags + ["unobservable_vial_window"]
                    )
                    rows.append(record)
                    continue
                if label.end_s > duration_s + 1 / fps:
                    record["quality_flags"] = ";".join(
                        label_flags + ["labeled_window_exceeds_video"]
                    )
                    rows.append(record)
                    continue
                first = int(np.ceil(label.start_s * fps))
                stop = min(int(np.ceil(label.end_s * fps)), int(n_frames))
                samples = [
                    _read_sample(cap, index, fps)
                    for index in range(first, stop, step_frames)
                ]
                for sample in samples:
                    if sample.get("clock_ok") and not (
                        label.start_s - 1e-6 <= sample["decoded_s"] < label.end_s
                    ):
                        sample["usable"] = False
                        sample["clock_ok"] = False
                        sample["flags"].append("decoded_frame_outside_label")
                record["sample_times_s"] = [sample["t_s"] for sample in samples]
                record["decoded_times_s"] = [
                    sample.get("decoded_s", np.nan) for sample in samples
                ]
                stable = _last_stable_window(
                    samples,
                    stride_s=effective_stride_s,
                    min_frames=min_stable_frames,
                    max_step_delta_e=max_step_delta_e,
                )
                if not stable:
                    flags = sorted(
                        {flag for sample in samples for flag in sample["flags"]}
                    )
                    record["quality_flags"] = ";".join(
                        label_flags + flags + ["no_stable_filled_window"]
                    )
                    rows.append(record)
                    continue
                if label.end_s - stable[-1]["t_s"] > max(2.0, 2 * effective_stride_s):
                    flags = sorted(
                        {flag for sample in samples for flag in sample["flags"]}
                    )
                    record["quality_flags"] = ";".join(
                        label_flags + flags + ["stable_window_not_near_labeled_end"]
                    )
                    rows.append(record)
                    continue
                colors = np.asarray([sample["rgb"] for sample in stable])
                rgb = np.median(colors, axis=0)
                lab = lab_values(rgb)
                representative = int(np.argmin(color_errors(colors, rgb)))
                record.update(
                    {
                        "r": rgb[0],
                        "g": rgb[1],
                        "b": rgb[2],
                        "L": lab[0],
                        "a": lab[1],
                        "lab_b": lab[2],
                        "stable_t_start": stable[0]["t_s"],
                        "stable_t_end": stable[-1]["t_s"],
                        "stable_t_measure": stable[representative]["t_s"],
                        "stable_decoded_t_measure": stable[representative]["decoded_s"],
                        "source_video_frame_index": stable[representative][
                            "decoded_frame_index"
                        ],
                        "stable_n_frames": len(stable),
                        "sample_frames": len(stable),
                        "source_clock_verified": all(
                            sample["clock_ok"] for sample in stable
                        ),
                        "temporal_delta_e_max": float(color_errors(colors, rgb).max()),
                        "spatial_delta_e_p95": max(
                            sample["spatial_delta_e_p95"] for sample in stable
                        ),
                        "quality_flags": ";".join(label_flags),
                        "usable": not label_flags,
                    }
                )
                rows.append(record)
        finally:
            cap.release()
        audits.append(audit)
    measurements = (
        pd.DataFrame(rows) if rows else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    )
    if not measurements.empty:
        for recipe_id, group in measurements.groupby("recipe_id"):
            if len(group[ML_COLUMNS].drop_duplicates()) != 1:
                raise ValueError(
                    f"CSI recipe {recipe_id} changes composition across videos"
                )
    for audit in audits:
        part = measurements[measurements.video_id.eq(audit["video_id"])]
        audit["usable_vials"] = int(part.usable.sum())
        audit["source_clock_verified_vials"] = int(part.source_clock_verified.sum())
        audit["quality_flags"] = (
            part.quality_flags.str.split(";")
            .explode()
            .loc[lambda s: s.ne("")]
            .value_counts()
            .to_dict()
        )
    return {"measurements": measurements, "audit": audits, "missing_inputs": missing}
