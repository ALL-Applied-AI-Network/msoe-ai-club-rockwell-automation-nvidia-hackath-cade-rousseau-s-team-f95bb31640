"""CSI label sampling and explicit state-machine segmentation of unlabeled video.

All decoded patches, including rejects, remain available as physical evidence.
Segmentation may compare against an empty scene; measured color never does.
"""

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from changeover_guard.optical_knn import (
    FEATURES,
    FRACTIONS,
    ML,
    inspect_patch,
    observation_state,
    optical_features,
)


def portable_path(value, root=None):
    path = Path(str(value).replace("\\", "/")).expanduser()
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def camera_directory(value):
    path = portable_path(value)
    if path.name in ("labels", "videos"):
        return path.parent
    return path if path.name == "vial_fill" else path / "vial_fill"


@dataclass(frozen=True)
class SequenceConfig:
    # x, y, width, height in a 640x480 source. Review against actual video.
    liquid_column: tuple = (295, 270, 32, 127)
    holder_box: tuple = (265, 210, 110, 45)
    reference_size: tuple = (640, 480)
    sample_fps: float = 5.0
    settle_seconds: float = 1.5
    motion_threshold: float = 2.0
    fill_od_threshold: float = 0.04
    meniscus_min_edge: float = 8.0
    tau_mix: float = 0.08
    tolerance: float = 2.2

    def __post_init__(self):
        values = [
            self.sample_fps,
            self.settle_seconds,
            self.motion_threshold,
            self.fill_od_threshold,
            self.meniscus_min_edge,
            self.tau_mix,
            self.tolerance,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Sequence thresholds must be finite and positive")
        width, height = self.reference_size
        for x, y, w, h in (self.liquid_column, self.holder_box):
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
                raise ValueError("ROI outside reference image")


def _scaled_box(image, box, config):
    h, w = image.shape[:2]
    sx, sy = w / config.reference_size[0], h / config.reference_size[1]
    x, y, bw, bh = box
    return round(x * sx), round(y * sy), max(1, round(bw * sx)), max(1, round(bh * sy))


def liquid_patch(image, config):
    """Locate a candidate meniscus, then exclude top 20% and bottom 15%.

    Edge-derived height is a proxy until calibrated. A weak edge leaves height
    unavailable and uses the configured column for the raw color crop.
    """
    x, y, w, h = _scaled_box(image, config.liquid_column, config)
    column = image[y : y + h, x : x + w, ::-1].copy()
    if column.size == 0:
        raise ValueError("Liquid ROI has no pixels")
    edges = np.abs(cv2.Sobel(column.astype(np.float32), cv2.CV_32F, 0, 1)).mean(
        axis=(1, 2)
    )
    # Exclude image/crop boundary and the glass base from candidate edge search.
    lower, upper = 2, max(3, int(h * 0.80))
    peak = lower + int(np.argmax(edges[lower:upper]))
    strong = bool(edges[peak] >= config.meniscus_min_edge)
    meniscus = peak if strong else 0
    height = h - meniscus
    top, bottom = meniscus + int(0.20 * height), h - int(0.15 * height)
    patch = column[top:bottom].copy()
    # Standard-size raw RGB evidence; measurement uses the original crop.
    thumbnail = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
    return (
        patch,
        thumbnail,
        {
            "roi_pixels": [x, y + top, x + w, y + bottom],
            "meniscus_y": y + meniscus if strong else np.nan,
            "height_px": height * config.reference_size[1] / image.shape[0]
            if strong
            else np.nan,
            "meniscus_edge": float(edges[peak]),
        },
    )


def read_labels(path, video_id):
    labels = pd.read_csv(path)
    required = {
        "vial",
        "recipe_id",
        *ML,
        "total_ml",
        "start_s",
        "end_s",
        "start_frame",
        "end_frame",
    }
    missing = required - set(labels)
    if missing:
        raise ValueError(f"{path.name}: missing label columns {sorted(missing)}")
    labels = labels.sort_values("vial").reset_index(drop=True)
    for name in required:
        labels[name] = pd.to_numeric(labels[name], errors="raise")
    labels[ML] = labels[ML].fillna(0.0)
    if len(labels) != 30 or labels.vial.tolist() != list(range(1, 31)):
        raise ValueError(
            f"{path.name}: expected exactly vials 1–30, without duplicates"
        )
    expected = np.repeat(np.arange(5, 15), 3)
    if not np.array_equal(labels.recipe_id, expected):
        raise ValueError("Expected ten contiguous three-vial recipe blocks, IDs 5–14")
    if not np.isfinite(labels[list(required)].to_numpy(float)).all():
        raise ValueError("Nonfinite label times, frames or volumes")
    if (labels[ML] < 0).any().any() or (labels.total_ml <= 0).any():
        raise ValueError("Recipe volumes must be nonnegative with positive totals")
    if not np.allclose(labels[ML].sum(axis=1), labels.total_ml):
        raise ValueError("Component volumes must sum to total_ml")
    for _, group in labels.groupby("recipe_id"):
        if len(group[ML].drop_duplicates()) != 1:
            raise ValueError("Recipe composition changes within a block")
    if (labels.start_s < 0).any() or (labels.end_s <= labels.start_s).any():
        raise ValueError("Invalid labeled time intervals")
    if (labels.start_frame < 0).any() or (labels.end_frame <= labels.start_frame).any():
        raise ValueError("Invalid labeled frame intervals")
    if not np.equal(labels[["start_frame", "end_frame"]] % 1, 0).all().all():
        raise ValueError("Frame indices must be integers")
    if (labels.start_frame.to_numpy()[1:] < labels.end_frame.to_numpy()[:-1]).any():
        raise ValueError("Overlapping labeled vial intervals")
    labels["vial_in_run"] = (labels.vial - 1) % 3 + 1
    labels["partition"] = np.where(
        labels.vial_in_run.eq(1), "changeover_test", "settled_loov"
    )
    labels["video_id"] = video_id
    labels["vial_key"] = labels.vial.map(lambda i: f"csi/video{video_id}/vial{i}")
    labels[FRACTIONS] = labels[ML].to_numpy() / labels.total_ml.to_numpy()[:, None]
    labels["sequence_index"] = labels.vial.astype(int)
    return labels


def discover_labels(camera):
    files = []
    for path in sorted((camera / "labels").glob("*.csv")):
        match = re.fullmatch(r"video_?(\d+)_vial_recipes\.csv", path.name)
        if match or path.name == "vial_recipes.csv":
            files.append((int(match.group(1)) if match else 5, path))
    if len({i for i, _ in files}) != len(files):
        raise ValueError("Multiple CSVs identify the same video")
    return files


def video_path(camera, video_id):
    return camera / "videos" / f"vial_fill_video_{video_id}.mp4"


def _failed_sample(index, flag):
    return {
        "frame_index": int(index),
        "decoded_frame": np.nan,
        "decoded_s": np.nan,
        "usable": False,
        "ObservationState": observation_state([flag]),
        "quality_flags": flag,
        "quality_details": flag,
        **dict.fromkeys(["r", "g", "b", *FEATURES, "height_px"], np.nan),
    }


def sample_frame(cap, index, fps, config, previous=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        return _failed_sample(index, "DECODE_FAILED"), None
    actual = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    decoded_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    patch, thumbnail, geometry = liquid_patch(frame, config)
    sample = inspect_patch(
        patch, previous=previous, timestamp_offset_s=decoded_s - index / fps
    )
    if not np.isfinite(actual) or abs(actual - index) > 1:
        sample["quality_flags"] = ";".join(
            filter(None, [sample["quality_flags"], "STALE_FRAME_INDEX"])
        )
        sample["quality_details"] += f"; decoded frame {actual}, requested {index}"
        sample["usable"] = False
        sample["ObservationState"] = observation_state(
            sample["quality_flags"].split(";")
        )
    sample.update(geometry)
    sample.update(
        frame_index=int(index),
        decoded_frame=actual,
        decoded_s=decoded_s,
        thumbnail=thumbnail,
        source_fps=fps,
    )
    return sample, patch


def aggregate_vial(label, samples, rois):
    record = dict(label)
    numeric = [s for s in samples if np.isfinite(s.get("r", np.nan))]
    flags = sorted({f for s in samples for f in s["quality_flags"].split(";") if f})
    record.update(
        quality_flags=";".join(flags),
        quality_details="; ".join(
            dict.fromkeys(s["quality_details"] for s in samples if s["quality_details"])
        ),
        ObservationState=observation_state(flags),
        usable=not flags,
        sampled_frames=len(samples),
        decoded_frames=len(numeric),
        source_clock_verified=all(
            np.isfinite(s.get("decoded_s", np.nan))
            and not any(
                f in s["quality_flags"]
                for f in ("STALE", "DECODE", "MISSING", "INVALID")
            )
            for s in samples
        ),
    )
    if numeric:
        rgb = np.median([[s[c] for c in ("r", "g", "b")] for s in numeric], axis=0)
        record.update(dict(zip(["r", "g", "b"], rgb)))
        record.update(dict(zip(FEATURES, optical_features(rgb))))
        representative = min(
            numeric,
            key=lambda s: np.linalg.norm(
                np.array([s[c] for c in ("r", "g", "b")]) - rgb
            ),
        )
        record.update(
            source_video_frame_index=representative["frame_index"],
            stable_t_measure=representative["decoded_s"],
            roi_pixels=representative["roi_pixels"],
            height_px=representative["height_px"],
            meniscus_y=representative.get("meniscus_y", np.nan),
            meniscus_edge=representative.get("meniscus_edge", np.nan),
            chromatic_variance=representative.get("chromatic_variance", np.nan),
        )
        rois[record["vial_key"]] = representative["thumbnail"]
    else:
        record.update(dict.fromkeys(["r", "g", "b", *FEATURES, "height_px"], np.nan))
    return record


def load_labeled_sequences(dataset_root, config=None):
    config = config or SequenceConfig()
    camera = camera_directory(dataset_root)
    files = discover_labels(camera)
    rows, frame_rows, audits, missing, rois = [], [], [], [], {}
    for video_id, path in files:
        labels = read_labels(path, video_id)
        video = video_path(camera, video_id)
        cap = cv2.VideoCapture(str(video)) if video.is_file() else None
        audit = {
            "video_id": video_id,
            "labels": str(path),
            "source_video": str(video),
            "label_rows": len(labels),
            "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_count": 0,
            "decoded_samples": 0,
        }
        try:
            opened = cap is not None and cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if opened else 0
            valid_metadata = np.isfinite([fps, count]).all() and fps > 0 and count > 0
            if not opened or not valid_metadata:
                missing.append(str(video))
                audit["status"] = "missing_or_invalid_video"
            else:
                audit.update(status="decoded", fps=fps, frame_count=int(count))
            for label in labels.to_dict("records"):
                label["source_video"] = str(video)
                indices = np.rint(
                    np.linspace(label["end_frame"] - 45, label["end_frame"] - 10, 5)
                ).astype(int)
                samples, previous = [], None
                for index in indices:
                    if not opened or not valid_metadata:
                        sample = _failed_sample(index, "MISSING_VIDEO")
                    elif index < label["start_frame"] or index >= min(
                        label["end_frame"], count
                    ):
                        sample = _failed_sample(index, "INVALID_LABELED_WINDOW")
                    elif (
                        max(
                            abs(label["start_frame"] / fps - label["start_s"]),
                            abs(label["end_frame"] / fps - label["end_s"]),
                        )
                        > 2
                    ):
                        sample = _failed_sample(index, "STALE_LABEL_CLOCK")
                    else:
                        sample, previous = sample_frame(
                            cap, index, fps, config, previous
                        )
                    samples.append(sample)
                    frame_rows.append(
                        {
                            **label,
                            **{k: v for k, v in sample.items() if k != "thumbnail"},
                        }
                    )
                rows.append(aggregate_vial(label, samples, rois))
            audit["decoded_samples"] = sum(
                r["decoded_frames"] for r in rows if r["video_id"] == video_id
            )
            audits.append(audit)
        finally:
            if cap is not None:
                cap.release()
    if not files:
        missing.append(str(camera / "labels" / "video*_vial_recipes.csv"))
    measurements = pd.DataFrame(rows)
    if not measurements.empty:
        for recipe, group in measurements.groupby("recipe_id"):
            if len(group[ML].drop_duplicates()) != 1:
                raise ValueError(
                    f"Recipe {recipe} changes composition across labeled videos"
                )
    return {
        "measurements": measurements,
        "frames": pd.DataFrame(frame_rows),
        "rois": rois,
        "audit": audits,
        "missing_inputs": missing,
        "labeled_video_paths": [video_path(camera, i) for i, _ in files],
    }


class UnlabeledVideoSegmenter:
    """5 FPS search: empty scene → present/filling → settled → exit.

    Requires an observed absent-holder scene to initialize the background. A
    video starting mid-fill is reported in the audit; it cannot supply a known
    empty reference. No fixed three-vial counts are imposed on detections.
    """

    def __init__(self, config=None):
        self.config = config or SequenceConfig()

    def segment(self, video):
        video = portable_path(video)
        audit = {
            "source_video": str(video),
            "frame_count": 0,
            "decoded_samples": 0,
            "background_initialized": False,
            "events": 0,
        }
        if not video.is_file():
            return {
                "events": pd.DataFrame(),
                "rois": {},
                "audit": {**audit, "status": "missing_video"},
            }
        cap = cv2.VideoCapture(str(video))
        rows, rois = [], {}
        started = time.perf_counter()
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if (
                not cap.isOpened()
                or not np.isfinite([fps, count]).all()
                or min(fps, count) <= 0
            ):
                audit["status"] = "invalid_video"
                return {"events": pd.DataFrame(), "rois": rois, "audit": audit}
            stride = max(1, round(fps / self.config.sample_fps))
            audit.update(
                fps=fps, frame_count=int(count), effective_sample_fps=fps / stride
            )
            background = None
            event = None
            previous = None
            stable_since = None
            last_settled = None
            last_sample = None

            def finish(end, truncated=False):
                nonlocal event, stable_since, last_settled
                if event is None:
                    return
                sample = dict(last_settled if last_settled is not None else last_sample)
                flags = list(filter(None, sample["quality_flags"].split(";")))
                if last_settled is None:
                    flags.append("UNSETTLED_MOTION")
                if truncated:
                    flags.append("EVENT_TRUNCATED")
                sample["quality_flags"] = ";".join(sorted(set(flags)))
                sample["quality_details"] += "; " + "; ".join(flags)
                idx = len(rows) + 1
                label = {
                    "vial": idx,
                    "vial_idx": idx,
                    "sequence_index": idx,
                    "vial_key": f"unlabeled/{video.stem}/vial{idx}",
                    "source_video": str(video),
                    "start_frame": event,
                    "end_frame": end,
                    "settled_frame_idx": last_settled["frame_index"]
                    if last_settled is not None
                    else np.nan,
                    "estimated_total_ml": np.nan,
                    "partition": "unlabeled",
                }
                rows.append(aggregate_vial(label, [sample], rois))
                event, stable_since, last_settled = None, None, None

            for index in range(int(count)):
                if not cap.grab():
                    audit["decode_ended_at_frame"] = index
                    break
                if index % stride:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    stable_since, last_settled, previous = None, None, None
                    continue
                audit["decoded_samples"] += 1
                patch, thumb, geometry = liquid_patch(image, self.config)
                # Fixed full column for motion / background signals avoids changing ROI shapes.
                x, y, w, h = _scaled_box(image, self.config.liquid_column, self.config)
                column = image[y : y + h, x : x + w, ::-1].astype(float)
                hx, hy, hw, hh = _scaled_box(image, self.config.holder_box, self.config)
                holder = image[hy : hy + hh, hx : hx + hw]
                present = (holder.mean(axis=2) < 50).mean() > 0.35
                bottom_rgb = np.median(column[int(0.75 * h) :].reshape(-1, 3), axis=0)
                od = optical_features(bottom_rgb)[3:]
                if not present:
                    finish(index)
                    background = od
                    audit["background_initialized"] = True
                    previous = column
                    continue
                if background is None:
                    audit["initial_occupied_samples_skipped"] = (
                        audit.get("initial_occupied_samples_skipped", 0) + 1
                    )
                    previous = column
                    continue
                mad = (
                    float(np.abs(column - previous).mean())
                    if previous is not None
                    else np.inf
                )
                filled = np.linalg.norm(od - background) > self.config.fill_od_threshold
                chromatic_variance = float(column.std(axis=(0, 1)).mean())
                if event is None and filled:
                    event = index
                if event is not None:
                    decoded_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    sample = inspect_patch(
                        patch, timestamp_offset_s=decoded_s - index / fps
                    )
                    if mad == 0:
                        sample["quality_flags"] = ";".join(
                            filter(None, [sample["quality_flags"], "STALE_FRAME"])
                        )
                        sample["quality_details"] += (
                            "; Consecutive column pixels identical (MAD=0)"
                        )
                    sample.update(geometry)
                    sample.update(
                        frame_index=index,
                        decoded_s=decoded_s,
                        thumbnail=thumb,
                        chromatic_variance=chromatic_variance,
                    )
                    last_sample = sample
                    if mad < self.config.motion_threshold:
                        if stable_since is None:
                            stable_since = index
                        if (index - stable_since) / fps >= self.config.settle_seconds:
                            last_settled = sample
                    else:
                        stable_since = None
                        # Keep last settled sample only for the short exit movement.
                        if (
                            last_settled is not None
                            and (index - last_settled["frame_index"]) / fps > 0.6
                        ):
                            last_settled = None
                previous = column
            finish(int(count), truncated=True)
            audit.update(
                status="segmented" if rows else "no_events_detected", events=len(rows)
            )
        finally:
            cap.release()
            audit["elapsed_seconds"] = time.perf_counter() - started
            audit["within_15s_target"] = audit["elapsed_seconds"] < 15
        return {"events": pd.DataFrame(rows), "rois": rois, "audit": audit}
