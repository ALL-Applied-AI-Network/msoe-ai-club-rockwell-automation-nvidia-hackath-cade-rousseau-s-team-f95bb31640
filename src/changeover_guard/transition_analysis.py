"""Validated vial identities, recipe-run boundaries and observed recovery curves.

Independent observations are production runs/sessions, never adjacent frames.
Targets from a run's own tail are explicitly exploratory and the target vials
are withheld from the curve. They cannot certify clean fluid or train a model.
"""

import hashlib
import html
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from changeover_guard.contracts import Recipe
from changeover_guard.vial_color import (
    VialCalibration,
    color_errors,
    lab_values,
    measure_vial_color,
    scaled_roi,
)

ML_COLUMNS = ["t100_ml", "t200_ml", "t400_ml"]
RGB_COLUMNS = ["r", "g", "b"]


def read_label_csv(path: str | Path) -> pd.DataFrame:
    """Accept plain CSV and the markdown emphasis present in pasted headers."""
    table = pd.read_csv(path)
    table.columns = [
        html.unescape(str(c)).replace("**", "").strip() for c in table.columns
    ]
    if table.columns.duplicated().any():
        raise ValueError("label CSV has duplicate column names")
    for name in table:
        if not (
            pd.api.types.is_object_dtype(table[name])
            or pd.api.types.is_string_dtype(table[name])
        ):
            continue
        table[name] = table[name].map(
            lambda value: (
                html.unescape(value).replace("**", "").strip()
                if isinstance(value, str)
                else value
            )
        )
        table[name] = table[name].replace("", np.nan)
    return table


def load_session_labels(
    recipe_csv: str | Path,
    frame_csv: str | Path | None = None,
    session_id="video5",
    video_start: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read either elapsed-second labels or the October 29 wall-clock schema.

    New recipe columns use start_time/end_time (HH:MM:SS). The corresponding
    frame labels provide timestamp/video_s, so timestamp - video_s establishes
    the absolute video origin. Midnight rollover is handled explicitly.
    """
    labels = read_label_csv(recipe_csv)
    timing = {"schema": "elapsed_seconds", "video_origin": None}
    if not {"start_s", "end_s"}.issubset(labels):
        if not {"start_time", "end_time"}.issubset(labels):
            raise ValueError("recipes need start_s/end_s or start_time/end_time")
        origin = pd.Timestamp(video_start) if video_start is not None else None
        if frame_csv is not None and Path(frame_csv).is_file():
            frames = read_label_csv(frame_csv)
            if {"timestamp", "video_s"}.issubset(frames):
                timestamp = pd.to_datetime(
                    frames.timestamp.astype(str).str.replace(
                        r"^(\d+):(\d+):(\d+) ", r"\1-\2-\3 ", regex=True
                    ),
                    format="mixed",
                    errors="raise",
                )
                elapsed = pd.to_numeric(frames.video_s, errors="raise")
                if (
                    not len(elapsed)
                    or not np.isfinite(elapsed).all()
                    or (elapsed < 0).any()
                    or not elapsed.is_monotonic_increasing
                ):
                    raise ValueError(
                        "frame video_s must be finite, nonnegative and ordered"
                    )
                inferred = timestamp - pd.to_timedelta(elapsed, unit="s")
                inferred_origin = inferred.iloc[0]
                if (inferred - inferred_origin).dt.total_seconds().abs().max() > 0.1:
                    raise ValueError(
                        "frame timestamp/video_s disagree about the video origin"
                    )
                if (
                    origin is not None
                    and abs((origin - inferred_origin).total_seconds()) > 0.1
                ):
                    raise ValueError("explicit video_start disagrees with frame labels")
                origin = inferred_origin
        if origin is None or pd.isna(origin):
            raise ValueError(
                "wall-clock labels require frame timestamp/video_s or an explicit video_start"
            )
        previous = origin
        start_seconds, end_seconds = [], []
        for row in labels.itertuples():
            if pd.isna(row.start_time) or pd.isna(row.end_time):
                if (
                    not hasattr(row, "status")
                    or pd.isna(row.status)
                    or str(row.status).lower() == "ok"
                ):
                    raise ValueError(
                        "missing vial timing requires an explicit non-ok label status"
                    )
                # A blackout is a missing observation, not an inferred pour.
                start_seconds.append(np.nan)
                end_seconds.append(np.nan)
                continue
            times = []
            for value in (row.start_time, row.end_time):
                text = str(value)
                if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", text):
                    # pandas requires a two-digit hour in a timedelta string.
                    text = text.zfill(8) if "." not in text else text
                    hour, minute, second = map(float, text.split(":"))
                    if hour >= 24 or minute >= 60 or second >= 60:
                        raise ValueError("invalid wall-clock time")
                    value_time = previous.normalize() + pd.Timedelta(
                        seconds=hour * 3600 + minute * 60 + second
                    )
                    if value_time < previous:
                        if (previous - value_time).total_seconds() <= 12 * 3600:
                            raise ValueError("wall-clock label times are out of order")
                        value_time += pd.Timedelta(days=1)
                else:
                    value_time = pd.Timestamp(text)
                    if pd.isna(value_time) or value_time < previous:
                        raise ValueError(
                            "wall-clock label times are missing or out of order"
                        )
                times.append(value_time)
                previous = value_time
            start_seconds.append((times[0] - origin).total_seconds())
            end_seconds.append((times[1] - origin).total_seconds())
        labels["start_s"], labels["end_s"] = start_seconds, end_seconds
        timing = {
            "schema": "wall_clock",
            "video_origin": origin.isoformat(),
            "origin_source": "explicit"
            if video_start is not None
            else "frame_timestamp_minus_video_s",
        }
    return validate_vials(labels, session_id), timing


def validate_vials(table: pd.DataFrame, session_id="video5") -> pd.DataFrame:
    required = {"vial", "recipe_id", "start_s", "end_s", *ML_COLUMNS}
    if not required.issubset(table):
        raise ValueError(
            f"vial labels missing columns: {sorted(required - set(table))}"
        )
    if table.empty:
        raise ValueError("vial labels are empty")
    data = table.copy().reset_index(drop=True)
    if "session_id" not in data:
        data["session_id"] = str(session_id)
    if data.session_id.isna().any():
        raise ValueError("session_id cannot be missing")
    data["session_id"] = data.session_id.astype(str)
    for name in ["vial", "recipe_id", "start_s", "end_s", *ML_COLUMNS]:
        data[name] = pd.to_numeric(data[name], errors="raise")
    data[ML_COLUMNS] = data[ML_COLUMNS].fillna(0)
    numeric = data[["vial", "recipe_id", *ML_COLUMNS]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("vial labels must be finite")
    data["timing_observable"] = data[["start_s", "end_s"]].notna().all(axis=1)
    status_ok = (
        data["status"].astype(str).str.lower().eq("ok")
        if "status" in data
        else pd.Series(True, index=data.index)
    )
    missing_status = (
        data["status"].isna() if "status" in data else pd.Series(True, index=data.index)
    )
    if (~data.timing_observable & (status_ok | missing_status)).any():
        raise ValueError("missing vial timing requires an explicit non-ok label status")
    if np.isinf(data[["start_s", "end_s"]].to_numpy()).any():
        raise ValueError("vial times must be finite or explicitly unobservable")
    for name in ("vial", "recipe_id"):
        if not (data[name] == np.floor(data[name])).all() or (data[name] < 1).any():
            raise ValueError(f"{name} must contain positive integers")
        data[name] = data[name].astype(int)
    if data.duplicated(["session_id", "vial"]).any():
        raise ValueError("each session/vial must map to exactly one recipe")
    if (data.start_s < 0).any() or (data.end_s <= data.start_s).any():
        raise ValueError("vial time windows must have 0 <= start_s < end_s")
    for row in data.itertuples():
        recipe = Recipe(
            row.recipe_id,
            {
                c[:4].upper(): getattr(row, c)
                for c in ML_COLUMNS
                if getattr(row, c) != 0
            },
        )
        recipe.validate()
        if not recipe.ml:
            raise ValueError("recipe must dispense a positive total volume")
    for recipe_id, rows in data.groupby("recipe_id"):
        if len(rows[ML_COLUMNS].drop_duplicates()) != 1:
            raise ValueError(
                f"recipe {recipe_id} changes composition without changing recipe_id"
            )
    data["requested_ml"] = data[ML_COLUMNS].sum(axis=1)
    if "total_ml" in data:
        declared = pd.to_numeric(data.total_ml, errors="raise")
        if not np.isfinite(declared).all() or not np.allclose(
            declared, data.requested_ml
        ):
            raise ValueError("total_ml differs from the sum of recipe components")
    if "dispensed_ml" in data:
        data["dispensed_ml"] = pd.to_numeric(data.dispensed_ml, errors="raise")
        if not np.isfinite(data.dispensed_ml).all() or (data.dispensed_ml <= 0).any():
            raise ValueError(
                "dispensed_ml must contain positive finite measured volumes"
            )
        if "volume_source" not in data:
            data["volume_source"] = "measured_total_requested_composition"
    else:
        data["dispensed_ml"] = data.requested_ml
        data["volume_source"] = "requested_proxy"
    if "flush_ml" not in data:
        data["flush_ml"] = np.nan  # unknown is not zero
    data["flush_ml"] = pd.to_numeric(data.flush_ml, errors="raise")
    if np.isinf(data.flush_ml).any() or (data.flush_ml.dropna() < 0).any():
        raise ValueError("flush_ml must be nonnegative or unknown")
    sessions = []
    for session, rows in data.groupby("session_id", sort=False):
        rows = rows.copy()
        if (
            not rows.vial.is_monotonic_increasing
            or not rows.loc[rows.timing_observable, "start_s"].is_monotonic_increasing
        ):
            raise ValueError("vials and timestamps must be ordered within each session")
        if len(rows) > 1 and (np.diff(rows.vial) != 1).any():
            raise ValueError(
                "missing vial labels make throughput unknown; supply a contiguous session"
            )
        observed_times = rows[rows.timing_observable]
        if (
            observed_times.start_s.to_numpy()[1:] < observed_times.end_s.to_numpy()[:-1]
        ).any():
            raise ValueError("vial time windows overlap")
        changes = rows.recipe_id.ne(rows.recipe_id.shift())
        rows["run_number"] = changes.cumsum()
        rows["run_id"] = [f"{session}/run{number}" for number in rows.run_number]
        rows["vial_key"] = [f"{session}/vial{vial}" for vial in rows.vial]
        rows["vial_in_run"] = rows.groupby("run_id").cumcount() + 1
        rows["cumulative_ml"] = rows.groupby("run_id").dispensed_ml.cumsum()
        rows["volume_start_ml"] = rows.cumulative_ml - rows.dispensed_ml
        row_status_ok = (
            rows["status"].astype(str).str.lower().eq("ok")
            if "status" in rows
            else pd.Series(True, index=rows.index)
        )
        uncertain_volume = (
            ~row_status_ok | ~rows.timing_observable
        ) & rows.volume_source.eq("requested_proxy")
        rows["volume_usable"] = ~uncertain_volume.groupby(rows.run_id).cummax()
        rows["is_transition"] = changes & rows.run_number.gt(1)
        if (rows.flush_ml.fillna(0).gt(0) & ~changes).any():
            raise ValueError(
                "a mid-run flush needs a new explicit session/history boundary"
            )
        sessions.append(rows)
    return pd.concat(sessions, ignore_index=True)


def label_inventory(table: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Inventory experimental units even when image/video bytes are unavailable."""
    runs, transitions = [], []
    for session, rows in table.groupby("session_id", sort=False):
        previous = None
        for run_id, group in rows.groupby("run_id", sort=False):
            status = (
                group.status.astype(str).str.lower()
                if "status" in group
                else pd.Series("ok", index=group.index)
            )
            record = {
                "session_id": session,
                "run_id": run_id,
                "recipe_id": int(group.recipe_id.iloc[0]),
                "first_vial": int(group.vial.iloc[0]),
                "last_vial": int(group.vial.iloc[-1]),
                "labeled_vials": len(group),
                "timed_vials": int(group.timing_observable.sum()),
                "operator_error_vials": int(status.ne("ok").sum()),
                "requested_total_ml": float(group.requested_ml.sum()),
                "volume_history_usable": bool(group.volume_usable.all()),
                "start_s": float(group.start_s.iloc[0]),
                "end_s": float(group.end_s.iloc[-1]),
                **group[ML_COLUMNS].iloc[0].to_dict(),
            }
            runs.append(record)
            if previous is not None:
                transitions.append(
                    {
                        "transition_id": run_id,
                        "session_id": session,
                        "previous_recipe_id": previous["recipe_id"],
                        "next_recipe_id": record["recipe_id"],
                        "previous_run_id": previous["run_id"],
                        "first_vial": record["first_vial"],
                        "next_run_labeled_vials": len(group),
                        "next_run_timed_vials": record["timed_vials"],
                        "previous_volume_history_usable": previous[
                            "volume_history_usable"
                        ],
                        "next_volume_history_usable": record["volume_history_usable"],
                    }
                )
            previous = record
    run_table = pd.DataFrame(runs)
    transition_table = pd.DataFrame(
        transitions,
        columns=[
            "transition_id",
            "session_id",
            "previous_recipe_id",
            "next_recipe_id",
            "previous_run_id",
            "first_vial",
            "next_run_labeled_vials",
            "next_run_timed_vials",
            "previous_volume_history_usable",
            "next_volume_history_usable",
        ],
    )
    pairs = (
        (
            transition_table.groupby(["previous_recipe_id", "next_recipe_id"])
            .agg(
                occurrences=("transition_id", "size"),
                independent_sessions=("session_id", "nunique"),
            )
            .reset_index()
            .to_dict("records")
        )
        if not transition_table.empty
        else []
    )
    recipes = table[["recipe_id", *ML_COLUMNS]].drop_duplicates()
    recipe_ids = sorted(map(int, recipes.recipe_id))
    seen_pairs = {(p["previous_recipe_id"], p["next_recipe_id"]) for p in pairs}
    inventory = {
        "vials": len(table),
        "sessions": int(table.session_id.nunique()),
        "recipe_runs": len(runs),
        "transitions": len(transitions),
        "distinct_recipes": len(recipes),
        "directed_pairs": pairs,
        "unobserved_directions": [
            [a, b]
            for a in recipe_ids
            for b in recipe_ids
            if a != b and (a, b) not in seen_pairs
        ],
        "mixture_recipes": int((recipes[ML_COLUMNS].gt(0).sum(axis=1) > 1).sum()),
        "unobservable_vials": table.loc[~table.timing_observable, "vial_key"].tolist(),
        "operator_error_vials": table.loc[
            table.status.astype(str).str.lower().ne("ok"), "vial_key"
        ].tolist()
        if "status" in table
        else [],
        "note": "Counts describe labels. Multiple changes within one recording do not establish independent sessions or physical recovery.",
    }
    return inventory, run_table, transition_table


def validate_frame_labels(path: Path, table: pd.DataFrame) -> dict:
    """Audit still labels without trusting their known offset for frame selection."""
    if not path.is_file():
        return {
            "status": "missing",
            "path": str(path),
            "warning": "Still-label identity audit unavailable.",
        }
    frames = read_label_csv(path)
    required = {"filename", "in_fill", "vial", "recipe_id"}
    if (
        not required.issubset(frames)
        or frames.filename.isna().any()
        or frames.filename.duplicated().any()
    ):
        raise ValueError(
            "frame labels need unique filenames and in_fill,vial,recipe_id"
        )
    if not frames.in_fill.isin([0, 1]).all():
        raise ValueError("in_fill must contain 0 or 1")
    known = table.set_index("vial").recipe_id.to_dict()
    for row in frames[frames.vial.notna()].itertuples():
        if (
            row.vial not in known
            or pd.isna(row.recipe_id)
            or row.recipe_id != known[row.vial]
        ):
            raise ValueError(
                "frame labels disagree with the unique vial-to-recipe mapping"
            )
    images = path.parent.parent / "images"
    missing_images = [name for name in frames.filename if not (images / name).is_file()]
    return {
        "status": "audited",
        "frames": len(frames),
        "labeled_vials": int(frames.vial.nunique()),
        "expected_images": len(frames),
        "available_images": len(frames) - len(missing_images),
        "missing_image_examples": missing_images[:3],
        "warning": "Review clock domains and full-vial alignment in the contact sheets. Original video5 stills can be offset by one vial.",
    }


def measure_video(
    table: pd.DataFrame,
    video: str | Path,
    calibration: VialCalibration,
    contact_dir: Path,
    offsets=(-0.6, -0.45, -0.3),
) -> pd.DataFrame:
    """Aggregate three full-vial frames per physical vial, retaining failed rows.

    Decoded timestamps are checked against the requested video times. Multiple
    frames improve robustness but never increase the independent sample count.
    """
    if table.session_id.nunique() != 1:
        raise ValueError("measure one session/video at a time")
    if not offsets or not np.isfinite(offsets).all() or max(offsets) >= 0:
        raise ValueError("provide finite negative full-vial sampling offsets")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"could not open video: {video}")
    contact_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video has no usable frame rate")
        for row in table.itertuples():
            if not row.timing_observable:
                records.append(
                    {
                        "vial_key": row.vial_key,
                        "camera_id": calibration.camera_id,
                        "calibration_id": calibration.calibration_id,
                        "sample_times_s": [],
                        "decoded_times_s": [],
                        "actual_source_times_s": [],
                        "source_clock": "video_container_seconds",
                        "source_clock_verified": False,
                        "sample_frames": 0,
                        "contact_frame": None,
                        "usable": False,
                        "quality_flags": "unobservable_vial_window;label_status_not_ok",
                        **dict.fromkeys(RGB_COLUMNS, np.nan),
                    }
                )
                continue
            readings, requested, actual, flags, image_for_review = [], [], [], [], None
            for offset in offsets:
                when = row.end_s + offset
                requested.append(float(when))
                if not row.start_s <= when < row.end_s:
                    flags.append("sampling_time_outside_vial_window")
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
                ok, image = cap.read()
                if not ok:
                    flags.append("missing_video_frame")
                    continue
                decoded = (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps
                actual.append(float(decoded))
                if (
                    abs(decoded - when) > max(0.15, 1.5 / fps)
                    or not row.start_s <= decoded < row.end_s
                ):
                    flags.append("video_timestamp_mismatch")
                reading = measure_vial_color(image, calibration)
                flags.extend(reading["flags"])
                if reading["usable"]:
                    readings.append(reading)
                image_for_review = image.copy()
            contact_file = None
            if image_for_review is not None:
                x0, y0, x1, y1 = scaled_roi(image_for_review.shape, calibration)
                cv2.rectangle(image_for_review, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(
                    image_for_review,
                    f"vial {row.vial} recipe {row.recipe_id} end {row.end_s:.2f}s",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                contact_file = contact_dir / f"vial_{row.vial:05d}.png"
                if not cv2.imwrite(str(contact_file), image_for_review):
                    raise OSError(f"could not save contact frame: {contact_file}")
            record = {
                "vial_key": row.vial_key,
                "camera_id": calibration.camera_id,
                "calibration_id": calibration.calibration_id,
                "sample_times_s": requested,
                "decoded_times_s": actual,
                "actual_source_times_s": actual,
                "source_clock": "video_container_seconds",
                "source_clock_verified": len(actual) == len(offsets)
                and "video_timestamp_mismatch" not in flags,
                "sample_frames": len(readings),
                "contact_frame": str(contact_file) if contact_file else None,
            }
            if len(readings) != len(offsets):
                flags.append("incomplete_frame_samples")
            if hasattr(row, "status") and str(row.status).lower() != "ok":
                flags.append("label_status_not_ok")
            if readings:
                rgb = np.median([reading["rgb"] for reading in readings], axis=0)
                temporal = float(
                    np.max(color_errors(np.array([r["rgb"] for r in readings]), rgb))
                )
                if temporal > 2.2 / 3:
                    flags.append("unstable_full_vial_frames")
                record.update(dict(zip(RGB_COLUMNS, rgb)))
                record.update(dict(zip(["L", "a", "lab_b"], lab_values(rgb))))
                record.update(
                    {
                        "temporal_delta_e_max": temporal,
                        "spatial_delta_e_p95": max(
                            r["spatial_delta_e_p95"] for r in readings
                        ),
                        "valid_fraction": min(r["valid_fraction"] for r in readings),
                    }
                )
            else:
                record.update(dict.fromkeys(RGB_COLUMNS, np.nan))
            record.update(
                {"usable": not flags, "quality_flags": ";".join(sorted(set(flags)))}
            )
            records.append(record)
    finally:
        cap.release()
    return table.merge(pd.DataFrame(records), on="vial_key", validate="one_to_one")


def measure_timestamped_stills(
    table: pd.DataFrame,
    frame_csv: Path,
    calibration: VialCalibration,
    contact_dir: Path,
    full_vial_window_s=2.5,
    clock_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use the explicitly aligned wall-clock stills when container time differs.

    One still nearest end_s - 0.6 is selected inside the last 2.5 seconds and
    must carry the same vial/recipe identity. Its actual time is retained. A
    single still provides no temporal repeatability estimate.
    """
    frames = read_label_csv(frame_csv)
    if not {"video_s", "timestamp", "vial", "recipe_id"}.issubset(frames):
        raise ValueError(
            "timestamped-still sampling requires explicit frame timestamp/video_s and identities"
        )
    contact_dir.mkdir(parents=True, exist_ok=True)
    audited = {}
    if clock_audit is not None:
        required = {"filename", "source_sha256", "camera_timestamp"}
        if (
            not required.issubset(clock_audit)
            or clock_audit[list(required)].isna().any().any()
            or clock_audit.filename.duplicated().any()
        ):
            raise ValueError(
                "clock audit needs unique filenames, source_sha256 and camera_timestamp"
            )
        audited = clock_audit.set_index("filename").to_dict("index")
    records = []
    for row in table.itertuples():
        record = {
            "vial_key": row.vial_key,
            "camera_id": calibration.camera_id,
            "calibration_id": calibration.calibration_id,
            "sample_times_s": [],
            "decoded_times_s": [],
            "actual_source_times_s": [],
            "source_clock": "wall_clock_relative_seconds",
            "source_clock_verified": False,
            "sample_frames": 0,
            "contact_frame": None,
            "measurement_source": "timestamped_still",
            "temporal_delta_e_max": None,
            "measurement_notes": "Single still; temporal repeatability not measured.",
            **dict.fromkeys(RGB_COLUMNS, np.nan),
        }
        flags = []
        if not row.timing_observable:
            flags.append("unobservable_vial_window")
        else:
            candidates = frames[
                (frames.vial == row.vial)
                & (frames.recipe_id == row.recipe_id)
                & frames.video_s.ge(max(row.start_s, row.end_s - full_vial_window_s))
                & frames.video_s.le(row.end_s - 0.15)
            ]
            if candidates.empty:
                flags.append("no_aligned_full_vial_still")
            else:
                chosen = candidates.loc[
                    (candidates.video_s - (row.end_s - 0.6)).abs().idxmin()
                ]
                source = frame_csv.parent.parent / "images" / chosen.filename
                image = cv2.imread(str(source))
                record.update(
                    {
                        "sample_times_s": [row.end_s - 0.6],
                        "actual_source_times_s": [float(chosen.video_s)],
                        "source_image": str(source),
                        "source_timestamp": str(chosen.timestamp),
                    }
                )
                if image is None:
                    flags.append("missing_indexed_still")
                else:
                    record["source_sha256"] = hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()
                    audit = audited.get(chosen.filename)
                    if audit is not None:
                        if audit["source_sha256"] != record["source_sha256"]:
                            flags.append("camera_clock_audit_hash_mismatch")
                        else:
                            camera_time = pd.Timestamp(audit["camera_timestamp"])
                            metadata_time = pd.Timestamp(chosen.timestamp)
                            age = (metadata_time - camera_time).total_seconds()
                            record["source_camera_timestamp"] = camera_time.isoformat()
                            record["source_timestamp_age_s"] = age
                            record["actual_source_times_s"] = [
                                float(chosen.video_s - age)
                            ]
                            if abs(age) > 0.5:
                                flags.append("stale_camera_frame")
                            else:
                                record["source_clock_verified"] = True
                    reading = measure_vial_color(image, calibration)
                    flags.extend(reading["flags"])
                    if reading.get("rgb") is not None:
                        record.update(dict(zip(RGB_COLUMNS, reading["rgb"])))
                        record.update(dict(zip(["L", "a", "lab_b"], reading["lab"])))
                        record["spatial_delta_e_p95"] = reading["spatial_delta_e_p95"]
                        record["valid_fraction"] = reading["valid_fraction"]
                    record["sample_frames"] = 1
                    x0, y0, x1, y1 = scaled_roi(image.shape, calibration)
                    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(
                        image,
                        f"vial {row.vial} recipe {row.recipe_id}: wall-clock still",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    contact = contact_dir / f"vial_{row.vial:05d}.png"
                    if not cv2.imwrite(str(contact), image):
                        raise OSError(f"could not save contact frame: {contact}")
                    record["contact_frame"] = str(contact)
        if hasattr(row, "status") and str(row.status).lower() != "ok":
            flags.append("label_status_not_ok")
        if not record["source_clock_verified"]:
            record["measurement_notes"] += " Camera timestamp is not verified."
        record.update(
            {"usable": not flags, "quality_flags": ";".join(sorted(set(flags)))}
        )
        records.append(record)
    return table.merge(pd.DataFrame(records), on="vial_key", validate="one_to_one")


def validate_clean_controls(
    controls: pd.DataFrame, calibration: VialCalibration
) -> pd.DataFrame:
    required = {
        "control_id",
        "source_run_id",
        "source_session_id",
        "recipe_id",
        "camera_id",
        "calibration_id",
        "is_clean_control",
        *RGB_COLUMNS,
        *ML_COLUMNS,
    }
    if (
        not required.issubset(controls)
        or controls[list(required - set(ML_COLUMNS))].isna().any().any()
    ):
        raise ValueError(
            "clean controls need complete IDs, provenance, RGB, composition and is_clean_control"
        )
    result = controls.copy()
    result[ML_COLUMNS] = result[ML_COLUMNS].fillna(0)
    if result.empty or result.control_id.duplicated().any():
        raise ValueError("clean control IDs must be nonempty and unique")
    if not result.is_clean_control.isin([True, 1]).all():
        raise ValueError(
            "only explicitly verified clean controls may fit the optical model"
        )
    if (
        not result.camera_id.eq(calibration.camera_id).all()
        or not result.calibration_id.eq(calibration.calibration_id).all()
    ):
        raise ValueError("clean controls belong to a different camera/calibration")
    lab_values(result[RGB_COLUMNS].to_numpy(float))
    if (result[RGB_COLUMNS].to_numpy(float) <= 1).any():
        raise ValueError(
            "clipped clean-control channels cannot identify finite absorbance; remeasure with a usable exposure"
        )
    for row in result.itertuples():
        recipe = Recipe(
            int(row.recipe_id),
            {
                c[:4].upper(): float(getattr(row, c))
                for c in ML_COLUMNS
                if getattr(row, c) != 0
            },
        )
        recipe.validate()
        if not recipe.ml or not np.isfinite(list(recipe.ml.values())).all():
            raise ValueError(
                "clean control recipe quantities must be finite and positive"
            )
    for _, rows in result.groupby(["source_session_id", "source_run_id"]):
        if (
            rows.recipe_id.nunique() != 1
            or len(rows[ML_COLUMNS].drop_duplicates()) != 1
        ):
            raise ValueError("a clean control run must have one recipe and composition")
    return result


def recovery_curves(
    measurements: pd.DataFrame,
    controls: pd.DataFrame | None = None,
    threshold=2.2,
    consecutive=3,
    tail_count=2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    if (
        not np.isfinite(threshold)
        or threshold <= 0
        or consecutive < 1
        or tail_count < 2
    ):
        raise ValueError(
            "threshold/consecutive/tail_count must be positive; use at least two tail controls"
        )
    if measurements.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    tails = []
    for run_id, rows in measurements.groupby("run_id", sort=False):
        # Keep target vials separate from evaluated vials, including bad frames.
        tail = rows.tail(tail_count)
        if len(rows) <= tail_count or len(tail) < tail_count or not tail.usable.all():
            continue
        rgb = tail[RGB_COLUMNS].median().to_numpy()
        if color_errors(tail[RGB_COLUMNS].to_numpy(), rgb).max() < threshold / 3:
            tails.append(
                {
                    "source_run_id": run_id,
                    "recipe_id": int(rows.recipe_id.iloc[0]),
                    "rgb": rgb,
                    "excluded_keys": set(tail.vial_key),
                }
            )
    curves, summaries, targets = [], [], []
    for session, session_rows in measurements.groupby("session_id", sort=False):
        runs = list(session_rows.groupby("run_id", sort=False))
        for index, (run_id, rows) in enumerate(runs):
            if index == 0:
                continue  # initial recipe history is unknown
            previous = runs[index - 1][1]
            recipe_id = int(rows.recipe_id.iloc[0])
            target, source, excluded, source_runs = None, "missing", set(), []
            if controls is not None:
                eligible = controls[
                    (controls.recipe_id == recipe_id)
                    & ~controls.source_run_id.eq(run_id)
                ]
                if not eligible.empty:
                    if not np.allclose(eligible[ML_COLUMNS], rows[ML_COLUMNS].iloc[0]):
                        raise ValueError(
                            "clean control composition differs from the evaluated recipe"
                        )
                    # Average captures within a control run first; runs have equal weight.
                    independent = eligible.groupby(
                        ["source_session_id", "source_run_id"]
                    )[RGB_COLUMNS].median()
                    target = independent.median().to_numpy()
                    source, source_runs = (
                        "independent_clean_controls",
                        list(eligible.source_run_id.unique()),
                    )
            if target is None:
                candidate = [
                    t
                    for t in tails
                    if t["recipe_id"] == recipe_id and t["source_run_id"] != run_id
                ]
                if candidate:
                    target = np.median([t["rgb"] for t in candidate], axis=0)
                    source, source_runs = (
                        "other_run_tail_surrogate",
                        [t["source_run_id"] for t in candidate],
                    )
                else:
                    candidate = [t for t in tails if t["source_run_id"] == run_id]
                    if candidate:
                        target, excluded = (
                            candidate[0]["rgb"],
                            candidate[0]["excluded_keys"],
                        )
                        source, source_runs = "same_run_tail_surrogate", [run_id]
            target_record = {
                "run_id": run_id,
                "recipe_id": recipe_id,
                "source": source,
                "source_runs": source_runs,
                "excluded_vial_keys": sorted(excluded),
                "rgb": None if target is None else target.tolist(),
                "lab": None if target is None else lab_values(target).tolist(),
            }
            targets.append(target_record)
            current = rows.copy()
            current["transition_id"] = run_id
            current["previous_recipe_id"] = int(previous.recipe_id.iloc[0])
            current["target_source"] = source
            current["target_withheld"] = current.vial_key.isin(excluded)
            current["delta_e_to_clean"] = np.nan
            valid = current.usable & ~current.target_withheld
            if target is not None:
                current.loc[valid, "delta_e_to_clean"] = color_errors(
                    current.loc[valid, RGB_COLUMNS].to_numpy(), target
                )
                for c, value in zip(RGB_COLUMNS, target):
                    current[f"target_{c}"] = value
            current["below_threshold_exploratory"] = (
                current.delta_e_to_clean.lt(threshold) & valid
            )
            passing = current.below_threshold_exploratory.to_numpy()
            first = next(
                (
                    i
                    for i in range(len(passing) - consecutive + 1)
                    if passing[i : i + consecutive].all()
                ),
                None,
            )
            clear_row = None if first is None else current.iloc[first]
            confirmed = None if first is None else current.iloc[first + consecutive - 1]
            if first is not None:
                censor_reason = None
            elif target is None:
                censor_reason = "no_independent_target_or_stable_tail"
            elif len(current) < consecutive:
                censor_reason = "insufficient_followup"
            else:
                censor_reason = (
                    "session_ended_or_recipe_changed_before_confirmed_recovery"
                )
            summary = {
                "transition_id": run_id,
                "session_id": session,
                "previous_recipe_id": int(previous.recipe_id.iloc[0]),
                "next_recipe_id": recipe_id,
                "previous_total_processed_ml": float(previous.dispensed_ml.sum()),
                "flush_ml": float(rows.flush_ml.iloc[0]),
                "target_source": source,
                "previous_composition_ml": previous[ML_COLUMNS].iloc[0].to_dict(),
                "next_composition_ml": rows[ML_COLUMNS].iloc[0].to_dict(),
                "first_vial": int(rows.vial.iloc[0]),
                "last_observed_vial": int(rows.vial.iloc[-1]),
                "observed_vials": len(rows),
                "usable_curve_vials": int(current.delta_e_to_clean.notna().sum()),
                "initial_delta_e": float(current.delta_e_to_clean.iloc[0]),
                "right_censored": first is None,
                "censor_reason": censor_reason,
                "observed_volume_ml": float(rows.dispensed_ml.sum()),
                "vials_to_clear": None
                if clear_row is None
                else int(clear_row.vial_in_run),
                "volume_to_clear_ml": None
                if clear_row is None or not clear_row.volume_usable
                else float(clear_row.cumulative_ml),
                "seconds_to_clear": None
                if clear_row is None
                else float(clear_row.end_s - rows.start_s.iloc[0]),
                "clearance_confirmed_at_vial": None
                if confirmed is None
                else int(confirmed.vial),
                "volume_to_confirm_ml": None
                if confirmed is None or not confirmed.volume_usable
                else float(confirmed.cumulative_ml),
                "low_quality": bool((~rows.usable).any()),
                "volume_source": ";".join(rows.volume_source.unique()),
                "volume_history_usable": bool(rows.volume_usable.all()),
                "recovery_scope": "observed_clean_reference"
                if source == "independent_clean_controls"
                else "exploratory_surrogate",
                "production_status": "PENDING",
            }
            if target is not None:
                previous_rgb = (
                    previous.loc[previous.usable, RGB_COLUMNS]
                    .tail(tail_count)
                    .median()
                    .to_numpy()
                )
                summary["direction"] = "unknown"
                if np.isfinite(previous_rgb).all():
                    summary["direction"] = (
                        "dark_to_light"
                        if lab_values(target)[0] > lab_values(previous_rgb)[0]
                        else "light_to_dark_or_equal"
                    )
            summaries.append(summary)
            curves.append(current)
    return (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        pd.DataFrame(summaries),
        targets,
    )


def transition_support(summary: pd.DataFrame, curves: pd.DataFrame) -> dict:
    """Conservative modeling prerequisites, not a statistical power calculation."""
    if summary.empty:
        return {
            "transitions": 0,
            "sessions": 0,
            "shared_decay_supported": False,
            "transition_specific_supported": False,
            "reason": "No observed recipe changes.",
        }
    qualified = []
    for transition_id, rows in (
        curves.groupby("transition_id") if not curves.empty else []
    ):
        usable = rows[rows.usable & rows.delta_e_to_clean.notna()]
        if (
            usable.vial_in_run.nunique() >= 4
            and rows.target_source.eq("independent_clean_controls").all()
            and ("volume_usable" not in rows or rows.volume_usable.all())
            and (
                "source_clock_verified" not in rows or rows.source_clock_verified.all()
            )
        ):
            qualified.append(transition_id)
    valid = summary[summary.transition_id.isin(qualified)]
    pairs = (
        summary.groupby(["previous_recipe_id", "next_recipe_id"])
        .agg(
            occurrences=("transition_id", "size"),
            independent_sessions=("session_id", "nunique"),
        )
        .reset_index()
        .to_dict("records")
    )
    shared = len(valid) >= 5 and valid.session_id.nunique() >= 3
    repeated = []
    if not valid.empty:
        for pair, rows in valid.groupby(["previous_recipe_id", "next_recipe_id"]):
            if rows.session_id.nunique() >= 3:
                repeated.append(list(map(int, pair)))
    return {
        "transitions": len(summary),
        "sessions": int(summary.session_id.nunique()),
        "directed_pairs": pairs,
        "qualified_transitions": qualified,
        "qualified_sessions": int(valid.session_id.nunique()),
        "shared_decay_supported": shared,
        "transition_specific_supported": bool(repeated),
        "repeated_supported_pairs": repeated,
        "reason": (
            "Shared single-exponential fit can be evaluated by held-out session."
            if shared
            else "Need at least five transitions in three sessions, each with four usable vials and independent clean controls."
        ),
        "limitation": "Counts are minimum identifiability checks, not proof of statistical power. Frames and same-session transitions are correlated.",
    }
