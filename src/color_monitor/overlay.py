"""Overlay images for eyeballing the ROI on settled frames from every video."""
import cv2
import numpy as np

from . import config as C
from .extract import video_path


def _grab(video_id, t_s):
    cap = cv2.VideoCapture(str(video_path(video_id)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def save_overlays(cycles, path, cols=8):
    tiles = []
    for c in cycles[cycles.status == "ok"].itertuples():
        mid = (c.stable_t_start + c.stable_t_end) / 2
        img = _grab(c.video_id, mid)
        if img is None:
            continue
        x, y, w, h = C.LIQUID_ROI
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        tile = cv2.resize(img[100:440, 200:450], None, fx=0.6, fy=0.6)
        cv2.putText(tile, f"v{c.video_id} c{c.cycle} {mid:.0f}s", (3, 14), 0, 0.45, (0, 0, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def _measurement_tile(img, title, t_s, scale=0.55):
    """Draw the measured liquid ROI and a readable label on one video frame."""
    result = img.copy()
    x, y, w, h = C.LIQUID_ROI
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 3)
    label = f"{title}  t={t_s:.1f}s"
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.resize(result, None, fx=scale, fy=scale)


def save_paired_measurement_overlays(vials, path, pairs_per_row=2):
    """Save empty/filled representative frames for every measurable vial.

    Each displayed frame is the sampled frame nearest the median Lab value of
    the window used for measurement.  The yellow rectangle is the exact liquid
    ROI.  Returns the written path, or ``None`` when no pair can be decoded.
    """
    required = {
        "video_id",
        "vial_id",
        "recipe_id",
        "empty_t_measure",
        "stable_t_measure",
    }
    missing = required - set(vials)
    if missing:
        raise ValueError(f"paired overlay missing columns: {sorted(missing)}")
    if pairs_per_row < 1:
        raise ValueError("pairs_per_row must be at least one")

    pairs = []
    for vial in vials.sort_values("vial_id").itertuples():
        if not np.isfinite(vial.empty_t_measure) or not np.isfinite(
            vial.stable_t_measure
        ):
            continue
        empty = _grab(int(vial.video_id), float(vial.empty_t_measure))
        filled = _grab(int(vial.video_id), float(vial.stable_t_measure))
        if empty is None or filled is None:
            continue
        prefix = f"vial {int(vial.vial_id)} / recipe {int(vial.recipe_id)}"
        empty_tile = _measurement_tile(
            empty, f"{prefix} / EMPTY", float(vial.empty_t_measure)
        )
        filled_tile = _measurement_tile(
            filled, f"{prefix} / FILLED", float(vial.stable_t_measure)
        )
        pairs.append(np.hstack([empty_tile, filled_tile]))

    if not pairs:
        return None
    blank = np.zeros_like(pairs[0])
    while len(pairs) % pairs_per_row:
        pairs.append(blank.copy())
    rows = [
        np.hstack(pairs[index:index + pairs_per_row])
        for index in range(0, len(pairs), pairs_per_row)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise OSError(f"could not write paired overlay to {path}")
    return path
