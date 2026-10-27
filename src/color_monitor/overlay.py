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
