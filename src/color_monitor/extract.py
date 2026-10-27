"""Frame-level CIELAB measurements sampled from the vial_fill videos.

The mp4s are the source of truth: the still images' timestamps do not line up
with the label timeline (label times are mp4 frame indices at 30 fps)."""
import cv2
import pandas as pd

from . import config as C
from .color import measure_frame


def video_path(video_id):
    return C.VIAL_FILL / "videos" / f"vial_fill_video_{video_id}.mp4"


def iter_frames(video_id, stride_s=C.SAMPLE_STRIDE_S):
    cap = cv2.VideoCapture(str(video_path(video_id)))
    if not cap.isOpened():
        raise FileNotFoundError(video_path(video_id))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(stride_s * fps))
    i = 0
    while cap.grab():
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                yield i / fps, img
        i += 1
    cap.release()


def extract_frames(video_ids=(1, 2, 3, 4, 5)):
    recs = []
    for vid in video_ids:
        for t, img in iter_frames(vid):
            recs.append({"video_id": vid, "t_s": round(t, 3), **measure_frame(img)})
    return pd.DataFrame(recs)
