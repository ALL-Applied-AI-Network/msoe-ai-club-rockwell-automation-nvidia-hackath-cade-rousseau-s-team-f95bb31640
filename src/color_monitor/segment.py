"""Split a video's frame measurements into physical vial cycles and find the
stable (settled) color of each one.

A cycle is: empty vial -> liquid arrives -> color settles -> carrier swap.
Cycles are found from the images themselves because the label windows do not
line up with when liquid is actually visible (see audit)."""
import numpy as np
import pandas as pd

from . import config as C
from .color import de00


def _empty_runs(df):
    empty = ((df.L >= C.EMPTY_MIN_L) & df.carrier_present).to_numpy()
    runs, start = [], None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(df) - 1))
    return [r for r in runs if r[1] - r[0] + 1 >= C.EMPTY_MIN_FRAMES]


def split_cycles(df):
    """df: one video's frames sorted by time. Returns list of frame DataFrames,
    each covering fill-start .. just before the next empty-vial run; the
    `closed` column says whether that next run (a vial swap) was observed."""
    df = df.sort_values("t_s").reset_index(drop=True)
    runs = _empty_runs(df)
    starts = [0] + [r[1] + 1 for r in runs]
    ends = [r[0] - 1 for r in runs] + [len(df) - 1]
    out = []
    for k, (s, e) in enumerate(zip(starts, ends)):
        seg = df.iloc[s:e + 1].copy()
        if len(seg) == 0 or (k == 0 and len(seg) < C.STABLE_MIN_FRAMES and runs):
            continue  # nothing, or the leftover tail of a vial that began before the recording
        # A physical vial can have a slightly different tint, scuff pattern, or
        # lighting response from its neighbours.  Retain the empty-vial phase
        # immediately before this fill so downstream color comparisons can
        # remove that per-vial background.  The first partial cycle has no
        # observed pre-fill phase and remains explicitly unpaired.
        empty_lab = np.full(3, np.nan)
        empty_n_frames = 0
        empty_spread_de = np.nan
        if k > 0:
            empty = df.iloc[runs[k - 1][0]:runs[k - 1][1] + 1]
            empty = empty[
                empty.carrier_present
                & empty.L.notna()
                & empty.a.notna()
                & empty.b.notna()
            ]
            if len(empty) >= C.EMPTY_MIN_FRAMES:
                labs = empty[["L", "a", "b"]].to_numpy(float)
                empty_lab = np.median(labs, axis=0)
                empty_n_frames = len(empty)
                empty_spread_de = float(
                    de00(labs, empty_lab.reshape(1, 3)).max()
                )
        seg["empty_L"], seg["empty_a"], seg["empty_b"] = empty_lab
        seg["empty_n_frames"] = empty_n_frames
        seg["empty_spread_de"] = empty_spread_de
        seg["closed"] = k < len(runs)  # False: video ended before the vial was swapped out
        out.append(seg)
    return out


def stable_window(seg):
    """Last sustained low-change window of usable frames, or None."""
    use = seg[seg.carrier_present & seg.L.notna()
              & (seg.L < C.EMPTY_MIN_L)].reset_index(drop=True)
    if len(use) < C.STABLE_MIN_FRAMES:
        return None
    lab = use[["L", "a", "b"]].to_numpy()
    step = de00(lab[1:], lab[:-1])
    best, run_start = None, 0
    for i in range(len(step) + 1):
        broken = i == len(step) or step[i] > C.STABLE_MAX_STEP_DE
        if broken:
            if i - run_start + 1 >= C.STABLE_MIN_FRAMES:
                best = (run_start, i)  # keep the last qualifying window
            run_start = i + 1
    if best is None:
        return None
    w = use.iloc[best[0]:best[1] + 1]
    wl = w[["L", "a", "b"]].to_numpy()
    med = np.median(wl, axis=0)
    return {
        "stable_t_start": w.t_s.iloc[0], "stable_t_end": w.t_s.iloc[-1],
        "stable_n_frames": len(w),
        "L": med[0], "a": med[1], "b": med[2],
        "max_step_de": float(step[best[0]:best[1]].max()) if best[1] > best[0] else 0.0,
        "spread_de": float(de00(wl, med.reshape(1, 3)).max()),
    }


def measure_cycles(frames):
    """One row per physical cycle per video."""
    rows = []
    for vid, g in frames.groupby("video_id"):
        for k, seg in enumerate(split_cycles(g), start=1):
            w = stable_window(seg)
            row = {"video_id": vid, "cycle": k,
                   "t_start": seg.t_s.iloc[0], "t_end": seg.t_s.iloc[-1],
                   "frames_available": len(seg), "closed": bool(seg.closed.iloc[0]),
                   "empty_L": seg.empty_L.iloc[0],
                   "empty_a": seg.empty_a.iloc[0],
                   "empty_b": seg.empty_b.iloc[0],
                   "empty_n_frames": int(seg.empty_n_frames.iloc[0]),
                   "empty_spread_de": seg.empty_spread_de.iloc[0]}
            if w is None:
                row.update(status="color measurement uncertain")
            else:
                row.update(w, status="ok")
            rows.append(row)
    return pd.DataFrame(rows)
