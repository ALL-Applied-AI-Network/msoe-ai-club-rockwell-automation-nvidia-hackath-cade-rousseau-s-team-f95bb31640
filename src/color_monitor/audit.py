"""Join recipe labels to observed vial cycles and report mismatches."""
import numpy as np
import pandas as pd

from . import config as C

TANK_ORDER = ["T200", "T400", "T100"]


def load_catalog():
    """Fixed recipe catalog from the supplied metadata, validated."""
    v = pd.read_csv(C.VIAL_RECIPES)
    cat = (v.groupby("recipe_id")[["t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order"]]
           .first().fillna({"t100_ml": 0, "t200_ml": 0, "t400_ml": 0}))
    problems = []
    for rid, r in cat.iterrows():
        if abs(r.t100_ml + r.t200_ml + r.t400_ml - r.total_ml) > 1e-6:
            problems.append(f"recipe {rid}: tank volumes do not sum to total_ml")
        used = [t for t, ml in zip(["T100", "T200", "T400"], [r.t100_ml, r.t200_ml, r.t400_ml]) if ml > 0]
        expect = ">".join(t for t in TANK_ORDER if t in used)
        if r.dispense_order != expect:
            problems.append(f"recipe {rid}: dispense_order {r.dispense_order} != {expect}")
    if problems:
        raise ValueError("; ".join(problems))
    return cat


def build_vial_table(cycles):
    """One row per labeled vial in video 5; observed cycle k <-> vial k."""
    v = pd.read_csv(C.VIAL_RECIPES).fillna({"t100_ml": 0, "t200_ml": 0, "t400_ml": 0})
    c5 = cycles[cycles.video_id == 5].sort_values("cycle").reset_index(drop=True)
    v["video_id"] = 5
    v["vial_id"] = v.vial
    v["cycle"] = np.where(v.vial <= len(c5), v.vial, np.nan)
    t = v.merge(c5, on=["video_id", "cycle"], how="left")
    t["label_lag_s"] = t.t_start - t.start_s
    t["previous_recipe_id"] = t.recipe_id.shift(1)
    t["position_in_run"] = t.groupby("recipe_id").cumcount() + 1
    t["status"] = t.status.fillna("no observed cycle")
    cols = ["video_id", "vial_id", "recipe_id", "start_s", "end_s", "t100_ml", "t200_ml", "t400_ml",
            "dispense_order", "previous_recipe_id", "position_in_run", "cycle", "t_start",
            "label_lag_s", "frames_available", "closed", "stable_n_frames", "stable_t_start", "stable_t_end",
            "L", "a", "b", "max_step_de", "spread_de", "status"]
    return t[cols]


def audit_report(frames, cycles, vials):
    n_vials = pd.read_csv(C.DATASET / "videos.csv").set_index("video_id").n_vials
    lines = ["DATA AUDIT", "=" * 60,
             f"Frames sampled every {C.SAMPLE_STRIDE_S:g}s from the vial_fill mp4s "
             "(stills are not used: their timestamps do not match the label timeline)."]
    for vid, g in cycles.groupby("video_id"):
        bad = int((g.status != "ok").sum())
        lines.append(f"video {vid}: {len(g)} observed vial cycles vs {n_vials[vid]} documented vials; "
                     f"{bad} without a stable color window")
    lag = vials.label_lag_s.dropna()
    lines.append(f"video 5: observed fill start minus labeled start_s: "
                 f"min={lag.min():.0f}s median={lag.median():.0f}s max={lag.max():.0f}s "
                 "(cycle k is matched to vial k; large lags would mean the match is wrong)")
    lab = pd.read_csv(C.FRAME_LABELS)
    lines.append(f"Labeled still frames between_fills: {(lab.in_fill == 0).sum()} "
                 "(not used; vial membership comes from observed cycles).")
    lines.append("Videos 1-4 have no recipe labels: measured, but not used for references.")
    return "\n".join(lines)
