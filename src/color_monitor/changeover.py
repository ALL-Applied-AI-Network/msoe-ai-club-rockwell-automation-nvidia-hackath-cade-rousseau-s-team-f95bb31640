"""Per-vial report in production order, with recipe-changeover tracking."""
import numpy as np
import pandas as pd

from . import config as C
from .reference import build_references, check_vial


def vial_report(vials, catalog):
    """Each vial is judged against references built WITHOUT that vial."""
    rows = []
    for v in vials.sort_values("vial_id").itertuples():
        refs = build_references(vials, catalog, exclude_vial=v.vial_id)
        basis = "not in reference" if v.position_in_run == 1 else "held out"
        if v.position_in_run > 1 and refs.loc[v.recipe_id, "status"] != "ok":
            refs = build_references(vials, catalog)  # too few other vials left to hold this one out
            basis = "reference includes this vial"
        elif v.position_in_run == 1:
            refs = build_references(vials, catalog)
        lab = np.array([v.L, v.a, v.b], dtype=float)
        if v.status != "ok":
            d, verdict = None, v.status
        else:
            d, verdict = check_vial(lab, v.recipe_id, refs)
        prev = v.previous_recipe_id
        changed = pd.notna(prev) and prev != v.recipe_id
        rows.append({
            "vial_id": v.vial_id, "recipe_id": v.recipe_id,
            "previous_recipe_id": prev, "position_in_run": v.position_in_run,
            "L": v.L, "a": v.a, "b": v.b, "stable_n_frames": v.stable_n_frames,
            "delta_e00_to_recipe": d,
            "tolerance_de": refs.loc[v.recipe_id, "tolerance_de"],
            "verdict": verdict, "reference_basis": basis,
            "after_recipe_change": bool(changed) and v.position_in_run == 1,
        })
    rep = pd.DataFrame(rows)
    rep["changeover_note"] = _changeover_notes(rep)
    return rep


def _changeover_notes(rep):
    notes = [""] * len(rep)
    for rid, run in rep.groupby("recipe_id", sort=False):
        first = run.iloc[0]
        if pd.isna(first.previous_recipe_id):
            continue
        prev = int(first.previous_recipe_id)
        idx = list(run.index)
        ok = (run.verdict == "consistent").to_numpy()
        n = C.SETTLE_CONSECUTIVE_VIALS
        settled = next((i for i in range(len(ok) - n + 1) if ok[i:i + n].all()), None)
        if settled is None:
            msg = f"color did not settle within {len(run)} vial(s)."
        elif settled == 0:
            msg = f"first vial already consistent with recipe {int(rid)}."
        else:
            msg = (f"first {settled} vial(s) differ from the recipe {int(rid)} reference; "
                   f"color settled by vial {settled + 1}.")
        notes[idx[0]] = (f"WATCH: recipe changed {prev} -> {int(rid)}; the shared path may still "
                         f"hold recipe {prev} liquid. Observed: {msg}")
    return notes
