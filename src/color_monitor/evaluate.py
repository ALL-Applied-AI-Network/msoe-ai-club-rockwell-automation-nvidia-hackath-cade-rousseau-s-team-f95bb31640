"""Evaluation on the one labeled run (video 5). Unit of holdout is the vial:
the reference for a vial never includes that vial."""
import numpy as np
import pandas as pd

from .color import de00
from .reference import build_references, check_vial, eligible_reference_vials


def evaluate(vials, catalog, report):
    out = ["EVALUATION (video 5 only; held out by vial, not by video)", "=" * 60]
    ok = vials[vials.status == "ok"]

    out.append("\n1. Can we measure color consistently?")
    out.append(f"  stable-window spread (max dE00 to window median): "
               f"median={ok.spread_de.median():.2f} max={ok.spread_de.max():.2f}")
    pair = []
    for rid, g in eligible_reference_vials(vials).groupby("recipe_id"):
        lab = g[["L", "a", "b"]].to_numpy(float)
        d = [de00(lab[i:i + 1], lab[j:j + 1])[0]
             for i in range(len(lab)) for j in range(i + 1, len(lab))]
        if d:
            pair.append((rid, max(d)))
    out.append("  max dE00 between settled vials of the same recipe: " +
               ", ".join(f"r{int(r)}={d:.1f}" for r, d in pair))

    out.append("\n2. Can we detect a wrong color for the intended recipe?")
    out.append("  (leave-one-vial-out; with 2 settled vials per recipe the held-out reference is "
               "one vial, so its tolerance is the dE00 floor)")
    fa = total = fp = fp_total = 0
    confusable = {}
    for v in ok[ok.position_in_run >= 2].itertuples():
        refs = build_references(vials, catalog, exclude_vial=v.vial_id, min_vials=1)
        lab = np.array([v.L, v.a, v.b], float)
        _, verdict = check_vial(lab, v.recipe_id, refs)
        if verdict in ("consistent", "OUT OF RANGE"):
            total += 1
            fa += verdict == "OUT OF RANGE"
        for rid in catalog.index:
            if rid == v.recipe_id or refs.loc[rid, "status"] != "ok":
                continue
            fp_total += 1
            if check_vial(lab, rid, refs)[1] == "consistent":
                fp += 1
                confusable.setdefault(int(v.recipe_id), set()).add(int(rid))
    out.append(f"  false alarms (settled vial vs its own recipe): {fa}/{total}")
    out.append(f"  false passes (settled vial accepted as a different recipe): {fp}/{fp_total}"
               f" = {fp / max(fp_total, 1):.0%}")
    if confusable:
        out.append("  confusable: " + "; ".join(
            f"recipe {k} also passes as {sorted(v)}" for k, v in sorted(confusable.items())))
    full = build_references(vials, catalog)
    out.append("  reference tolerances (dE00): " + "; ".join(
        f"r{int(r)}={x.tolerance_de:.1f}" if x.status == "ok" else f"r{int(r)}=insufficient data"
        for r, x in full.iterrows()))

    out.append("\n3. Can we identify changeover effects?")
    for c in report[report.after_recipe_change].itertuples():
        run = report[report.recipe_id == c.recipe_id]
        ds = ", ".join("n/a" if pd.isna(d) else f"{d:.1f}" for d in run.delta_e00_to_recipe)
        out.append(f"  {int(c.previous_recipe_id)} -> {int(c.recipe_id)}: dE00 of vials 1.."
                   f"{len(run)} = [{ds}]  {list(run.verdict)}")
    out.append("\nCaveats: each recipe has only 3 vials and vial 1 is excluded from references. "
               "There are no independent good/bad outcome labels, so a 'false alarm' means "
               "'inconsistent with the other vials of the same recipe'.")
    return "\n".join(out)
