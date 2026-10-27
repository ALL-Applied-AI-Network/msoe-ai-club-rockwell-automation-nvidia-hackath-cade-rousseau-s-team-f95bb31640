"""Fixed recipe catalog and per-recipe color references."""
import numpy as np
import pandas as pd

from . import config as C
from .color import de00


class UnknownRecipe(ValueError):
    pass


def eligible_reference_vials(vials):
    """Settled vials away from a recipe transition: measured, swapped out
    afterwards, and not the first vial of a recipe run."""
    return vials[(vials.status == "ok") & vials.closed.fillna(False).astype(bool)
                 & (vials.position_in_run >= 2)]


def build_references(vials, catalog, exclude_vial=None, min_vials=C.MIN_REFERENCE_VIALS):
    """Empirical reference per catalog recipe from eligible vials.

    Returns DataFrame indexed by recipe_id: L, a, b, tolerance_de, n_vials,
    status ('ok' or 'insufficient data'). A one-vial reference has no measured
    spread, so its tolerance is just the floor; only the evaluation uses that."""
    ref = eligible_reference_vials(vials)
    if exclude_vial is not None:
        ref = ref[ref.vial_id != exclude_vial]
    rows = []
    for rid in catalog.index:
        g = ref[ref.recipe_id == rid]
        row = {"recipe_id": rid, "n_vials": len(g), "L": np.nan, "a": np.nan, "b": np.nan,
               "tolerance_de": np.nan, "status": "insufficient data"}
        if len(g) >= min_vials:
            lab = g[["L", "a", "b"]].to_numpy(float)
            med = np.median(lab, axis=0)
            spread = de00(lab, med.reshape(1, 3)).max()
            row.update(L=med[0], a=med[1], b=med[2], status="ok",
                       tolerance_de=max(C.TOLERANCE_FLOOR_DE, C.TOLERANCE_MARGIN * spread))
        rows.append(row)
    return pd.DataFrame(rows).set_index("recipe_id")


def check_vial(lab, recipe_id, refs):
    """Compare a measured color with the reference of its intended recipe only.
    Returns (delta_e or None, verdict)."""
    if recipe_id not in refs.index:
        raise UnknownRecipe(f"recipe {recipe_id} is not in the fixed catalog")
    r = refs.loc[recipe_id]
    if r.status != "ok":
        return None, "no reference (insufficient data)"
    if np.isnan(lab).any():
        return None, "color measurement uncertain"
    d = float(de00(lab, r[["L", "a", "b"]].to_numpy(float))[0])
    return d, ("consistent" if d <= r.tolerance_de else "OUT OF RANGE")
