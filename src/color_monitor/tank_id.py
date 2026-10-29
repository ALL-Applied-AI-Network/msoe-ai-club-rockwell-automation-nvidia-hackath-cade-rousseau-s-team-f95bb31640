"""Optional mode: identify the base color of each tank from single-tank vials."""
import numpy as np

from .color import de00

IDENTIFICATION_RECIPES = {
    "T100 only": {"t100_ml": 50, "t200_ml": 0, "t400_ml": 0, "vials": 3},
    "T200 only": {"t100_ml": 0, "t200_ml": 50, "t400_ml": 0, "vials": 3},
    "T400 only": {"t100_ml": 0, "t200_ml": 0, "t400_ml": 50, "vials": 3},
}


def identify_base_color(vial_labs, tolerance_de):
    """vial_labs: (n, 3) stable CIELAB colors of one tank's single-tank vials.

    tolerance_de has no default on purpose: set it from measured repeatability
    and the plant team's judgment."""
    lab = np.asarray(vial_labs, dtype=float).reshape(-1, 3)
    if len(lab) < 2:
        return {"base_lab": None, "spread_de": None, "accepted": False,
                "reason": "need at least 2 vials"}
    med = np.median(lab, axis=0)
    spread = float(de00(lab, med.reshape(1, 3)).max())
    ok = spread <= tolerance_de
    return {"base_lab": med, "spread_de": spread, "accepted": ok,
            "reason": "vials agree" if ok else
            "vials disagree: check path residue, lighting, capture timing"}
