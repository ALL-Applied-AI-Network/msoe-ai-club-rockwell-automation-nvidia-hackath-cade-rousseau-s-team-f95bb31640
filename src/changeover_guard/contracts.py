from dataclasses import dataclass
from datetime import datetime

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

TANKS = ("T100", "T200", "T400")             # T300 out of service, never used
DISPENSE_ORDER = ("T200", "T400", "T100")
MAX_VIAL_ML, MIN_COMPONENT_ML, MAX_T200_ML, QUEUE_SIZE = 80, 10, 50, 3

# Starting thresholds; tuned on synthetic data in week 2
COLOR_WARN_DE = 10.0        # CIEDE2000 distance that counts as "off color"
FILL_TOL_ML = 5.0
# Tank-camera fill readings are percentages of tank height, so they need a
# separate tolerance from the vial-volume tolerance above.
TANK_FILL_WARN_PERCENT = 5.0

RGB = tuple[float, float, float]             # sRGB, 0-255


@dataclass(frozen=True)
class Recipe:
    recipe_id: int
    ml: dict[str, float]                     # only tanks in use, e.g. {"T200": 10, "T100": 30}

    def validate(self) -> None:
        for tank, v in self.ml.items():
            if tank not in TANKS:
                raise ValueError(f"{tank}: not a usable tank")
            if v < MIN_COMPONENT_ML or v > MAX_VIAL_ML:
                raise ValueError(f"{tank}: {v} mL outside {MIN_COMPONENT_ML}-{MAX_VIAL_ML}")
        if self.ml.get("T200", 0) > MAX_T200_ML:
            raise ValueError(f"T200: {self.ml['T200']} mL above {MAX_T200_ML}")
        if sum(self.ml.values()) > MAX_VIAL_ML:
            raise ValueError(f"total {sum(self.ml.values())} mL above {MAX_VIAL_ML}")


@dataclass
class TankReading:          # measure -> mixing
    timestamp: datetime
    colors: dict[str, RGB]


@dataclass
class VialMeasurement:      # measure -> advise
    timestamp: datetime
    color: RGB
    fill_ml: float


@dataclass
class Prediction:           # mixing -> advise, dashboard
    recipe: Recipe
    prev_recipe: Recipe | None
    clean_color: RGB
    first_vial_color: RGB
    delta_e: float
    warn: bool


@dataclass
class Check:                # advise -> dashboard, eval
    color_delta_e: float
    fill_error_ml: float
    ok: bool


@dataclass
class Advice:               # advise -> dashboard
    cause: str | None       # e.g. "leftover orange from Recipe 6"
    flush: bool
    next_recipe_id: int | None


def delta_e(a: RGB, b: RGB) -> float:    # CIEDE2000 via scikit-image
    lab = [rgb2lab(np.asarray(c, dtype=float).reshape(1, 1, 3) / 255) for c in (a, b)]
    return float(deltaE_ciede2000(*lab).item())


def color_difference(actual: RGB, measured: RGB) -> tuple[float, bool]:
    """Return the color difference and whether it reaches the shared warning limit."""
    # Use one contract threshold for dashboard warnings and tests. A difference
    # exactly on the boundary is off-color, matching the existing UI behavior.
    difference = delta_e(actual, measured)
    return difference, difference >= COLOR_WARN_DE
