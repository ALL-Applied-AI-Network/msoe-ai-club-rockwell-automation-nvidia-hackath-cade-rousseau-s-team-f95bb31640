"""JSON shapes the web UI reads. Colors are sRGB 0-255 lists, as in contracts.py."""
from typing import Literal

from pydantic import BaseModel

RGB = tuple[float, float, float]
# "model": computed by the pipeline. "pending": the model code isn't in yet, so model fields are null.
Source = Literal["model", "pending"]


class RecipeOut(BaseModel):
    recipe_id: int
    ml: dict[str, float]                     # only tanks in use, e.g. {"T400": 50, "T100": 20}
    total_ml: float
    clean_rgb: RGB | None                    # measured clean-vial color (twin/calibration.json), if known


class FrameOut(BaseModel):
    index: int
    timestamp: str
    vial_fill: str                           # image filenames, served by /api/images/{camera}/{filename}
    tank_level: str
    in_fill: bool | None
    vial: int | None
    recipe_id: int | None


class TanksOut(BaseModel):
    colors: dict[str, RGB]                   # measure.tanks()
    fill_percent: dict[str, float | None]    # measure.tank_fill_levels(); None when no liquid line is visible


class InspectionOut(BaseModel):
    vial: int
    recipe_id: int
    prev_recipe_id: int | None
    target_ml: float
    image_url: str                           # the full vial, from the vial_fill video
    expected_rgb: RGB | None
    source: Source
    status: Literal["PASS", "FAIL", "PENDING"]
    observed_rgb: RGB | None = None
    delta_e: float | None = None
    fill_ml: float | None = None
    cause: str | None = None
    next_recipe_id: int | None = None


class Transition(BaseModel):
    prev_recipe_id: int
    recipe_id: int
    first_vial_rgb: RGB | None = None
    delta_e: float | None = None             # predicted first-vial color difference
    warn: bool | None = None


class ScheduleIn(BaseModel):
    recipe_ids: list[int]                    # in the order the operator would run them


class ScheduleOut(BaseModel):
    source: Source
    transitions: list[Transition]            # predicted carryover for each changeover in the given order
    suggested: list[int] | None = None       # lower-carryover order, once the scheduler exists
