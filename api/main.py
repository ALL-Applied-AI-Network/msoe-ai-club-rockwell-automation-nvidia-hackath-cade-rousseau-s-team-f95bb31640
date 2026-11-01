"""Backend for the web UI: wraps the changeover_guard pipeline as JSON endpoints.

    uv run uvicorn api.main:app --reload     # http://localhost:8000/docs lists every endpoint

Inspection steps that are still stubs (NotImplementedError) come back with source "pending" and null
model fields, never with made-up values. Scheduling uses the empirical carryover-cost model in
changeover_guard.costs. Once web/out exists (bun run build), the UI is served at /.
"""

import json
from datetime import timedelta
from functools import cache
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api.baselines import router as baseline_router
from api.schemas import (
    FrameOut,
    InspectionOut,
    RecipeOut,
    ScheduleIn,
    ScheduleOut,
    TanksOut,
    Transition,
)
from changeover_guard import advise, measure, mixing
from changeover_guard.contracts import Recipe, TankReading
from changeover_guard.costs import build_transition_matrix
from changeover_guard.dataset import frames, vial_labels
from changeover_guard.optimizer import optimize_sequence

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
CALIBRATION = ROOT / "twin" / "calibration.json"
WEB_OUT = ROOT / "web" / "out"
CAMERAS = ("vial_fill", "tank_level")
LABELED_VIDEO = 5  # the only video with recipe labels
# Vials are inspected on the video at their labeled end time, as twin/calibrate.py does: the still-image
# labels (video5_frame_labels.csv) run about one vial behind the camera.
BEFORE_END_S = 0.6  # full vial, cart not moving yet
WORST_TRANSITION_WEIGHT = 0.5
MAX_QUEUE = 12                               # exact optimizer takes ~1 s at 12 batches, ~1 min at 16

app = FastAPI(title="Changeover Guard API")
app.include_router(baseline_router)


@cache
def _frames(video_id: int):
    if not (DATA / "vial_fill" / "image_index.csv").exists():
        raise HTTPException(
            503, "dataset not found: unzip it so data/vial_fill/image_index.csv exists"
        )
    df = frames(DATA, video_id)
    if df.empty:
        raise HTTPException(404, f"no frames for video {video_id}")
    return df


@cache
def _vials() -> dict[int, tuple[Recipe, float]]:
    """Labeled vials of video 5: recipe and the video time (s) of the full vial."""
    _frames(LABELED_VIDEO)  # same dataset check
    return {
        vial: (recipe, end_s - BEFORE_END_S)
        for vial, recipe, _, end_s in vial_labels(DATA)
    }


def _recipes_by_vial() -> dict[int, Recipe]:
    return {vial: recipe for vial, (recipe, _) in _vials().items()}


@cache
def _clean_rgb() -> dict[int, tuple]:
    return {
        int(k): tuple(v)
        for k, v in json.loads(CALIBRATION.read_text())["target_rgb"].items()
    }


def _recipe_out(r: Recipe) -> RecipeOut:
    return RecipeOut(
        recipe_id=r.recipe_id,
        ml=r.ml,
        total_ml=sum(r.ml.values()),
        clean_rgb=_clean_rgb().get(r.recipe_id),
    )


def _opt_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def _read(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise HTTPException(500, f"could not read {Path(path).name}")
    return image


def _video_frame(camera: str, t: float) -> np.ndarray:
    """BGR frame of camera's video 5 at t seconds."""
    cap = cv2.VideoCapture(
        str(DATA / camera / "videos" / f"{camera}_video_{LABELED_VIDEO}.mp4")
    )
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, f"no {camera} video frame at {t:.1f} s")
    return image


def _vial(vial: int) -> tuple[Recipe, float]:
    if vial not in _vials():
        raise HTTPException(404, f"no labeled vial {vial}")
    return _vials()[vial]


def _tank_reading(video_id: int, index: int) -> TankReading:
    row = _frames(video_id).iloc[index]
    return measure.tanks(_read(row.tank_level), row.timestamp.to_pydatetime())


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "data": (DATA / "vial_fill" / "image_index.csv").exists()}


@app.get("/api/recipes")
def recipes() -> list[RecipeOut]:
    """The recipes in the labeled video, with their measured clean colors."""
    unique = {r.recipe_id: r for r in _recipes_by_vial().values()}
    return [_recipe_out(r) for _, r in sorted(unique.items())]


@app.get("/api/videos/{video_id}/frames")
def video_frames(video_id: int) -> list[FrameOut]:
    """Every synchronized vial_fill + tank_level still of a video, with fill labels where they exist."""
    df = _frames(video_id)
    labels = df.reindex(
        columns=["in_fill", "vial", "recipe_id"]
    )  # all NaN for unlabeled videos
    return [
        FrameOut(
            index=i,
            timestamp=row.timestamp.isoformat(),
            vial_fill=Path(row.vial_fill).name,
            tank_level=Path(row.tank_level).name,
            in_fill=None if pd.isna(lab.in_fill) else bool(lab.in_fill),
            vial=_opt_int(lab.vial),
            recipe_id=_opt_int(lab.recipe_id),
        )
        for i, (row, lab) in enumerate(zip(df.itertuples(), labels.itertuples()))
    ]


@app.get("/api/images/{camera}/{filename}")
def image(camera: str, filename: str) -> FileResponse:
    if camera not in CAMERAS or Path(filename).name != filename:
        raise HTTPException(404, "unknown image")
    path = DATA / camera / "images" / filename
    if not path.is_file():
        raise HTTPException(404, "unknown image")
    return FileResponse(path)


@app.get("/api/videos/{video_id}/frames/{index}/tanks")
def tanks(video_id: int, index: int) -> TanksOut:
    """Tank colors and fill levels read from one tank_level frame."""
    if not 0 <= index < len(_frames(video_id)):
        raise HTTPException(404, f"no frame {index}")
    tank_image = _read(_frames(video_id).iloc[index].tank_level)
    return TanksOut(
        colors=_tank_reading(video_id, index).colors,
        fill_percent=measure.tank_fill_levels(tank_image),
    )


@app.get("/api/vials")
def vials() -> list[InspectionOut]:
    """Every labeled vial of video 5, inspected (about 1 s for all 30)."""
    return [inspection(vial) for vial in sorted(_vials())]


@app.get("/api/vials/{vial}/image")
def vial_image(vial: int) -> Response:
    """The vial_fill video frame of the full vial, as JPEG."""
    _, t = _vial(vial)
    _, jpeg = cv2.imencode(".jpg", _video_frame("vial_fill", t))
    return Response(jpeg.tobytes(), media_type="image/jpeg")


@app.get("/api/vials/{vial}/inspection")
def inspection(vial: int) -> InspectionOut:
    """Check one labeled vial of video 5: expected vs observed color and fill, likely cause, advice."""
    recipe, t = _vial(vial)
    prev = _recipes_by_vial().get(vial - 1)
    out = InspectionOut(
        vial=vial,
        recipe_id=recipe.recipe_id,
        prev_recipe_id=prev.recipe_id if prev else None,
        target_ml=sum(recipe.ml.values()),
        image_url=f"/api/vials/{vial}/image",
        expected_rgb=_clean_rgb().get(recipe.recipe_id),
        source="pending",
        status="PENDING",
    )
    # ponytail: video time on the first still's clock; only used as the measurement's timestamp
    timestamp = _frames(LABELED_VIDEO).timestamp.iloc[0].to_pydatetime() + timedelta(
        seconds=t
    )
    try:
        tank_reading = measure.tanks(_video_frame("tank_level", t), timestamp)
        measured = measure.vial(_video_frame("vial_fill", t), timestamp)
        prediction = mixing.predict(prev, recipe, tank_reading)
        check = advise.check(prediction, measured)
        queue = [r for v, r in sorted(_recipes_by_vial().items()) if v > vial]
        advice = advise.advise(prediction, check, queue, tank_reading)
    except NotImplementedError:
        return out
    return out.model_copy(
        update={
            "source": "model",
            "status": "PASS" if check.ok else "FAIL",
            "expected_rgb": prediction.clean_color,
            "observed_rgb": measured.color,
            "delta_e": check.color_delta_e,
            "fill_ml": measured.fill_ml,
            "cause": advice.cause,
            "next_recipe_id": advice.next_recipe_id,
        }
    )


@app.post("/api/schedule")
def schedule(body: ScheduleIn) -> ScheduleOut:
    """Evaluate a queue and optimize it while preserving its first batch."""
    known = {r.recipe_id: r for r in _recipes_by_vial().values()}
    missing = [rid for rid in body.recipe_ids if rid not in known]
    if missing:
        raise HTTPException(422, f"unknown recipes: {missing}")
    if not body.recipe_ids:
        raise HTTPException(422, "at least one recipe is required")
    if len(body.recipe_ids) > MAX_QUEUE:
        raise HTTPException(422, f"at most {MAX_QUEUE} batches per plan (exact search is exponential)")

    selected = [known[recipe_id] for recipe_id in body.recipe_ids]
    matrix = build_transition_matrix(selected)
    result = optimize_sequence(
        body.recipe_ids,
        matrix,
        worst_transition_weight=WORST_TRANSITION_WEIGHT,
    )
    warnings: list[str] = []

    def changeovers(order) -> list[Transition]:
        out = []
        for previous, target in pairwise(order):
            cost = matrix.transition(previous.recipe_id, target.recipe_id)
            out.append(
                Transition(
                    prev_batch_id=previous.batch_id,
                    batch_id=target.batch_id,
                    prev_recipe_id=previous.recipe_id,
                    recipe_id=target.recipe_id,
                    first_vial_rgb=cost.first_vial_rgb,
                    delta_e=cost.first_vial_delta_e,
                    warn=cost.warn,
                    cost=cost.total,
                    uncertainty=cost.uncertainty,
                )
            )
            warnings.extend(cost.warnings)
        return out

    transitions = changeovers(result.submitted)
    suggested_transitions = changeovers(result.suggested)

    return ScheduleOut(
        source="model",
        transitions=transitions,
        suggested_transitions=suggested_transitions,
        submitted=[batch.recipe_id for batch in result.submitted],
        suggested=[batch.recipe_id for batch in result.suggested],
        submitted_batch_ids=[batch.batch_id for batch in result.submitted],
        suggested_batch_ids=[batch.batch_id for batch in result.suggested],
        fixed_start_recipe_id=result.fixed_start.recipe_id,
        submitted_cost=result.submitted_cost,
        suggested_cost=result.suggested_cost,
        submitted_transition_sum=result.submitted_transition_sum,
        suggested_transition_sum=result.suggested_transition_sum,
        submitted_worst_transition=result.submitted_worst_transition,
        suggested_worst_transition=result.suggested_worst_transition,
        relative_reduction=result.relative_reduction,
        transition_sum_reduction=result.transition_sum_reduction,
        worst_transition_reduction=result.worst_transition_reduction,
        worst_transition_weight=result.worst_transition_weight,
        method=result.method,
        warnings=list(
            dict.fromkeys(
                [
                    *warnings,
                    f"Objective includes a {WORST_TRANSITION_WEIGHT:g}x penalty on the worst individual transition.",
                ]
            )
        ),
    )


if WEB_OUT.is_dir():
    app.mount("/", StaticFiles(directory=WEB_OUT, html=True), name="web")
