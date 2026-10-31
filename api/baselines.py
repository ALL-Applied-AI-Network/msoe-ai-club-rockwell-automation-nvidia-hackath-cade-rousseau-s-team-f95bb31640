"""Baseline preview and immutable, explicitly confirmed baseline snapshots."""

import base64
import binascii
import json
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from changeover_guard.baseline import analyze

router = APIRouter(prefix="/api/baselines", tags=["Tank initialization"])
BASELINE_ROOT = Path(__file__).parents[1] / "outputs" / "baselines"


class BaselineImage(BaseModel):
    image_base64: str = Field(max_length=12_000_000)
    source_name: str = Field(min_length=1, max_length=200)
    regions: dict[str, tuple[float, float, float, float]] | None = None


class ConfirmBaseline(BaselineImage):
    confirmed: Literal[True]


def _analyze(body: BaselineImage):
    try:
        payload = base64.b64decode(body.image_base64, validate=True)
        return analyze(payload, body.source_name, body.regions)
    except (ValueError, binascii.Error, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/analyze")
def preview(body: BaselineImage):
    result, _ = _analyze(body)
    return result


@router.post("", status_code=201)
def save(body: ConfirmBaseline):
    if body.regions is None:
        raise HTTPException(422, "Review all four liquid regions before saving.")
    result, image = _analyze(body)
    baseline_id = str(uuid4())
    folder = BASELINE_ROOT / baseline_id
    folder.mkdir(parents=True)
    result.update(id=baseline_id, confirmed=True, image_url=f"/api/baselines/{baseline_id}/image")
    if not cv2.imwrite(str(folder / "image.png"), image):
        raise HTTPException(500, "Could not save the baseline image.")
    temporary = folder / "baseline.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(folder / "baseline.json")
    return result


@router.get("/{baseline_id}")
def load(baseline_id: UUID):
    path = BASELINE_ROOT / str(baseline_id) / "baseline.json"
    if not path.is_file():
        raise HTTPException(404, "Baseline not found. Select a tank image to initialize again.")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{baseline_id}/image")
def image(baseline_id: UUID):
    load(baseline_id)
    return FileResponse(BASELINE_ROOT / str(baseline_id) / "image.png", media_type="image/png")
