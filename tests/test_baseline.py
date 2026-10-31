import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage.color import rgb2lab

from api import baselines
from api.main import app
from changeover_guard.baseline import TANK_IDS, analyze
from changeover_guard.contracts import TANKS

ROOT = Path(__file__).parents[1]


@pytest.fixture
def image_payload():
    # Controlled four-liquid scene with a few glare and hardware pixels.
    image = np.full((540, 1176, 3), 240, dtype=np.uint8)
    colors = [(200, 184, 98), (197, 59, 7), (120, 102, 12), (1, 29, 137)]
    for (left, right), rgb in zip(((.02, .19), (.30, .47), (.59, .76), (.89, .99)), colors):
        x1, x2 = int(left * 1176), int(right * 1176)
        image[360:460, x1:x2] = rgb[::-1]
        image[390:394, x1 + 30:x1 + 36] = 255
        image[410:414, x1 + 20:x1 + 26] = 0
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes(), colors


def test_four_tanks_have_standard_lab_and_correct_bgr_order(image_payload):
    payload, colors = image_payload
    result, _ = analyze(payload, "top.png")
    assert list(result["tanks"]) == list(TANK_IDS)
    for tank, rgb in zip(TANK_IDS, colors):
        record = result["tanks"][tank]
        assert record["rgb"] == list(rgb)
        np.testing.assert_allclose(record["lab"], rgb2lab(np.array(rgb).reshape(1, 1, 3) / 255).reshape(3))
        assert record["sample_pixels"] > 40
    assert result["tanks"]["T300"]["recipe_enabled"] is False
    assert {k for k, v in result["tanks"].items() if v["recipe_enabled"]} == set(TANKS)
    assert result["confirmed"] is False


@pytest.mark.parametrize("path", sorted((ROOT / "generated/images/top_view_tank_color").glob("*.png")))
def test_existing_top_view_images_detect_all_four_tanks(path):
    result, _ = analyze(path.read_bytes(), path.name)
    assert set(result["tanks"]) == set(TANK_IDS)
    targets = json.loads((path.parent / "ai_tank_colors.json").read_text())["images"]
    expected = next(record["requested_rgb"] for record in targets if record["filename"] == path.name)
    for tank, rgb in expected.items():
        # Generated target images are only approximate color fixtures.
        measured = result["tanks"][tank]["rgb"]
        assert np.linalg.norm(np.array(measured) - rgb) < 65


def test_manual_regions_allow_neutral_liquids():
    image = np.full((540, 1176, 3), 128, dtype=np.uint8)
    payload = cv2.imencode(".png", image)[1].tobytes()
    with pytest.raises(ValueError, match="locate liquid"):
        analyze(payload, "gray.png")
    boxes = {tank: [i * .25 + .02, .7, i * .25 + .20, .85] for i, tank in enumerate(TANK_IDS)}
    result, _ = analyze(payload, "gray.png", boxes)
    for record in result["tanks"].values():
        np.testing.assert_allclose(record["lab"], [53.585, 0, 0], atol=.01)


def test_rejects_bad_regions_and_images(image_payload):
    payload, _ = image_payload
    result, _ = analyze(payload, "top.png")
    regions = {tank: record["region"] for tank, record in result["tanks"].items()}
    with pytest.raises(ValueError, match="exactly one"):
        analyze(payload, "top.png", {})
    regions["T100"] = [-.1, 0, .1, .1]
    with pytest.raises(ValueError, match="inside"):
        analyze(payload, "top.png", regions)
    regions["T100"] = regions["T200"]
    with pytest.raises(ValueError, match="overlap"):
        analyze(payload, "top.png", regions)
    with pytest.raises(ValueError, match="decoded"):
        analyze(b"not an image", "top.png")


def test_rejects_featureless_colored_image():
    image = np.empty((540, 1176, 3), dtype=np.uint8)
    image[:] = (20, 80, 200)
    payload = cv2.imencode(".png", image)[1].tobytes()
    with pytest.raises(ValueError, match="featureless"):
        analyze(payload, "solid.png")


def test_api_preview_confirm_reload_and_image(image_payload, tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "BASELINE_ROOT", tmp_path)
    payload, _ = image_payload
    body = {"image_base64": base64.b64encode(payload).decode(), "source_name": "top.png"}
    with TestClient(app) as client:
        preview = client.post("/api/baselines/analyze", json=body)
        assert preview.status_code == 200
        assert list(tmp_path.iterdir()) == []  # A preview must not activate a baseline.
        assert client.post("/api/baselines", json=body).status_code == 422
        assert client.post("/api/baselines", json={**body, "confirmed": True}).status_code == 422
        body["regions"] = {tank: record["region"] for tank, record in preview.json()["tanks"].items()}
        response = client.post("/api/baselines", json={**body, "confirmed": True})
        assert response.status_code == 201
        saved = response.json()
        assert saved["confirmed"]
        assert saved["tanks"] == preview.json()["tanks"]
        assert client.get(f"/api/baselines/{saved['id']}").json() == saved
        image = client.get(saved["image_url"])
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert np.array_equal(cv2.imdecode(np.frombuffer(image.content, np.uint8), cv2.IMREAD_COLOR),
                              cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR))
        # Each confirmation creates an immutable snapshot, retaining the earlier one.
        second = client.post("/api/baselines", json={**body, "confirmed": True}).json()
        assert second["id"] != saved["id"]
        assert client.get(f"/api/baselines/{saved['id']}").json() == saved
        assert client.post("/api/baselines/analyze", json={**body, "image_base64": "!!!"}).status_code == 422
        assert client.get("/api/baselines/not-a-uuid").status_code == 422
        assert client.get("/api/baselines/00000000-0000-0000-0000-000000000000").status_code == 404
