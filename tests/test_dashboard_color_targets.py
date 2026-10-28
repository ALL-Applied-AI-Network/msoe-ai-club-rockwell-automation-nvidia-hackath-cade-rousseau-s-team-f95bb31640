"""Check that dashboard color references follow edits to their JSON files."""

import json
from pathlib import Path

from dashboard.color_targets import load_ai_color_targets

ROOT = Path(__file__).parents[1] / "generated" / "images"


def test_each_color_image_set_uses_its_current_manifest():
    # Both image sets reuse filenames; their RGB values must come from the
    # matching manifest, including any values edited after image generation.
    for image_set in ("top_view_tank_color", "front_view_tank_color"):
        manifest_path = ROOT / image_set / "ai_tank_colors.json"
        records = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
        assert load_ai_color_targets(manifest_path) == {
            record["filename"]: record["requested_rgb"] for record in records
        }


def test_color_targets_refresh_after_edit_to_same_manifest(tmp_path: Path):
    # Reproduce the live-page bug: the path stays fixed while requested RGBs
    # change. The second read must show the new value without clearing cache.
    manifest_path = tmp_path / "ai_tank_colors.json"
    record = {"filename": "ai_tank_variant_01.png", "requested_rgb": {"T100": [1, 2, 3]}}
    manifest_path.write_text(json.dumps({"images": [record]}), encoding="utf-8")
    assert load_ai_color_targets(manifest_path)[record["filename"]]["T100"] == [1, 2, 3]

    record["requested_rgb"]["T100"] = [123, 196, 195]
    manifest_path.write_text(json.dumps({"images": [record]}), encoding="utf-8")
    assert load_ai_color_targets(manifest_path)[record["filename"]]["T100"] == [123, 196, 195]
