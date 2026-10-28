"""Load editable color references for the dashboard's generated tank images."""

import json
from pathlib import Path


def load_ai_color_targets(manifest_path: Path) -> dict[str, dict[str, list[int]]]:
    """Read current RGB requests, indexed by image filename."""
    if not manifest_path.exists():
        return {}
    # These small manifests may be edited while Streamlit is running. Reading
    # on each rerun prevents the old requested_rgb values from remaining in UI.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["filename"]: record["requested_rgb"] for record in manifest["images"]}
