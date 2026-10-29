"""Turn a model's inspection_result.json into the fields the operator screen shows."""

import json
from pathlib import Path

TEXT_FIELDS = ("vial", "recipe", "previous", "expected", "observed", "status", "cause", "action", "fill")
COLOR_FIELDS = ("expected_rgb", "observed_rgb")

# Missing fields show as blanks, never as demo text, so an operator cannot
# mistake an illustrative cause for the model's explanation.
BLANK = dict.fromkeys(TEXT_FIELDS, "—") | {
    "cause": "No cause provided by the model.",
    "action": "No action provided by the model.",
    "fill": "Not measured",
    "expected_hex": "#ffffff",
    "observed_hex": "#ffffff",
}


def rgb_to_hex(rgb) -> str:
    """Convert an sRGB 0-255 triple (the contracts.py color format) to #rrggbb."""
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError(f"color must be a list of 3 numbers 0-255, got {rgb!r}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rgb):
        raise ValueError(f"color must be a list of 3 numbers 0-255, got {rgb!r}")
    return "#" + "".join(f"{max(0, min(255, round(v))):02x}" for v in rgb)


def load_result(path: Path) -> tuple[dict, list[str]]:
    """Return the display fields and any JSON keys the screen does not use.

    Raises OSError, ValueError or TypeError when the file is missing or malformed.
    """
    incoming = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(incoming, dict):
        raise TypeError("JSON root must be an object")
    result = BLANK | {k: incoming[k] for k in TEXT_FIELDS if k in incoming}
    for field in COLOR_FIELDS:
        if field in incoming:
            result[field.replace("_rgb", "_hex")] = rgb_to_hex(incoming[field])
    ignored = sorted(set(incoming) - set(TEXT_FIELDS) - set(COLOR_FIELDS))
    return result, ignored
