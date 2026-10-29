"""Check that the operator screen shows model JSON as-is, never mixed with demo text."""

import json
from pathlib import Path

import pytest

from dashboard.operator_screen_prototype.operator_result import BLANK, load_result


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "inspection_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rgb_lists_become_swatch_colors(tmp_path: Path):
    # contracts.py colors are sRGB 0-255 tuples; the screen's swatches need hex.
    result, ignored = load_result(_write(tmp_path, {"expected_rgb": [45, 125, 192], "observed_rgb": [108, 152, 102.4]}))
    assert (result["expected_hex"], result["observed_hex"]) == ("#2d7dc0", "#6c9866")
    assert ignored == []


def test_missing_fields_are_blank_and_unknown_keys_reported(tmp_path: Path):
    # A partial result must not inherit a demo cause such as "orange residue".
    result, ignored = load_result(_write(tmp_path, {"status": "PASS", "expected_hex": "#000000"}))
    assert result["status"] == "PASS"
    assert result["cause"] == BLANK["cause"]
    assert result["expected_hex"] == BLANK["expected_hex"]
    assert ignored == ["expected_hex"]


@pytest.mark.parametrize("payload", [[1, 2, 3], {"observed_rgb": "#ff0000"}, {"observed_rgb": [1, 2]}, {"observed_rgb": ["a", 2, 3]}])
def test_malformed_json_raises_caught_error(tmp_path: Path, payload):
    # The screen catches these and falls back with a warning instead of crashing.
    with pytest.raises((ValueError, TypeError)):
        load_result(_write(tmp_path, payload))
