"""Focused tests for the eval-only comparison logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval import compare as compare_module
from eval.compare import (
    LEGACY_METHOD,
    RECIPE_METHOD,
    _threshold_context,
    _validate_measurements,
    build_comparison,
    load_or_measure,
    summarize,
    write_svg,
)


def _example_vials() -> pd.DataFrame:
    """Build two recipes where carryover exactly resembles a known wrong recipe."""

    rows = [
        # Recipe 1's third vial establishes a dark clean reference.
        (3, 1, np.nan, 3, 10.0, 0.0, 0.0, "ok"),
        # Recipe 2 first looks exactly like recipe 1, then returns to recipe 2.
        (4, 2, 1, 1, 10.0, 0.0, 0.0, "ok"),
        (5, 2, 2, 2, 50.0, 0.0, 0.0, "ok"),
        (6, 2, 2, 3, 50.0, 0.0, 0.0, "ok"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "vial_id",
            "recipe_id",
            "previous_recipe_id",
            "position_in_run",
            "L",
            "a",
            "b",
            "status",
        ],
    ).assign(spread_de=0.0, stable_n_frames=5)


def _complete_measurements() -> pd.DataFrame:
    """Build the complete 30-row shape required by the real cache contract."""

    rows = []
    vial_id = 1
    for recipe_id in range(5, 15):
        for position in (1, 2, 3):
            previous_recipe = np.nan if vial_id == 1 else recipe_id
            if position == 1 and recipe_id > 5:
                previous_recipe = recipe_id - 1
            start_s = float(vial_id * 40)
            rows.append(
                {
                    "vial_id": vial_id,
                    "recipe_id": recipe_id,
                    "previous_recipe_id": previous_recipe,
                    "position_in_run": position,
                    "cycle": vial_id,
                    "start_s": start_s,
                    "t_start": start_s + 2.123456789,
                    "label_lag_s": 2.123456789,
                    "frames_available": 20,
                    "L": recipe_id * 3.123456789,
                    "a": float(position),
                    "b": 0.0,
                    "spread_de": 0.25,
                    "stable_n_frames": 5,
                    "status": "ok",
                }
            )
            vial_id += 1
    return pd.DataFrame(rows)


def test_recipe_context_catches_a_known_but_wrong_color() -> None:
    """Only intended-recipe matching should reject a known wrong color."""

    comparison, excluded, _ = build_comparison(_example_vials(), threshold=10.0)
    carryover = comparison[comparison["expected_carryover"]].iloc[0]

    assert excluded.empty
    assert carryover["nearest_known_recipe_id"] == 1
    assert carryover["legacy_decision"] == "PASS"
    assert carryover["recipe_aware_decision"] == "FLAG"


def test_summary_uses_the_same_cases_for_both_methods() -> None:
    """Paired scoring must not improve one method by changing its test set."""

    comparison, _, _ = build_comparison(_example_vials(), threshold=10.0)
    summary = summarize(comparison, threshold=10.0).set_index("method")

    assert summary.loc[LEGACY_METHOD, "scored_vials"] == 2
    assert summary.loc[RECIPE_METHOD, "scored_vials"] == 2
    assert summary.loc[LEGACY_METHOD, "carryover_recall"] == 0.0
    assert summary.loc[RECIPE_METHOD, "carryover_recall"] == 1.0
    assert summary.loc[LEGACY_METHOD, "false_alarm_rate"] == 0.0
    assert summary.loc[RECIPE_METHOD, "false_alarm_rate"] == 0.0


def test_missing_last_vial_reference_is_reported_not_silently_scored() -> None:
    """An uncertain reference should lower coverage instead of becoming a guess."""

    vials = _example_vials()
    vials.loc[vials["vial_id"] == 6, ["L", "a", "b"]] = np.nan
    vials.loc[vials["vial_id"] == 6, "status"] = "color measurement uncertain"

    comparison, excluded, references = build_comparison(vials, threshold=10.0)

    assert comparison.empty
    assert set(excluded["vial_id"]) == {4, 5}
    value = references.loc[references["recipe_id"] == 2, "usable_reference"].item()
    assert not bool(value)


def test_measurement_cache_rejects_truncated_data() -> None:
    """Coverage must not improve silently when a cached vial disappears."""

    measurements = _complete_measurements()
    _validate_measurements(measurements)

    with pytest.raises(ValueError, match="vial 1-30"):
        _validate_measurements(measurements.iloc[:-1])


def test_fresh_and_cached_measurements_are_identical(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisted precision must be the exact precision used for scoring."""

    measured = _complete_measurements()
    fingerprint = {"value": "source-a"}
    measure_calls = {"count": 0}

    def fake_measure_video(_data_root):
        measure_calls["count"] += 1
        return measured.copy()

    # Isolate cache behavior from the large real video while retaining the
    # production validation, rounding, metadata, and reload path.
    monkeypatch.setattr(compare_module, "_configure_monitor", lambda _root: None)
    monkeypatch.setattr(
        compare_module,
        "_cache_metadata",
        lambda _root: {"fingerprint": fingerprint["value"]},
    )
    monkeypatch.setattr(compare_module, "measure_video", fake_measure_video)
    cache_path = tmp_path / "vial_measurements.csv"

    fresh = load_or_measure(tmp_path, cache_path, refresh=True)
    cached = load_or_measure(tmp_path, cache_path, refresh=False)
    pd.testing.assert_frame_equal(fresh, cached)
    assert measure_calls["count"] == 1
    assert cache_path.with_suffix(".meta.json").is_file()

    # A provenance change must force a fresh measurement instead of silently
    # scoring a cache created from different data, code, or settings.
    fingerprint["value"] = "source-b"
    load_or_measure(tmp_path, cache_path, refresh=False)
    assert measure_calls["count"] == 2

    # A structurally valid numeric edit must also invalidate the cache hash.
    edited = pd.read_csv(cache_path)
    edited.loc[0, "L"] += 1
    edited.to_csv(cache_path, index=False)
    load_or_measure(tmp_path, cache_path, refresh=False)
    assert measure_calls["count"] == 3


def test_svg_labels_unavailable_metrics_as_not_available(tmp_path) -> None:
    """A missing metric must not be drawn and labeled as a real zero."""

    comparison, _, _ = build_comparison(_example_vials(), threshold=10.0)
    summary = summarize(comparison, threshold=10.0)
    summary.loc[0, "false_alarm_rate"] = np.nan
    output = tmp_path / "comparison.svg"

    write_svg(summary, output)

    assert "n/a" in output.read_text(encoding="utf-8")


def test_custom_threshold_is_not_described_as_the_shared_default() -> None:
    """Generated claims must distinguish the contract value from an override."""

    default = float(compare_module.COLOR_WARN_DE)
    default_label, default_note = _threshold_context(default)
    custom_label, custom_note = _threshold_context(default + 1)

    assert "shared project threshold" in default_label
    assert "inherits" in default_note
    assert custom_label == "the selected threshold"
    assert "caller-selected" in custom_note
    assert f"default remains ΔE00 = {default:g}" in custom_note
