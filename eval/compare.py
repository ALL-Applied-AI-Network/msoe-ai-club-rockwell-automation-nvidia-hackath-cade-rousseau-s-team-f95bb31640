"""Compare known-color inspection with intended-recipe inspection.

This module intentionally lives in ``eval/`` and does not alter the production
pipeline. Both methods receive the same measured vial colors and clean-color
references; only the decision rule changes. That isolates the value of recipe
context without pretending that the unfinished prediction/advice pipeline has
already been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from changeover_guard.contracts import COLOR_WARN_DE
from color_monitor import config as monitor_config
from color_monitor.audit import build_vial_table
from color_monitor.color import de00
from color_monitor.extract import extract_frames
from color_monitor.segment import measure_cycles

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
VIDEO_ID = 5
REFERENCE_POSITION = 3
CACHE_SCHEMA_VERSION = 1

# The shared project threshold is the primary result. The smaller values show
# whether the old/new ranking survives stricter fixed reporting points instead
# of presenting only the most favorable operating point.
SENSITIVITY_THRESHOLDS = (2.3, 5.0, float(COLOR_WARN_DE))

LEGACY_METHOD = "Legacy: any known color"
RECIPE_METHOD = "Recipe-aware: intended recipe"
LAB_COLUMNS = ["L", "a", "b"]
MEASUREMENT_COLUMNS = [
    "vial_id",
    "recipe_id",
    "previous_recipe_id",
    "position_in_run",
    "cycle",
    "start_s",
    "t_start",
    "label_lag_s",
    "frames_available",
    "L",
    "a",
    "b",
    "spread_de",
    "stable_n_frames",
    "status",
]

# Changes to any of these measurement modules invalidate the derived cache.
# Hashing the code avoids silently comparing old measurements after an ROI or
# cycle-segmentation edit elsewhere in the repository.
MEASUREMENT_CODE_FILES = (
    Path(__file__).resolve(),
    REPO_ROOT / "src" / "color_monitor" / "config.py",
    REPO_ROOT / "src" / "color_monitor" / "color.py",
    REPO_ROOT / "src" / "color_monitor" / "extract.py",
    REPO_ROOT / "src" / "color_monitor" / "segment.py",
    REPO_ROOT / "src" / "color_monitor" / "audit.py",
)


def _configure_monitor(data_root: Path) -> None:
    """Point the existing video measurement code at the caller's data root.

    The color monitor currently contains a stale default path. Reassigning its
    path constants here keeps this compatibility fix local to the evaluation
    runner, as requested, and leaves production code unchanged.
    """

    vial_fill = data_root / "vial_fill"
    required = [
        vial_fill / "videos" / f"vial_fill_video_{VIDEO_ID}.mp4",
        vial_fill / "labels" / "video5_vial_recipes.csv",
        vial_fill / "labels" / "video5_frame_labels.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Evaluation data is incomplete; missing:\n  - {formatted}"
        )

    monitor_config.DATASET = data_root
    monitor_config.VIAL_FILL = vial_fill
    monitor_config.IMAGES = vial_fill / "images"
    monitor_config.IMAGE_INDEX = vial_fill / "image_index.csv"
    monitor_config.VIAL_RECIPES = vial_fill / "labels" / "video5_vial_recipes.csv"
    monitor_config.FRAME_LABELS = vial_fill / "labels" / "video5_frame_labels.csv"


def _sha256(path: Path) -> str:
    """Hash a source file in bounded chunks for cache provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata(data_root: Path) -> dict[str, object]:
    """Describe every input and setting that can affect vial measurements."""

    source_files = {
        "vial_fill_video_5.mp4": data_root
        / "vial_fill"
        / "videos"
        / "vial_fill_video_5.mp4",
        "video5_vial_recipes.csv": data_root
        / "vial_fill"
        / "labels"
        / "video5_vial_recipes.csv",
        "video5_frame_labels.csv": data_root
        / "vial_fill"
        / "labels"
        / "video5_frame_labels.csv",
    }
    payload: dict[str, object] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "video_id": VIDEO_ID,
        "inputs": {
            name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in source_files.items()
        },
        "measurement_code": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _sha256(path)
            for path in MEASUREMENT_CODE_FILES
        },
        "dependencies": {
            package: version(package)
            for package in ("numpy", "pandas", "opencv-python-headless", "scikit-image")
        },
        "settings": {
            "liquid_roi": list(monitor_config.LIQUID_ROI),
            "holder_roi": list(monitor_config.HOLDER_ROI),
            "min_l": monitor_config.MIN_L,
            "max_l": monitor_config.MAX_L,
            "min_valid_pixels": monitor_config.MIN_VALID_PIXELS,
            "holder_dark_l": monitor_config.HOLDER_DARK_L,
            "holder_min_dark_frac": monitor_config.HOLDER_MIN_DARK_FRAC,
            "sample_stride_s": monitor_config.SAMPLE_STRIDE_S,
            "stable_max_step_de": monitor_config.STABLE_MAX_STEP_DE,
            "stable_min_frames": monitor_config.STABLE_MIN_FRAMES,
            "empty_min_l": monitor_config.EMPTY_MIN_L,
            "empty_min_frames": monitor_config.EMPTY_MIN_FRAMES,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _validate_measurements(vials: pd.DataFrame) -> None:
    """Reject stale, truncated, duplicated, or structurally invalid caches."""

    missing_columns = sorted(set(MEASUREMENT_COLUMNS) - set(vials.columns))
    if missing_columns:
        raise ValueError(
            f"Measurement cache is missing columns: {', '.join(missing_columns)}"
        )

    vial_ids = pd.to_numeric(vials["vial_id"], errors="coerce")
    expected_vials = set(range(1, 31))
    if (
        len(vials) != 30
        or vial_ids.isna().any()
        or set(vial_ids.astype(int)) != expected_vials
    ):
        raise ValueError(
            "Measurement cache must contain each labeled vial 1-30 exactly once"
        )
    if vial_ids.duplicated().any():
        raise ValueError("Measurement cache contains duplicate vial IDs")

    cycles = pd.to_numeric(vials["cycle"], errors="coerce")
    if cycles.isna().any() or set(cycles.astype(int)) != expected_vials:
        raise ValueError(
            "Measurement cache must contain each observed cycle 1-30 exactly once"
        )

    recipe_ids = pd.to_numeric(vials["recipe_id"], errors="coerce")
    expected_recipes = set(range(5, 15))
    if recipe_ids.isna().any() or set(recipe_ids.astype(int)) != expected_recipes:
        raise ValueError("Measurement cache must contain labeled recipes 5-14")
    positions = pd.to_numeric(vials["position_in_run"], errors="coerce")
    recipe_positions = pd.DataFrame(
        {"recipe_id": recipe_ids.astype(int), "position": positions}
    )
    for recipe_id, group in recipe_positions.groupby("recipe_id"):
        if len(group) != 3 or set(group["position"].astype(int)) != {1, 2, 3}:
            raise ValueError(
                f"Recipe {recipe_id} must contain positions 1, 2, and 3 exactly once"
            )


def measure_video(data_root: Path) -> pd.DataFrame:
    """Measure one stable CIELAB color per labeled vial in real video 5."""

    _configure_monitor(data_root.resolve())
    # The existing monitor samples at one frame per second, segments physical
    # vial cycles, and uses the last sustained stable-color window per cycle.
    frames = extract_frames((VIDEO_ID,))
    cycles = measure_cycles(frames)
    vials = build_vial_table(cycles)
    return vials[MEASUREMENT_COLUMNS].copy()


def load_or_measure(data_root: Path, cache_path: Path, refresh: bool) -> pd.DataFrame:
    """Reuse measurements only when source data, code, and settings still match."""

    data_root = data_root.resolve()
    _configure_monitor(data_root)
    current_metadata = _cache_metadata(data_root)
    metadata_path = cache_path.with_suffix(".meta.json")
    cached_metadata: dict[str, object] | None = None
    if metadata_path.is_file():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached_metadata = None

    cache_matches = (
        cache_path.is_file()
        and cached_metadata is not None
        and cached_metadata.get("fingerprint") == current_metadata["fingerprint"]
        and cached_metadata.get("cache_sha256") == _sha256(cache_path)
    )
    if cache_matches and not refresh:
        vials = pd.read_csv(cache_path)
    else:
        if cache_path.is_file() and not refresh:
            print("Measurement cache provenance changed; refreshing video 5.")
        vials = measure_video(data_root)
        _validate_measurements(vials)
        # Score the exact six-decimal values that are persisted so a fresh run
        # and a later cached run produce byte-identical evidence artifacts.
        numeric_columns = vials.select_dtypes(include=[np.number]).columns
        vials.loc[:, numeric_columns] = vials[numeric_columns].round(6)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vials.to_csv(cache_path, index=False, float_format="%.6f")
        persisted_metadata = {
            **current_metadata,
            "cache_sha256": _sha256(cache_path),
        }
        metadata_path.write_text(
            json.dumps(persisted_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        vials = pd.read_csv(cache_path)

    _validate_measurements(vials)
    return vials


def _expected_cases(vials: pd.DataFrame) -> pd.DataFrame:
    """Return the 19 cases defined before looking at measured color.

    The third vial supplies the clean reference and is never scored. Vial two
    is the clean proxy for every recipe. A first vial is a carryover-risk proxy
    only when its previous recipe is known, which excludes the recording's very
    first vial.
    """

    position = pd.to_numeric(vials["position_in_run"])
    recipe = pd.to_numeric(vials["recipe_id"])
    previous = pd.to_numeric(vials["previous_recipe_id"], errors="coerce")
    known_transition = (position == 1) & previous.notna() & (previous != recipe)
    clean_proxy = position == 2
    return vials[known_transition | clean_proxy].copy()


def build_comparison(
    vials: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create paired old/new decisions, exclusions, and reference metadata."""

    last_vials = vials[
        pd.to_numeric(vials["position_in_run"]) == REFERENCE_POSITION
    ].copy()
    last_vials["usable_reference"] = last_vials["status"].eq("ok") & last_vials[
        LAB_COLUMNS
    ].notna().all(axis=1)
    last_vials["reference_note"] = np.where(
        last_vials["usable_reference"],
        "used as clean-color reference",
        "not used: stable color measurement unavailable",
    )
    references = last_vials[last_vials["usable_reference"]].copy()
    references["recipe_id"] = pd.to_numeric(references["recipe_id"]).astype(int)
    references = references.sort_values("recipe_id").set_index("recipe_id", drop=False)

    if references.empty:
        raise ValueError("No usable third-vial references were measured")

    reference_labs = references[LAB_COLUMNS].to_numpy(float)
    reference_recipes = references["recipe_id"].to_numpy(int)
    cases: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []

    for vial in _expected_cases(vials).sort_values("vial_id").itertuples(index=False):
        recipe_id = int(vial.recipe_id)
        position = int(vial.position_in_run)
        previous_recipe = (
            None if pd.isna(vial.previous_recipe_id) else int(vial.previous_recipe_id)
        )
        expected_carryover = position == 1
        expected_outcome = (
            "carryover-risk proxy" if expected_carryover else "clean proxy"
        )

        exclusion_reason = None
        observed_lab = np.asarray([vial.L, vial.a, vial.b], dtype=float)
        if vial.status != "ok" or np.isnan(observed_lab).any():
            exclusion_reason = f"vial color unavailable ({vial.status})"
        elif recipe_id not in references.index:
            reference_row = last_vials[
                pd.to_numeric(last_vials["recipe_id"]).astype(int) == recipe_id
            ]
            reference_status = (
                str(reference_row.iloc[0]["status"])
                if not reference_row.empty
                else "missing"
            )
            exclusion_reason = (
                "intended recipe has no usable last-vial reference "
                f"({reference_status})"
            )

        if exclusion_reason is not None:
            exclusions.append(
                {
                    "vial_id": int(vial.vial_id),
                    "recipe_id": recipe_id,
                    "position_in_run": position,
                    "expected_outcome": expected_outcome,
                    "reason": exclusion_reason,
                }
            )
            continue

        # The legacy emulation asks only whether this resembles any database
        # color. The recipe-aware rule uses the same references but restricts
        # the comparison to the recipe that was actually scheduled.
        repeated_observation = np.repeat(
            observed_lab.reshape(1, 3), len(references), axis=0
        )
        all_distances = de00(repeated_observation, reference_labs)
        nearest_index = int(np.argmin(all_distances))
        nearest_recipe = int(reference_recipes[nearest_index])
        nearest_de = float(all_distances[nearest_index])

        intended_reference = references.loc[recipe_id]
        intended_lab = intended_reference[LAB_COLUMNS].to_numpy(float)
        intended_de = float(de00(observed_lab, intended_lab)[0])
        # Only position-one cases are actual recipe transitions; later vials
        # retain previous_recipe_id for traceability but should not be labeled
        # as transitions in the exported evidence.
        transition = (
            f"{previous_recipe}->{recipe_id}"
            if expected_carryover and previous_recipe is not None
            else ""
        )

        cases.append(
            {
                "vial_id": int(vial.vial_id),
                "recipe_id": recipe_id,
                "previous_recipe_id": previous_recipe,
                "transition": transition,
                "position_in_run": position,
                "expected_outcome": expected_outcome,
                "expected_carryover": expected_carryover,
                "measured_L": float(vial.L),
                "measured_a": float(vial.a),
                "measured_b": float(vial.b),
                "reference_vial_id": int(intended_reference["vial_id"]),
                "nearest_known_recipe_id": nearest_recipe,
                "nearest_known_delta_e": nearest_de,
                "intended_recipe_delta_e": intended_de,
                "legacy_flag": nearest_de >= threshold,
                "recipe_aware_flag": intended_de >= threshold,
                "legacy_decision": "FLAG" if nearest_de >= threshold else "PASS",
                "recipe_aware_decision": "FLAG" if intended_de >= threshold else "PASS",
            }
        )

    case_columns = [
        "vial_id",
        "recipe_id",
        "previous_recipe_id",
        "transition",
        "position_in_run",
        "expected_outcome",
        "expected_carryover",
        "measured_L",
        "measured_a",
        "measured_b",
        "reference_vial_id",
        "nearest_known_recipe_id",
        "nearest_known_delta_e",
        "intended_recipe_delta_e",
        "legacy_flag",
        "recipe_aware_flag",
        "legacy_decision",
        "recipe_aware_decision",
    ]
    exclusion_columns = [
        "vial_id",
        "recipe_id",
        "position_in_run",
        "expected_outcome",
        "reason",
    ]
    comparison = pd.DataFrame(cases, columns=case_columns)
    excluded = pd.DataFrame(exclusions, columns=exclusion_columns)
    reference_columns = [
        "vial_id",
        "recipe_id",
        "position_in_run",
        "L",
        "a",
        "b",
        "spread_de",
        "status",
        "usable_reference",
        "reference_note",
    ]
    return comparison, excluded, last_vials[reference_columns]


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a metric value while keeping empty denominators explicit."""

    return float(numerator / denominator) if denominator else float("nan")


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute pairwise proxy-label ROC AUC without another dependency."""

    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    wins = np.count_nonzero(comparisons > 0)
    ties = np.count_nonzero(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)


def method_metrics(
    comparison: pd.DataFrame,
    method: str,
    score_column: str,
    threshold: float,
) -> dict[str, int | float | str]:
    """Calculate paired confusion metrics for one decision rule."""

    labels = comparison["expected_carryover"].astype(bool).to_numpy()
    scores = comparison[score_column].to_numpy(float)
    predicted = scores >= threshold
    tp = int(np.count_nonzero(labels & predicted))
    fn = int(np.count_nonzero(labels & ~predicted))
    fp = int(np.count_nonzero(~labels & predicted))
    tn = int(np.count_nonzero(~labels & ~predicted))
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    return {
        "method": method,
        "threshold_de": float(threshold),
        "scored_vials": len(comparison),
        "carryover_proxies": int(tp + fn),
        "clean_proxies": int(tn + fp),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "carryover_recall": recall,
        "miss_rate": _safe_ratio(fn, tp + fn),
        "false_alarm_rate": _safe_ratio(fp, fp + tn),
        "precision": _safe_ratio(tp, tp + fp),
        "accuracy": _safe_ratio(tp + tn, len(comparison)),
        "balanced_accuracy": (recall + specificity) / 2,
        "proxy_label_auc": _roc_auc(labels, scores),
    }


def summarize(comparison: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build the side-by-side summary at one threshold."""

    return pd.DataFrame(
        [
            method_metrics(
                comparison,
                LEGACY_METHOD,
                "nearest_known_delta_e",
                threshold,
            ),
            method_metrics(
                comparison,
                RECIPE_METHOD,
                "intended_recipe_delta_e",
                threshold,
            ),
        ]
    )


def threshold_sweep(comparison: pd.DataFrame, primary_threshold: float) -> pd.DataFrame:
    """Evaluate both rules at the fixed reported sensitivity points."""

    thresholds = sorted({*SENSITIVITY_THRESHOLDS, float(primary_threshold)})
    return pd.concat(
        [summarize(comparison, threshold) for threshold in thresholds],
        ignore_index=True,
    )


def _percent(value: float) -> str:
    """Format report percentages and preserve unavailable values."""

    return "n/a" if pd.isna(value) else f"{value:.0%}"


def _markdown_metrics(summary: pd.DataFrame) -> list[str]:
    """Render the primary comparison as a compact Markdown table."""

    lines = [
        "| Method | TP | FN | FP | TN | Carryover-proxy recall | Miss rate | False-alarm rate | Balanced accuracy | Proxy-label AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.true_positive} | {row.false_negative} | "
            f"{row.false_positive} | {row.true_negative} | {_percent(row.carryover_recall)} | "
            f"{_percent(row.miss_rate)} | {_percent(row.false_alarm_rate)} | "
            f"{_percent(row.balanced_accuracy)} | {row.proxy_label_auc:.2f} |"
        )
    return lines


def _markdown_sweep(sensitivity: pd.DataFrame) -> list[str]:
    """Render threshold sensitivity without hiding the error trade-off."""

    lines = [
        "| Threshold ΔE00 | Method | Carryover-proxy recall | False-alarm rate | Balanced accuracy |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.threshold_de:g} | {row.method} | {_percent(row.carryover_recall)} | "
            f"{_percent(row.false_alarm_rate)} | {_percent(row.balanced_accuracy)} |"
        )
    return lines


def _threshold_context(threshold: float) -> tuple[str, str]:
    """Describe whether a run uses the inherited default or a CLI override."""

    if np.isclose(threshold, float(COLOR_WARN_DE)):
        return (
            "the pre-existing shared project threshold",
            (
                "The primary result inherits the pre-existing "
                "`changeover_guard.contracts.COLOR_WARN_DE` value; it was not fitted or validated "
                "on this benchmark. The fixed sweep below prevents a conclusion based on one "
                "threshold."
            ),
        )
    return (
        "the selected threshold",
        (
            f"This run uses the caller-selected threshold ΔE00 = {threshold:g}; the shared "
            f"project default remains ΔE00 = {float(COLOR_WARN_DE):g}. Neither value was fitted "
            "or validated on this benchmark. The fixed sweep below shows the threshold trade-off."
        ),
    )


def write_report(
    output_path: Path,
    vials: pd.DataFrame,
    comparison: pd.DataFrame,
    excluded: pd.DataFrame,
    references: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    expected_case_count: int,
) -> None:
    """Write a judge-readable report with explicit evidence and caveats."""

    legacy = summary[summary["method"] == LEGACY_METHOD].iloc[0]
    recipe = summary[summary["method"] == RECIPE_METHOD].iloc[0]
    threshold = float(legacy["threshold_de"])
    threshold_label, threshold_note = _threshold_context(threshold)
    recall_gain = float(recipe["carryover_recall"] - legacy["carryover_recall"])
    carryover_count = int(recipe["carryover_proxies"])
    clean_count = int(recipe["clean_proxies"])
    coverage = _safe_ratio(len(comparison), expected_case_count)
    label_lag = pd.to_numeric(vials["label_lag_s"], errors="coerce").dropna()
    lag_outliers = vials[pd.to_numeric(vials["label_lag_s"], errors="coerce") > 10]
    lag_outlier_vials = ", ".join(
        str(int(vial_id)) for vial_id in lag_outliers["vial_id"]
    )

    lines = [
        "# Known-color vs. recipe-aware inspection",
        "",
        (
            "> **Scope:** this is a paired decision-logic benchmark on the one labeled real run. "
            "It is not a validation of the unfinished twin-trained prediction/advice pipeline, and "
            "it does not reproduce proprietary Teledyne hardware settings."
        ),
        "",
        "## Headline result",
        "",
        (
            f"At {threshold_label} of **ΔE00 = {threshold:g}**, recipe-aware matching "
            f"detected **{int(recipe['true_positive'])}/{carryover_count}** transition-first-vial "
            f"proxies, compared with **{int(legacy['true_positive'])}/{carryover_count}** for the "
            f"any-known-color emulation. That is a **{recall_gain * 100:+.0f} percentage-point** "
            "change in carryover-proxy recall. The false-alarm counts were "
            f"**{int(recipe['false_positive'])}/{clean_count}** "
            f"and **{int(legacy['false_positive'])}/{clean_count}**, respectively."
        ),
        "",
        "![Bar chart comparing the two inspection rules](method_comparison.svg)",
        "",
        *_markdown_metrics(summary),
        "",
        "## What changed between the methods",
        "",
        (
            "Both methods use exactly the same measured CIELAB colors, third-vial references, "
            "CIEDE2000 distance, threshold, and scored vials."
        ),
        "",
        (
            "- **Legacy emulation:** passes when a vial resembles *any* known clean color. It can "
            "therefore accept the wrong recipe when that color is already in its database."
        ),
        (
            "- **Recipe-aware baseline:** passes only when the vial resembles the clean reference "
            "for the recipe that was actually scheduled."
        ),
        "",
        (
            "This isolates the value of recipe context. It does not attribute gains to different "
            "camera processing or a different threshold."
        ),
        "",
        "## Evaluation set",
        "",
        "- Real `vial_fill_video_5.mp4`: 30 vials, 10 recipes, 3 vials per recipe.",
        "- The third vial of each recipe is the documented clean-color reference and is not scored.",
        "- The second vial is a clean proxy.",
        (
            "- The first vial after a known recipe change is a carryover-risk proxy; recipe 5 vial 1 "
            "is excluded because its preceding recipe is unknown."
        ),
        (
            f"- Coverage: **{len(comparison)}/{expected_case_count} ({coverage:.0%})** planned cases; "
            f"**{int(references['usable_reference'].sum())}/{len(references)}** recipes had a usable "
            "last-vial color measurement."
        ),
        "",
        (
            "The supplied dataset has no independent good/bad or root-cause labels. These outcomes "
            "are process-derived proxies based on the event guide's documented carryover behavior."
        ),
        "",
        "## Measurement method",
        "",
        (
            f"The runner reuses the repository's existing `color_monitor` measurement path: "
            f"sample video 5 every {monitor_config.SAMPLE_STRIDE_S:g} second, measure a fixed "
            "liquid ROI in CIELAB, segment physical vial cycles, and take the final sustained "
            "stable-color window. Cycles are matched to labels by production order. The "
            "measurement algorithm is identical for both decision rules."
        ),
        "",
        (
            f"Cycle audit: **{vials['cycle'].nunique()} observed cycles / {len(vials)} labels**. "
            f"Observed cycle start minus labeled fill start had median "
            f"**{label_lag.median():.2f} s** and maximum **{label_lag.max():.2f} s**. "
            f"Lags above 10 s occurred for vial(s) **{lag_outlier_vials or 'none'}**. The pairing "
            "is ordinal (cycle 1 to vial 1, and so on), so those timing outliers remain an "
            "explicit review item rather than hidden evidence."
        ),
        "",
        "## Threshold sensitivity",
        "",
        threshold_note,
        "",
        *_markdown_sweep(sensitivity),
        "",
        "## Why the known-color rule misses transition vials",
        "",
        (
            "The following carryover-risk proxies passed the legacy rule at the primary threshold. "
            "`Nearest known recipe` shows which database color made the vial look acceptable."
        ),
        "",
        "| Vial | Transition | Nearest known recipe | Nearest ΔE00 | Intended-recipe ΔE00 | Recipe-aware decision |",
        "|---:|---:|---:|---:|---:|---|",
    ]

    legacy_misses = comparison[
        comparison["expected_carryover"].astype(bool)
        & ~comparison["legacy_flag"].astype(bool)
    ]
    if legacy_misses.empty:
        lines.append("| — | — | — | — | — | No legacy misses at this threshold |")
    else:
        for row in legacy_misses.itertuples(index=False):
            lines.append(
                f"| {row.vial_id} | {row.transition} | {row.nearest_known_recipe_id} | "
                f"{row.nearest_known_delta_e:.2f} | {row.intended_recipe_delta_e:.2f} | "
                f"{row.recipe_aware_decision} |"
            )

    lines.extend(["", "## Exclusions", ""])
    if excluded.empty:
        lines.append("No planned cases were excluded.")
    else:
        lines.extend(
            [
                "| Vial | Recipe | Position | Expected outcome | Reason |",
                "|---:|---:|---:|---|---|",
            ]
        )
        for row in excluded.itertuples(index=False):
            lines.append(
                f"| {row.vial_id} | {row.recipe_id} | {row.position_in_run} | "
                f"{row.expected_outcome} | {row.reason} |"
            )

    lines.extend(
        [
            "",
            "## Planned metrics not yet claimable",
            "",
            "| Metric | Status | What is missing |",
            "|---|---|---|",
            "| Leave-one-recipe-out generalization | Pending | No vial/carryover `synthetic/` training set exists. |",
            "| First-vial prediction ΔE00 | Pending | `mixing.predict()` is still a stub. |",
            "| Explanation correctness | Pending | `advise()` is still a stub and there are no independent cause labels. |",
            "| Ordering-advice comparison | Pending | Scheduling advice is not implemented or labeled. |",
            "",
            "## Interpretation limits",
            "",
            (
                "- This emulates the documented *logic* of a known-color database; it is not a "
                "head-to-head test against the Teledyne system."
            ),
            (
                "- References and test vials come from the same production run. The result shows the "
                "benefit of intended-recipe context, not cross-run robustness."
            ),
            (
                "- There are only three vials per recipe. Confidence intervals would be too wide to "
                "support a production-accuracy claim."
            ),
            (
                "- The last-vial method is an offline evaluation oracle. A deployed system must obtain "
                "expected colors from the digital twin or a separately validated reference set."
            ),
            "",
            "## Reproduce",
            "",
            "```powershell",
            "uv run python eval/compare.py --refresh",
            "uv run pytest eval/test_compare.py",
            "```",
            "",
            (
                "See `per_vial.csv`, `summary.csv`, `threshold_sensitivity.csv`, `references.csv`, "
                "`excluded.csv`, and `vial_measurements.meta.json` beside this report for the "
                "underlying evidence and cache provenance."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(summary: pd.DataFrame, output_path: Path) -> None:
    """Create a dependency-free side-by-side metric chart for the report."""

    rows = summary.set_index("method")
    threshold = float(summary.iloc[0]["threshold_de"])
    metrics = [
        ("Carryover-proxy recall (higher is better)", "carryover_recall"),
        ("False-alarm rate (lower is better)", "false_alarm_rate"),
        ("Balanced accuracy (higher is better)", "balanced_accuracy"),
    ]
    methods = [
        (LEGACY_METHOD, "#6b7280"),
        (RECIPE_METHOD, "#15803d"),
    ]
    width, height = 920, 430
    bar_x, bar_width = 300, 520
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        "<title>Known-color versus recipe-aware inspection metrics</title>",
        "<desc>Three paired horizontal bar charts at the selected color threshold.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<text x="30" y="38" font-family="Arial, sans-serif" font-size="23" '
            f'font-weight="700" fill="#111827">Decision-rule comparison at ΔE00 {threshold:g}</text>'
        ),
    ]

    y = 78
    for label, column in metrics:
        elements.append(
            f'<text x="30" y="{y}" font-family="Arial, sans-serif" font-size="16" '
            f'font-weight="700" fill="#111827">{html.escape(label)}</text>'
        )
        y += 25
        for method, color in methods:
            raw_value = float(rows.loc[method, column])
            available = not np.isnan(raw_value)
            value = min(max(raw_value, 0.0), 1.0) if available else 0.0
            fill_width = value * bar_width
            display_value = f"{value:.0%}" if available else "n/a"
            elements.extend(
                [
                    (
                        f'<text x="30" y="{y + 16}" font-family="Arial, sans-serif" '
                        f'font-size="14" fill="#374151">{html.escape(method)}</text>'
                    ),
                    (
                        f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="22" '
                        'rx="3" fill="#e5e7eb"/>'
                    ),
                    (
                        f'<rect x="{bar_x}" y="{y}" width="{fill_width:.1f}" height="22" '
                        f'rx="3" fill="{color}"/>'
                    ),
                    (
                        f'<text x="{bar_x + bar_width + 12}" y="{y + 16}" '
                        f'font-family="Arial, sans-serif" font-size="14" font-weight="700" '
                        f'fill="#111827">{display_value}</text>'
                    ),
                ]
            )
            y += 30
        y += 22

    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse paths and the primary operating threshold."""

    parser = argparse.ArgumentParser(
        description="Compare any-known-color and intended-recipe inspection on real video 5."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="real dataset root"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for measurements and comparison artifacts",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(COLOR_WARN_DE),
        help="primary CIEDE2000 warning threshold",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="remeasure video 5 instead of reusing vial_measurements.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and write all reviewable artifacts under eval/."""

    args = parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0:
        raise ValueError("--threshold must be a positive finite number")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "vial_measurements.csv"
    vials = load_or_measure(args.data, cache_path, args.refresh)
    expected_case_count = len(_expected_cases(vials))
    comparison, excluded, references = build_comparison(vials, args.threshold)
    if comparison.empty:
        raise ValueError("No vials are jointly evaluable by both methods")

    summary = summarize(comparison, args.threshold)
    sensitivity = threshold_sweep(comparison, args.threshold)
    comparison.to_csv(output / "per_vial.csv", index=False, float_format="%.6f")
    excluded.to_csv(output / "excluded.csv", index=False)
    references.to_csv(output / "references.csv", index=False, float_format="%.6f")
    summary.to_csv(output / "summary.csv", index=False, float_format="%.6f")
    sensitivity.to_csv(
        output / "threshold_sensitivity.csv",
        index=False,
        float_format="%.6f",
    )
    write_svg(summary, output / "method_comparison.svg")
    write_report(
        output / "report.md",
        vials,
        comparison,
        excluded,
        references,
        summary,
        sensitivity,
        expected_case_count,
    )

    print(f"Measured vials: {(vials['status'] == 'ok').sum()}/{len(vials)}")
    print(f"Comparison coverage: {len(comparison)}/{expected_case_count}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nReport: {output / 'report.md'}")


if __name__ == "__main__":
    main()
