import importlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

# Streamlit puts this script's directory on sys.path, so import the helper
# beside app.py rather than requiring the project root to be importable.
from color_targets import load_ai_color_targets

from changeover_guard import measure
from changeover_guard.contracts import (
    COLOR_WARN_DE,
    TANK_FILL_WARN_PERCENT,
    color_difference,
)
from changeover_guard.dataset import frames, vial_labels

ROOT = Path(__file__).parents[1] / "data"
PROJECT_ROOT = Path(__file__).parents[1]
# UPDATED: use the top-view generated fixtures that match measure._TANK_ROIS.
# These assets support dashboard demonstration and tests; the project guidance
# still reserves model training for digital-twin synthetic data.
AI_TANK_ROOT = PROJECT_ROOT / "generated" / "images" / "top_view_tank_color"
AI_COLOR_MANIFEST = AI_TANK_ROOT / "ai_tank_colors.json"
# The front-view fixtures use the same camera layout as the real 640x480
# frames, but their color requests must be read from their own JSON file.
AI_FRONT_TANK_ROOT = PROJECT_ROOT / "generated" / "images" / "front_view_tank_color"
AI_FRONT_COLOR_MANIFEST = AI_FRONT_TANK_ROOT / "ai_tank_colors.json"
AI_FILL_ROOT = PROJECT_ROOT / "generated" / "images" / "tank_fill_level"
AI_FILL_MANIFEST = AI_FILL_ROOT / "json" / "tank_fill_levels.json"


def _annotated_tank_image(image, rois):
    annotated = image.copy()
    # Draw the liquid ROIs that measure.tanks() actually reads.
    for tank, (x1, y1, x2, y2) in rois.items():
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            tank,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def _annotated_fill_image(image, expected, measured, rois):
    """Draw expected and measured fill lines over each tank image.

    The red line is the actual/reference line recorded in the fixture JSON;
    the green line is the line reconstructed from measure.tank_fill_levels().
    """
    annotated = image.copy()
    # rois is the public, scaled view of measure._FILL_ROIS. It is passed in
    # from the caller so the measurement overlay and coordinate readout use
    # exactly the same calibration values.
    for tank, (x1, y1, x2, y2) in rois.items():
        expected_level = expected.get(tank)
        measured_level = measured.get(tank)
        difference = (
            None
            if expected_level is None or measured_level is None
            else abs(expected_level - measured_level)
        )
        # OpenCV colors are BGR: (255, 0, 0) is pure blue. The blue outline
        # marks the exact _FILL_ROIS region used by the fill detector.
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # A zero-width or zero-height calibration is still drawn so edits to
        # _FILL_ROIS are immediately visible, but it cannot contain a valid
        # fill line or produce a percentage measurement.
        valid_geometry = x2 > x1 and y2 > y1
        if valid_geometry and expected_level is not None:
            # The JSON target is the actual/reference fill line and is always
            # red, regardless of whether the detector agrees with it.
            expected_y = measure.fill_line_row((x1, y1, x2, y2), expected_level)
            cv2.line(annotated, (x1, expected_y), (x2, expected_y), (0, 0, 255), 3)
        if valid_geometry and measured_level is not None:
            # The measured line is always green so an operator can compare it
            # directly with the red reference line in the image.
            measured_y = measure.fill_line_row((x1, y1, x2, y2), measured_level)
            cv2.line(annotated, (x1, measured_y), (x2, measured_y), (0, 255, 0), 3)

        # Use red text only for measurements outside the five-point tolerance;
        # the line colors above remain fixed so red always means actual and
        # green always means measured.
        warning = difference is not None and difference > TANK_FILL_WARN_PERCENT
        label_color = (0, 0, 255) if warning else (255, 255, 255)
        label = f"{tank}: {measured_level:.1f}%" if measured_level is not None else f"{tank}: no line"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def load_ai_fill_targets():
    """Read actual tank levels from the editable fill-image manifest."""
    if not AI_FILL_MANIFEST.exists():
        return {}
    # Do not cache this small manifest: the red reference line must move when
    # an operator edits and saves the JSON.
    manifest = json.loads(AI_FILL_MANIFEST.read_text(encoding="utf-8"))
    return {
        record["filename"]: {
            tank: float(details["level_percent"])
            for tank, details in record["tank_levels"].items()
        }
        for record in manifest["images"]
    }


def _color_swatch(rgb) -> str:
    if rgb == "N/A":
        return ""
    red, green, blue = (max(0, min(255, round(channel))) for channel in rgb)
    return (
        f"<span style='display:inline-block; width:1.1rem; height:1.1rem; "
        f"margin:0 0.25rem; vertical-align:middle; border:1px solid #555; "
        f"border-radius:3px; background-color:rgb({red},{green},{blue});'></span>"
    )


def _image_timestamp(image_path: Path) -> datetime:
    """Read a tank frame timestamp from its filename when available."""
    try:
        return datetime.strptime(
            image_path.stem.removeprefix("tank_level_"), "%Y%m%d_%H%M%S_%f"
        )
    except ValueError:
        # Generated images have no capture timestamp; their file timestamp is
        # still more truthful than assigning the current time.
        return datetime.fromtimestamp(image_path.stat().st_mtime)


def _reload_measure_if_edited() -> None:
    """Pick up ROI edits in measure.py without restarting Streamlit."""
    # Python keeps imported modules in memory across reruns. Reload measure.py
    # only after a file edit; every call site uses measure.<name>, so both the
    # color and fill panels see the new ROIs.
    source_stat = Path(measure.__file__).stat()
    source_signature = (source_stat.st_mtime_ns, source_stat.st_size)
    if st.session_state.get("measure_source_signature") != source_signature:
        importlib.reload(measure)
        st.session_state["measure_source_signature"] = source_signature


def show_tank_fill_measurement() -> None:
    """Display measured tank fill levels beside their expected percentages."""
    st.subheader("Tank fill-level measurement")
    fill_targets = load_ai_fill_targets()
    fill_images = sorted(AI_FILL_ROOT.glob("tank_level_variant_*.png"))

    if not fill_images or not fill_targets:
        st.info("No generated tank fill-level images or manifest were found.")
        return

    # The fill image is selected independently from the color image above so
    # the dashboard can demonstrate both measurement pipelines in one view.
    fill_image_path = st.selectbox(
        "Fill-level test image",
        fill_images,
        format_func=lambda path: path.name,
        key="tank_fill_image",
    )
    image = cv2.imread(str(fill_image_path), cv2.IMREAD_COLOR)
    if image is None:
        st.error(f"Could not read {fill_image_path}")
        return

    try:
        measured = measure.tank_fill_levels(image)
    except (ValueError, cv2.error) as exc:
        # Keep rendering the image and blue ROI boxes even when a calibration
        # edit is temporarily invalid (for example, (50, 50, 50, 50)). This
        # makes ROI edits visible instead of leaving the previous image on
        # screen while the measurement error is reported.
        measured = {}
        st.warning(
            f"Fill measurement unavailable for {fill_image_path.name}: {exc}. "
            "The blue ROI boxes still show the current _FILL_ROIS values."
        )

    # Calculate the scaled coordinates once for this exact source image. The
    # stored _FILL_ROIS values are calibrated for 640x480; fill_rois() maps
    # them to the selected image resolution before drawing the blue boxes.
    displayed_rois = measure.fill_rois(image.shape)
    expected = fill_targets.get(fill_image_path.name, {})
    st.caption(
        "Expected and measured values are percentages of tank height. "
        f"A difference greater than {TANK_FILL_WARN_PERCENT:.0f}% is flagged in red."
    )
    st.image(
        _annotated_fill_image(image, expected, measured, displayed_rois),
        caption=(
            f"{fill_image_path.name}; red line = actual/reference, "
            "green line = measured; warning labels exceed the allowed "
            f"{TANK_FILL_WARN_PERCENT:.0f}% difference"
        ),
        width="stretch",
    )

    # Display the actual pixel coordinates used for the overlay so calibration
    # edits can be checked against the live image instead of inferred visually.
    st.caption(
        "Source _FILL_ROIS: "
        + " | ".join(
            f"{tank}: {roi}" for tank, roi in measure._FILL_ROIS.items()
        )
    )
    st.caption(
        "Rendered fill ROIs (scaled to "
        f"{image.shape[1]}x{image.shape[0]}): "
        + " | ".join(
            f"{tank}: ({x1}, {y1}) to ({x2}, {y2})"
            for tank, (x1, y1, x2, y2) in displayed_rois.items()
        )
    )

    # A malformed or zero-size ROI can be displayed for calibration, but it
    # cannot produce comparison cards until the measurement geometry is valid.
    if not measured:
        return

    # Show one compact comparison card per physical tank, including T300 for
    # equipment monitoring even though it is not a recipe source tank.
    columns = st.columns(len(measured))
    for column, (tank, measured_level) in zip(columns, measured.items()):
        expected_level = expected.get(tank)
        difference = (
            None
            if expected_level is None or measured_level is None
            else abs(expected_level - measured_level)
        )
        # No visible line means the level is unknown, which is itself a warning.
        is_warning = measured_level is None or (
            difference is not None and difference > TANK_FILL_WARN_PERCENT
        )
        text_color = "#c1121f" if is_warning else "#167c2e"
        difference_text = "N/A" if difference is None else f"{difference:.1f}%"
        expected_text = "N/A" if expected_level is None else f"{expected_level:.1f}%"
        measured_text = "no line" if measured_level is None else f"{measured_level:.1f}%"
        column.markdown(
            f"<div style='font-size:0.9rem; line-height:1.35; color:{text_color}'>"
            f"<strong>{tank}</strong><br>"
            f"Expected: {expected_text}<br>"
            f"Measured: {measured_text}<br>"
            f"Difference: {difference_text}"
            "</div>",
            unsafe_allow_html=True,
        )


def show_tank_measurement() -> None:
    sources = {}
    real_images = sorted((ROOT / "tank_level" / "images").glob("*.jpg"))
    ai_images = sorted(AI_TANK_ROOT.glob("ai_tank_variant_*.png"))
    front_images = sorted(AI_FRONT_TANK_ROOT.glob("ai_tank_variant_*.png"))
    if real_images:
        sources["Real tank_level images"] = real_images
    if ai_images:
        sources["AI top-view test images"] = ai_images
    if front_images:
        # Add the front-view fixtures after the real and top-view sets so the
        # dropdown keeps a predictable order and both AI sets remain available.
        sources["AI front-view test images"] = front_images

    if not sources:
        st.info("No real, top-view, or front-view tank images were found.")
        return

    st.header("Tank ROI color measurement")
    source_name = st.selectbox("Image set", list(sources))
    image_paths = sources[source_name]
    image_path = st.selectbox("Tank image", image_paths, format_func=lambda path: path.name)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        st.error(f"Could not read {image_path}")
        return

    # UPDATED: preserve the real frame timestamp and keep the rest of the
    # dashboard available when one frame cannot be measured.
    try:
        reading = measure.tanks(image, _image_timestamp(image_path))
        rois = measure.liquid_rois(image)
    except (ValueError, cv2.error) as exc:
        st.error(f"Could not measure tank colors in {image_path.name}: {exc}")
        return

    # Both generated sets reuse filenames, so select the matching manifest by
    # image source before comparing requested and measured tank colors.
    manifest_by_source = {
        "AI top-view test images": AI_COLOR_MANIFEST,
        "AI front-view test images": AI_FRONT_COLOR_MANIFEST,
    }
    manifest_path = manifest_by_source.get(source_name)
    requested_colors = (
        load_ai_color_targets(manifest_path).get(image_path.name, {})
        if manifest_path is not None
        else {}
    )
    st.image(
        _annotated_tank_image(image, rois),
        caption=f"{source_name}: {image_path.name}; green boxes are the measurement ROIs",
        width="stretch",
    )

    st.subheader("Measured tank colors")
    # Show the same CIEDE2000 warning boundary used for every image set.
    st.caption(f"Off-color warning at ΔE ≥ {COLOR_WARN_DE:g}.")
    columns = st.columns(len(reading.colors))
    for column, (tank, color) in zip(columns, reading.colors.items()):
        x1, y1, x2, y2 = rois[tank]
        requested = requested_colors.get(tank, "N/A")
        calculated = tuple(float(channel) for channel in color)
        # Compare the requested/actual RGB with the measured RGB through the
        # shared contract rule, then highlight only off-color ΔE values.
        # Use the shared contract helper so the dashboard cannot drift from
        # the warning boundary exercised by the measurement tests.
        if requested != "N/A":
            color_error_value, color_warning = color_difference(requested, calculated)
        else:
            color_error_value, color_warning = None, False
        color_error = "N/A" if color_error_value is None else f"{color_error_value:.2f}"
        if color_warning:
            color_error = f"<span style='color:#c1121f; font-weight:700'>{color_error}</span>"
        column.markdown(
            f"<div style='font-size: 0.9rem; line-height: 1.3'>"
            f"<strong>{tank}</strong><br>"
            f"Requested RGB: {_color_swatch(requested)}{requested}<br>"
            f"Measured RGB: {_color_swatch(calculated)}"
            f"{tuple(round(channel, 1) for channel in calculated)}<br>"
            f"Color error (ΔE): {color_error}"
            "</div>",
            unsafe_allow_html=True,
        )
        column.caption(f"ROI: ({x1}, {y1}) to ({x2}, {y2})")

    # Place the fill demonstration directly below the color measurement so an
    # operator can compare tank identity and tank level from the same screen.
    show_tank_fill_measurement()


st.title("Changeover Guard")
# Any rerun reloads edited ROIs; this button just triggers one on demand.
st.button("Reload calibration")
_reload_measure_if_edited()
show_tank_measurement()

if not (ROOT / "vial_fill").exists():
    st.info(
        "The real vial dataset is not available. The tank ROI viewer above can still "
        "display generated test images."
    )
    st.stop()


@st.cache_data
def load_frames():
    return frames(ROOT, 5)


@st.cache_data
def load_recipes():
    return {vial: recipe for vial, recipe, _, _ in vial_labels(ROOT)}


df = load_frames()
recipes = load_recipes()

row = df.iloc[st.slider("Frame", 0, len(df) - 1)]

left, right = st.columns(2)
left.image(row.vial_fill, caption="vial_fill")
right.image(row.tank_level, caption="tank_level")

st.write(f"Timestamp: {row.timestamp}")
if pd.notna(row.in_fill) and row.in_fill == 1:
    st.write("in_fill: 1")
    st.write(f"Vial: {int(row.vial)}")
    recipe = recipes.get(int(row.vial))
    if recipe:
        ml = ", ".join(f"{t} {v:g} mL" for t, v in recipe.ml.items())
        st.write(f"Recipe {recipe.recipe_id}: {ml}")
else:
    st.write("in_fill: 0 (between fills)")
