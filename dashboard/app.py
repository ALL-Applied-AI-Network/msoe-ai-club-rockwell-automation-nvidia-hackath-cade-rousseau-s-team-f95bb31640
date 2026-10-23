import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from changeover_guard.contracts import COLOR_WARN_DE, delta_e
from changeover_guard.dataset import frames, vial_labels
from changeover_guard.measure import tank_rois, tanks

ROOT = Path(__file__).parents[1] / "data"
PROJECT_ROOT = Path(__file__).parents[1]
AI_TANK_ROOT = PROJECT_ROOT / "generated" / "images" / "tank_color_variation"
AI_COLOR_MANIFEST = AI_TANK_ROOT / "ai_tank_colors.json"


def _annotated_tank_image(image):
    annotated = image.copy()
    # UPDATED: draw the exact public ROIs used by measure.tanks().
    for tank, (x1, y1, x2, y2) in tank_rois(image.shape).items():
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


@st.cache_data
def load_ai_color_targets():
    if not AI_COLOR_MANIFEST.exists():
        return {}
    manifest = json.loads(AI_COLOR_MANIFEST.read_text(encoding="utf-8"))
    return {record["filename"]: record["requested_rgb"] for record in manifest["images"]}


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


def show_tank_measurement() -> None:
    sources = {}
    real_images = sorted((ROOT / "tank_level" / "images").glob("*.jpg"))
    ai_images = sorted(AI_TANK_ROOT.glob("ai_tank_variant_*.png"))
    if real_images:
        sources["Real tank_level images"] = real_images
    if ai_images:
        sources["AI test images"] = ai_images

    if not sources:
        st.info("No tank images found in data/tank_level/images or generated/images/tank_color_variation.")
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
        reading = tanks(image, _image_timestamp(image_path))
    except (ValueError, cv2.error) as exc:
        st.error(f"Could not measure tank colors in {image_path.name}: {exc}")
        return

    # UPDATED: manifest values are generator requests, not pixel measurements.
    requested_colors = load_ai_color_targets().get(image_path.name, {})
    st.image(
        _annotated_tank_image(image),
        caption=f"{source_name}: {image_path.name}; green boxes are the measurement ROIs",
        width="stretch",
    )

    st.subheader("Measured tank colors")
    rois = tank_rois(image.shape)
    columns = st.columns(len(reading.colors))
    for column, (tank, color) in zip(columns, reading.colors.items()):
        x1, y1, x2, y2 = rois[tank]
        requested = requested_colors.get(tank, "N/A")
        calculated = tuple(float(channel) for channel in color)
        # UPDATED: keep the dashboard compact by coloring only the ΔE value
        # red when the measured error reaches the project warning threshold.
        color_error_value = None if requested == "N/A" else delta_e(requested, calculated)
        color_error = "N/A" if color_error_value is None else f"{color_error_value:.2f}"
        if color_error_value is not None and color_error_value >= COLOR_WARN_DE:
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


st.title("Changeover Guard")
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
