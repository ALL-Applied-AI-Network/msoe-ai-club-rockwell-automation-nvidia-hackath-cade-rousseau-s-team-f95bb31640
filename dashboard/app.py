import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from changeover_guard.contracts import delta_e
from changeover_guard.dataset import frames, vial_labels
from changeover_guard.measure import _REFERENCE_SIZE, _TANK_ROIS, tanks

ROOT = Path(__file__).parents[1] / "data"
PROJECT_ROOT = Path(__file__).parents[1]
AI_TANK_ROOT = PROJECT_ROOT / "generated" / "images" / "tank_color_variation"
AI_COLOR_MANIFEST = AI_TANK_ROOT / "ai_tank_colors.json"


def _scaled_rois(image):
    """Return the measurement ROIs scaled to the supplied image dimensions."""
    height, width = image.shape[:2]
    scale_x = width / _REFERENCE_SIZE[0]
    scale_y = height / _REFERENCE_SIZE[1]
    return {
        tank: (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        for tank, (x1, y1, x2, y2) in _TANK_ROIS.items()
    }


def _annotated_tank_image(image):
    annotated = image.copy()
    for tank, (x1, y1, x2, y2) in _scaled_rois(image).items():
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

    reading = tanks(image, datetime.now())
    actual_colors = load_ai_color_targets().get(image_path.name, {})
    st.image(
        _annotated_tank_image(image),
        caption=f"{source_name}: {image_path.name}; green boxes are the measurement ROIs",
        use_container_width=True,
    )

    st.subheader("Calculated tank colors")
    rois = _scaled_rois(image)
    columns = st.columns(len(reading.colors))
    for column, (tank, color) in zip(columns, reading.colors.items()):
        x1, y1, x2, y2 = rois[tank]
        actual = actual_colors.get(tank, "N/A")
        calculated = tuple(float(channel) for channel in color)
        color_error = "N/A" if actual == "N/A" else f"{delta_e(actual, calculated):.2f}"
        column.markdown(
            f"<div style='font-size: 0.9rem; line-height: 1.3'>"
            f"<strong>{tank}</strong><br>"
            f"Actual RGB: {_color_swatch(actual)}{actual}<br>"
            f"Calculated RGB: {_color_swatch(calculated)}"
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
