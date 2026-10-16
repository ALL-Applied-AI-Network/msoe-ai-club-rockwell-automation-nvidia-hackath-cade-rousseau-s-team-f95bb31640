from pathlib import Path

import pandas as pd
import streamlit as st

from changeover_guard.dataset import frames, vial_labels

ROOT = Path(__file__).parents[1] / "data"

st.title("Changeover Guard")

if not (ROOT / "vial_fill").exists():
    st.error(
        "Dataset not found. Unzip the competition data so that "
        "data/vial_fill and data/tank_level exist, then reload."
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
