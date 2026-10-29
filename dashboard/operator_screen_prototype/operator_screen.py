"""Changeover Guard operator mockup. Read-only simulated data; no PLC connection."""
from pathlib import Path

import streamlit as st

# Streamlit puts this script's directory on sys.path, so import the helper
# beside this file rather than requiring the project root to be importable.
from operator_result import load_result

st.set_page_config(page_title="Changeover Guard — Operator", layout="wide")
# Explicit light theme colors prevent white-on-white text when the user's
# Streamlit/browser theme is dark. Keep the layout close to the machine HMI.
st.markdown("""<style>
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"] {background-color:#eef2f6; color:#172b3a;}
.block-container {max-width:1250px; padding-top:1.4rem;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {color:#172b3a;}
[data-testid="stMetric"] {background:#ffffff; border:1px solid #cfd7df; padding:12px; border-radius:4px;}
[data-testid="stTabs"] button {color:#243849;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#125d90; border-bottom-color:#125d90;}
[data-testid="stAlert"] {border:1px solid #cbd5df;}
[data-testid="stAlert"] p, [data-testid="stAlert"] li,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color:#172b3a !important;}
.notice {padding:15px; background:#fff0ca; color:#583900; border-left:6px solid #d99815; font-weight:700;}
.ok {padding:15px; background:#dff3e4; color:#155a34; border-left:6px solid #22824b; font-weight:700;}
</style>""", unsafe_allow_html=True)

# Demo records are based on the challenge description; swatch colors are illustrative.
SCENARIOS = {
    "Vial 7 — color carryover": dict(vial=7, recipe="7", previous="6", expected="Blue", observed="Green-tinted", expected_hex="#2d7dc0", observed_hex="#6c9866", status="FAIL", cause="Possible orange residue from Recipe 6 in the shared mixing path changed the first blue vial to green.", action="Inspect and segregate the affected vial. Review the approved changeover procedure with the plant manager.", fill="Not measured"),
    "Vial 8 — clean fill": dict(vial=8, recipe="7", previous="6", expected="Blue", observed="Blue", expected_hex="#2d7dc0", observed_hex="#2d7dc0", status="PASS", cause="The later vial matches the expected Recipe 7 color. Residual fluid may have cleared.", action="Continue normal quality checks under approved procedures.", fill="Not measured"),
    "Vial 9 — clean fill": dict(vial=9, recipe="7", previous="6", expected="Blue", observed="Blue", expected_hex="#2d7dc0", observed_hex="#2d7dc0", status="PASS", cause="The later vial matches the expected Recipe 7 color.", action="Continue normal quality checks under approved procedures.", fill="Not measured"),
    "Simulated fill-level issue": dict(vial="Demo", recipe="7", previous="6", expected="Blue", observed="Blue", expected_hex="#2d7dc0", observed_hex="#2d7dc0", status="FAIL", cause="Simulated fill-volume measurement is outside the configured tolerance. The underlying cause has not been established.", action="Inspect the vial and verify fill measurements before following the approved quality procedure.", fill="65 mL measured / 70 mL target (simulated)"),
}

# Optional integration point: teammate's service can write a JSON result here.
# This file is never required for the mockup to run.
LIVE_RESULT_PATH = Path(__file__).with_name("inspection_result.json")

st.title("Changeover Guard | Fluid Mixing Station")
st.caption("OPERATOR ASSISTANT  •  LOCAL DEMO  •  READ-ONLY: NO MACHINE COMMANDS")
inspection, queue, evidence = st.tabs(["Vial Inspection", "Recipe Queue", "Camera Evidence"])

with inspection:
    left, right = st.columns([3, 1])
    with right:
        use_json = st.checkbox("Use JSON model output", value=False, help="Reads inspection_result.json next to this file when present")
    with left:
        scenario = st.selectbox("Select demonstration case", list(SCENARIOS), disabled=use_json)
    result = SCENARIOS[scenario].copy()
    if use_json:
        try:
            result, ignored = load_result(LIVE_RESULT_PATH)
            st.info("Displaying local JSON output. This is not a verified live machine connection.")
            if ignored:
                st.caption(f"Ignored unknown JSON keys: {', '.join(ignored)}")
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"Could not load model output ({exc}). Showing demonstration data.")
    failed = str(result["status"]).upper() != "PASS"
    st.markdown('<div class="notice">⚠ INSPECTION FLAGGED — REVIEW REQUIRED</div>' if failed else '<div class="ok">✓ INSPECTION PASSED</div>', unsafe_allow_html=True)
    st.write("")
    a, b, c, d = st.columns(4)
    a.metric("Current recipe", f"Recipe {result['recipe']}")
    b.metric("Previous recipe", f"Recipe {result['previous']}")
    c.metric("Vial", str(result["vial"]))
    d.metric("Inspection", str(result["status"]).upper())
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Expected vial color")
        st.color_picker("Recipe target (illustrative)", result["expected_hex"], disabled=True)
        st.write(result["expected"])
    with c2:
        st.subheader("Observed vial color")
        st.color_picker("Camera result (illustrative)", result["observed_hex"], disabled=True)
        st.write(result["observed"])
    st.markdown("**Likely cause / explanation**")
    if failed:
        st.warning(result["cause"])
    else:
        st.success(result["cause"])
    st.markdown("**Recommended operator action**")
    st.info(result["action"])
    st.caption(f"Fill check: {result['fill']}. Cause is a hypothesis unless independently verified.")
    if "acknowledged_case" not in st.session_state:
        st.session_state.acknowledged_case = None
    if st.button("Acknowledge displayed warning", disabled=not failed):
        st.session_state.acknowledged_case = scenario
    if failed and st.session_state.acknowledged_case == scenario:
        st.caption("Acknowledged in this browser session only; no equipment fault cleared.")
    with st.expander("Recipe context and why this case matters"):
        st.write("Recipe 6: 30 mL T-100 (clear) + 10 mL T-200 (orange).")
        st.write("Recipe 7: 20 mL T-100 (clear) + 50 mL T-400 (blue).")
        st.write("Dispense order is T-200 → T-400 → T-100, skipping unused tanks. The first vial of Recipe 7 was green; the next two were blue.")
        st.write("The current station's known-color check could accept a color that belongs to a different recipe. Our inspection should compare against the active recipe instead.")

with queue:
    st.subheader("Recipe order optimization — demonstration")
    st.caption("Suggested order is an EXAMPLE, not a calculated result. Production scheduler can queue at most three recipes at once.")
    original = ["Recipe 6 — Orange", "Recipe 7 — Blue", "Recipe 6 — Orange"]
    suggested = ["Recipe 6 — Orange", "Recipe 6 — Orange", "Recipe 7 — Blue"]
    l, r = st.columns(2)
    with l:
        st.markdown("#### Original order")
        st.dataframe([{"Slot": i+1, "Recipe": recipe} for i, recipe in enumerate(original)], hide_index=True, width="stretch")
    with r:
        st.markdown("#### Example grouped order")
        st.dataframe([{"Slot": i+1, "Recipe": recipe} for i, recipe in enumerate(suggested)], hide_index=True, width="stretch")
    st.info("**Goal:** Reduce predicted discoloration during recipe transitions. The optimization model will supply a proposed order and any estimated improvement once integrated.")
    st.markdown("**Operating constraints**")
    st.write("Maximum 80 mL per vial; selected components should be at least 10 mL; T-200 must not exceed 50 mL; T-300 must not be used.")
    st.warning("Do not change recipes without plant-manager guidance. This preview cannot write to the Batch Scheduler.")

with evidence:
    st.subheader("Camera evidence")
    st.caption("The challenge provides timestamp-synchronized vial_fill, tank_level, and cap_seal cameras. Select local image files to preview; nothing is uploaded to equipment.")
    files = st.file_uploader("Choose camera frames", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files:
        for i in range(0, len(files), 3):
            cols = st.columns(3)
            for col, file in zip(cols, files[i:i+3]):
                col.image(file, caption=file.name, width="stretch")
    else:
        st.info("No images selected. For the demo, load vial_fill frames of Vials 7–9 or synchronized tank/cap images.")
    st.markdown("**Evidence checklist**")
    st.write("1. Match frames by timestamp.  2. Identify active and prior recipes.  3. Compare observed color with the active recipe.  4. Review tank state and fill level if available.")
    st.caption("Camera evidence and model predictions will be wired up after the other components are ready.")
