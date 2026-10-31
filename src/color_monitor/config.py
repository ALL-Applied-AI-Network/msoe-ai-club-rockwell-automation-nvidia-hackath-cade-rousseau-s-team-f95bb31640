from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "data" / "csi-vial-fill-dataset"
VIAL_FILL = DATASET / "vial_fill"
IMAGES = VIAL_FILL / "images"
IMAGE_INDEX = VIAL_FILL / "image_index.csv"
VIAL_RECIPES = VIAL_FILL / "labels" / "video5_vial_recipes.csv"
FRAME_LABELS = VIAL_FILL / "labels" / "video5_frame_labels.csv"
OUT = ROOT / "outputs"

# (x, y, w, h) in the 640x480 vial_fill frame: liquid pooled at the vial base.
LIQUID_ROI = (295, 375, 32, 22)
# Black vial-holder block; dark here means the vial carrier is in position.
HOLDER_ROI = (265, 210, 110, 45)

# Pixel mask
MIN_L, MAX_L = 3.0, 97.0  # drop crushed blacks / blown highlights
MIN_VALID_PIXELS = 100
# Carrier considered present when this fraction of holder pixels is dark.
HOLDER_DARK_L = 25.0
HOLDER_MIN_DARK_FRAC = 0.6

SAMPLE_STRIDE_S = 1.0  # seconds between measured video frames

# Stability (frame counts assume SAMPLE_STRIDE_S)
STABLE_MAX_STEP_DE = 1.5  # consecutive-frame dE00 allowed inside a window
STABLE_MIN_FRAMES = 5

# Reference / thresholds
MIN_REFERENCE_VIALS = 2
TOLERANCE_FLOOR_DE = 2.3  # ~ just-noticeable difference; lower bound only
TOLERANCE_MARGIN = 1.5  # multiple of observed spread among settled vials
SETTLE_CONSECUTIVE_VIALS = 1  # consecutive in-range vials to declare settled

# Cycle segmentation: an "empty vial" frame has a bright ROI (bare vial base).
EMPTY_MIN_L = 33.0
EMPTY_MIN_FRAMES = 5
