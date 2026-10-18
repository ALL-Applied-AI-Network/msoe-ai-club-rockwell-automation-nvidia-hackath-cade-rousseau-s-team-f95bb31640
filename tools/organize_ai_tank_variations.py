"""Copy AI-generated tank variants into the project and write their RGB targets."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]
AI_DIR = Path(r"C:\Users\vassark\.codex\generated_images\01a14b35-0a8b-74f0-9f8f-97c17c07ecd2")
OUTPUT_DIR = ROOT / "generated" / "images"
MANIFEST = ROOT / "generated" / "ai_tank_colors.json"

# The first two images were generated before the parallel batch. The remaining
# paths correspond to the batch prompts in their original prompt order.
SOURCE_FILES = [
    "exec-d4d80db5-4bc7-40bb-a314-ac071e75615e.png",
    "exec-63ef49ed-b6f9-4e0c-8a18-d87efc8dbe16.png",
    "exec-53f5f751-e27c-44dd-be7a-4d101da99bfc.png",
    "exec-6ae2239e-e71b-4ce1-894a-6074aa54faf3.png",
    "exec-7137f39a-3b79-4a27-8560-a069b8520109.png",
    "exec-9e8e739c-032d-4cff-a20e-7e07b68717ab.png",
    "exec-6e4531d9-b072-48e3-863b-949fb33f8017.png",
    "exec-b173f185-bcdc-486e-bdcb-9ef47b485665.png",
    "exec-ef57ea8a-eb7f-4530-8846-96e15dffb455.png",
    "exec-d8a240aa-1a07-4f1b-a213-db25539a6b82.png",
    "exec-1fc717bb-3256-4a3f-9087-cdd874c5b920.png",
    "exec-2e63deb5-b729-4c05-bd59-48b56dcba02c.png",
    "exec-781949aa-6fdb-480c-b8f2-945a2232dcb3.png",
    "exec-9449391a-e752-4313-90af-1407188c1594.png",
    "exec-718133e9-bf46-4436-9a93-554c3b5df099.png",
    "exec-a9692cff-f2a4-479a-a346-88478a78ceb5.png",
    "exec-f1c5145a-26e3-48a9-b92e-f6f2fe7d2b80.png",
    "exec-f12cfdd3-c994-4517-9d6b-0d76892a57bd.png",
    "exec-ba263b7b-08f8-4e58-acc2-1a54c67f0217.png",
    "exec-62fffae5-3a4b-43f9-b87b-d29d19f2d512.png",
]

REQUESTED_RGB = [
    {"T100": (160, 220, 220), "T200": (245, 140, 20), "T400": (20, 80, 230)},
    {"T100": (235, 225, 150), "T200": (235, 70, 20), "T400": (20, 50, 180)},
    {"T100": (170, 230, 170), "T200": (245, 140, 20), "T400": (20, 100, 230)},
    {"T100": (235, 225, 150), "T200": (220, 140, 30), "T400": (30, 70, 190)},
    {"T100": (170, 220, 240), "T200": (235, 70, 20), "T400": (20, 50, 180)},
    {"T100": (220, 220, 220), "T200": (235, 180, 20), "T400": (20, 170, 200)},
    {"T100": (210, 190, 230), "T200": (245, 140, 20), "T400": (80, 30, 180)},
    {"T100": (160, 220, 220), "T200": (220, 100, 20), "T400": (20, 150, 170)},
    {"T100": (220, 220, 220), "T200": (235, 40, 30), "T400": (20, 90, 230)},
    {"T100": (170, 230, 170), "T200": (245, 140, 20), "T400": (130, 30, 180)},
    {"T100": (170, 210, 240), "T200": (245, 140, 20), "T400": (220, 30, 150)},
    {"T100": (235, 225, 150), "T200": (245, 140, 20), "T400": (30, 170, 80)},
    {"T100": (190, 220, 240), "T200": (220, 120, 30), "T400": (40, 170, 240)},
    {"T100": (160, 220, 220), "T200": (180, 60, 20), "T400": (30, 60, 170)},
    {"T100": (170, 230, 170), "T200": (235, 70, 20), "T400": (20, 160, 190)},
    {"T100": (220, 220, 220), "T200": (245, 140, 20), "T400": (40, 30, 150)},
    {"T100": (170, 230, 200), "T200": (245, 140, 20), "T400": (30, 150, 150)},
    {"T100": (235, 225, 150), "T200": (235, 70, 20), "T400": (30, 150, 150)},
    {"T100": (160, 220, 220), "T200": (235, 140, 20), "T400": (20, 100, 220)},
    {"T100": (170, 230, 170), "T200": (235, 70, 20), "T400": (30, 160, 120)},
]


def main() -> None:
    if len(SOURCE_FILES) != 20 or len(REQUESTED_RGB) != 20:
        raise ValueError("expected exactly 20 AI images and 20 RGB records")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (source_name, rgb) in enumerate(zip(SOURCE_FILES, REQUESTED_RGB), start=1):
        source = AI_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        filename = f"ai_tank_variant_{index:02d}.png"
        shutil.copy2(source, OUTPUT_DIR / filename)
        records.append({"filename": filename, "requested_rgb": rgb})

    MANIFEST.write_text(json.dumps({"images": records}, indent=2) + "\n", encoding="utf-8")
    print(f"copied {len(records)} AI images to {OUTPUT_DIR}")
    print(f"RGB manifest written to {MANIFEST}")


if __name__ == "__main__":
    main()
