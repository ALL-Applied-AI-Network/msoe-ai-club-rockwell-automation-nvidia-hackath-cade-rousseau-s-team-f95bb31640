"""Write a synthetic root in the data/ layout. Stdlib only: runs in uv and in Isaac Sim's Python.

One root holds one video: calling write_root again on the same root replaces its CSVs.
"""
import csv
from datetime import datetime, timedelta
from pathlib import Path

START = datetime(2026, 1, 1)
STEP_S = 2.0                                 # real stills are sampled about every 2 s
TANKS = ("T100", "T200", "T400")


def write_root(root: str | Path, video_id: int, frames: list[dict],
               recipes: dict[int, dict[str, float]]) -> list[tuple[Path, Path]]:
    """Write image indexes and labels; return [(vial_fill_path, tank_level_path)] to save images to.

    frames: in time order, dicts with in_fill (0/1), vial and recipe_id (None between fills).
    recipes: {vial: {"T100": mL, ...}} with only the tanks in use.
    """
    root = Path(root)
    labels = root / "vial_fill" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    paths, rows, span = [], [], {}
    for i, f in enumerate(frames):
        ts = (START + timedelta(seconds=i * STEP_S)).strftime("%Y-%m-%d %H:%M:%S.%f")
        names = {cam: f"{cam}_v{video_id}_{i:06d}.jpg" for cam in ("vial_fill", "tank_level")}
        rows.append((ts, names, f))
        paths.append(tuple(root / cam / "images" / n for cam, n in names.items()))
        if f["in_fill"]:
            start, _, rid = span.get(f["vial"], (i * STEP_S, None, f["recipe_id"]))
            span[f["vial"]] = (start, i * STEP_S, rid)

    for cam in ("vial_fill", "tank_level"):
        (root / cam / "images").mkdir(parents=True, exist_ok=True)
        with open(root / cam / "image_index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "timestamp", "video_id"])
            w.writerows((names[cam], ts, video_id) for ts, names, _ in rows)

    blank = lambda v: "" if v is None else v
    with open(labels / f"video{video_id}_frame_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "in_fill", "vial", "recipe_id"])
        w.writerows((names["vial_fill"], f["in_fill"], blank(f["vial"]), blank(f["recipe_id"]))
                    for _, names, f in rows)

    # dataset.vial_labels() only reads this name, whatever the video_id
    with open(labels / "video5_vial_recipes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "start_s", "end_s"])
        for vial, (start, end, rid) in sorted(span.items()):
            w.writerow([vial, rid, *(blank(recipes[vial].get(t)) for t in TANKS), start, end])
    return paths
