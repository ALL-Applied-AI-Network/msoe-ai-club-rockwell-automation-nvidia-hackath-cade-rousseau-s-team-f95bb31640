"""Load the real dataset (or a synthetic twin with the same layout)."""
from pathlib import Path

import pandas as pd

from changeover_guard.contracts import Recipe

_CAMERAS = ("vial_fill", "tank_level")


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.str.replace(r"^(\d+):(\d+):(\d+) ", r"\1-\2-\3 ", regex=True), format="mixed")


def _index(root: Path, cam: str, video_id: int) -> pd.DataFrame:
    df = pd.read_csv(root / cam / "image_index.csv", dtype={"video_id": str})
    df = df[df["video_id"] == str(video_id)]
    return pd.DataFrame({"timestamp": _ts(df["timestamp"]), cam: df["filename"]})


def frames(root: str | Path, video_id: int) -> pd.DataFrame:
    root = Path(root)
    df = _index(root, "vial_fill", video_id).merge(_index(root, "tank_level", video_id), on="timestamp")
    lab = root / "vial_fill" / "labels" / f"video{video_id}_frame_labels.csv"
    if lab.exists():
        cols = ["in_fill", "vial", "recipe_id"]
        labels = pd.read_csv(lab, usecols=["filename", *cols]).astype({c: "Int64" for c in cols})
        df = df.merge(labels.rename(columns={"filename": "vial_fill"}), on="vial_fill", how="left")
    for cam in _CAMERAS:
        df[cam] = [str(root / cam / "images" / f) for f in df[cam]]
    return df.sort_values("timestamp").reset_index(drop=True)


def vial_labels(root: str | Path) -> list[tuple[int, Recipe, float, float]]:
    df = pd.read_csv(Path(root) / "vial_fill" / "labels" / "video5_vial_recipes.csv").sort_values("vial")
    out = []
    for row in df.itertuples():
        ml = {c[:4].upper(): float(getattr(row, c)) for c in ("t100_ml", "t200_ml", "t400_ml")
              if pd.notna(getattr(row, c))}
        recipe = Recipe(int(row.recipe_id), ml)
        recipe.validate()
        out.append((int(row.vial), recipe, float(row.start_s), float(row.end_s)))
    return out
