from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from changeover_guard.dataset import frames, vial_labels

DATA = Path(__file__).parents[1] / "data"
needs_data = pytest.mark.skipif(not (DATA / "vial_fill").exists(), reason="dataset not unzipped into data/")


def test_frames_synthetic_root(tmp_path):
    (tmp_path / "vial_fill").mkdir()
    (tmp_path / "tank_level").mkdir()
    pd.DataFrame({
        "filename": ["v0.jpg", "v1.jpg", "v2.jpg"],
        "timestamp": ["2026-09-22 10:00:00.500000", "2026-09-22 10:00:01", "2026-09-22 10:00:02.000000"],
        "video_id": ["5", "5", "unassigned"],
    }).to_csv(tmp_path / "vial_fill" / "image_index.csv", index=False)
    pd.DataFrame({
        "filename": ["t0.jpg", "t1.jpg", "t2.jpg"],
        "timestamp": ["2026:9:22 10:00:00.500000", "2026:9:22 10:00:01", "2026:9:22 10:00:02.000000"],
        "video_id": ["5", "5", "5"],
    }).iloc[[1, 2]].to_csv(tmp_path / "tank_level" / "image_index.csv", index=False)
    df = frames(tmp_path, 5)
    assert len(df) == 1
    assert "in_fill" not in df.columns
    (tmp_path / "vial_fill" / "labels").mkdir()
    (tmp_path / "vial_fill" / "labels" / "video5_frame_labels.csv").write_text(
        "filename,in_fill,vial,recipe_id\nv1.jpg,0,,\n")
    df = frames(tmp_path, 5)
    assert all(str(df[c].dtype) == "Int64" for c in ("in_fill", "vial", "recipe_id"))
    assert df["in_fill"][0] == 0 and pd.isna(df["vial"][0])
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert Path(df["vial_fill"][0]).parts[-3:] == ("vial_fill", "images", "v1.jpg")


@needs_data
def test_vial_labels():
    labels = vial_labels(DATA)
    assert [v for v, *_ in labels] == list(range(1, 31))
    assert Counter(r.recipe_id for _, r, _, _ in labels) == {rid: 3 for rid in range(5, 15)}


@needs_data
def test_vial_labels_pair_recipe():
    r = vial_labels(DATA)[3][1]
    assert r.ml == {"T100": 30, "T200": 10}
    assert r.recipe_id == 6


@needs_data
def test_frames_video5():
    df = frames(DATA, 5)
    assert len(df) == 845
    assert all(Path(p).exists() for p in df["vial_fill"])
    assert all(Path(p).exists() for p in df["tank_level"])
    assert df["in_fill"].sum() == 509
