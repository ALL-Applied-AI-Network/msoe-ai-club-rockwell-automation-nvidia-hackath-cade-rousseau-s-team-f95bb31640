import sys
from pathlib import Path

from changeover_guard.dataset import frames, vial_labels

sys.path.insert(0, str(Path(__file__).parents[1] / "twin"))
from writer import write_root


def test_write_root_round_trip(tmp_path):
    fill = lambda vial, rid: {"in_fill": 1, "vial": vial, "recipe_id": rid}
    idle = {"in_fill": 0, "vial": None, "recipe_id": None}
    seq = [idle, fill(1, 7), fill(1, 7), idle, fill(2, 13), idle]
    paths = write_root(tmp_path, 100, seq, {1: {"T100": 20, "T400": 50}, 2: {"T200": 50, "T400": 20}})
    for vf, tl in paths:
        vf.write_bytes(b"")
        tl.write_bytes(b"")

    df = frames(tmp_path, 100)
    assert len(df) == len(seq)
    assert df["in_fill"].tolist() == [0, 1, 1, 0, 1, 0]
    assert df["vial"].isna().tolist() == [True, False, False, True, False, True]
    assert df["recipe_id"].dropna().tolist() == [7, 7, 13]
    assert [Path(p) for p in df["vial_fill"]] == [vf for vf, _ in paths]

    labels = vial_labels(tmp_path)
    assert [(v, r.recipe_id, r.ml, s, e) for v, r, s, e in labels] == [
        (1, 7, {"T100": 20, "T400": 50}, 2.0, 4.0),
        (2, 13, {"T200": 50, "T400": 20}, 8.0, 8.0),
    ]
