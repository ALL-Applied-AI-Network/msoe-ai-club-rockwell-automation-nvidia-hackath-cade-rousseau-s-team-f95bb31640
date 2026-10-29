"""python -m color_monitor [--refresh]   (writes everything to outputs/)"""
import sys

import pandas as pd

from . import config as C
from .audit import audit_report, build_vial_table, load_catalog
from .changeover import vial_report
from .evaluate import evaluate
from .extract import extract_frames
from .overlay import save_overlays
from .reference import build_references
from .segment import measure_cycles


def main():
    C.OUT.mkdir(exist_ok=True)
    frames_csv = C.OUT / "frames.csv"
    if "--refresh" in sys.argv or not frames_csv.exists():
        extract_frames().to_csv(frames_csv, index=False)
    frames = pd.read_csv(frames_csv)

    catalog = load_catalog()
    cycles = measure_cycles(frames)
    vials = build_vial_table(cycles)
    refs = build_references(vials, catalog)
    report = vial_report(vials, catalog)

    cycles.to_csv(C.OUT / "cycles.csv", index=False)
    vials.to_csv(C.OUT / "vial_table.csv", index=False)
    refs.to_csv(C.OUT / "recipe_references.csv")
    report.to_csv(C.OUT / "vial_report.csv", index=False)
    save_overlays(cycles, C.OUT / "roi_overlays.png")

    text = "\n\n".join([audit_report(frames, cycles, vials), evaluate(vials, catalog, report)])
    (C.OUT / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\nVIAL REPORT\n" + report.drop(columns=["stable_n_frames", "changeover_note"]).round(1).to_string(index=False))


main()
