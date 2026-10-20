# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Changeover guard": a hackathon project (Rockwell Automation + NVIDIA, Fluid Mixing Station 3 dataset). It reads tank and vial colors from camera frames, predicts the color of the first vial after a recipe changeover (leftover liquid from the previous recipe), and explains or advises on failures. Python >= 3.11, managed with `uv`. Final submission is due Nov 1, 2026; the README is to be replaced with a project write-up before then.

## Commands

```bash
uv sync                                   # install deps (incl. dev: pytest, ruff)
uv run pytest                             # all tests (testpaths = tests/)
uv run pytest tests/test_measure.py::test_name   # single test
uv run ruff check .                       # lint (no ruff config; defaults)
uv run streamlit run dashboard/app.py     # dashboard

# Digital twin pipeline (run from repo root)
uv run python twin/calibrate.py           # data/ video 5 -> twin/calibration.json (committed)
uv run python twin/manifest.py --limit 60 # -> synthetic/manifest.csv
uv run python twin/fake_render.py         # -> synthetic/ (video_id 100)

graphify update .                         # refresh graphify-out/ code graph (no LLM cost)
```

`data/` (real dataset, unzipped from the event guide; must contain `data/vial_fill/image_index.csv`) and `synthetic/` are gitignored. If `uv run pytest` reports 3 skipped, the data is missing or in a wrapper folder. `*.jpg` and `*.mp4` are gitignored too.

## Architecture

The pipeline is a chain of modules in `src/changeover_guard/` whose only coupling is the dataclasses in `contracts.py`:

`dataset.frames()` -> `measure` (`TankReading` / `VialMeasurement`) -> `mixing.predict()` (`Prediction`) -> `advise.check()` / `advise.advise()` (`Check`, `Advice`) -> `dashboard/app.py` and `eval/`.

- `contracts.py` is the shared interface and also holds the domain constants: usable tanks (`T100`, `T200`, `T400`; T300 is out of service), dispense order, mL limits enforced by `Recipe.validate()`, the `COLOR_WARN_DE` / `FILL_TOL_ML` thresholds, and `delta_e()` (CIEDE2000 on sRGB 0-255). The README requires telling the whole team before changing it.
- Colors are RGB 0-255 tuples everywhere in contracts; OpenCV images are BGR, so `measure.py` flips at the boundary.
- `measure.tanks()` reads fixed per-tank ROIs (`_TANK_ROIS`, defined for 640x480 and scaled for other sizes) using a median over non-shadow, non-glare pixels. `measure.vial()`, `mixing.predict()`, `advise.check()` and `advise.advise()` are still `NotImplementedError` stubs; their signatures are the contract.
- `dashboard/app.py` imports the private `_TANK_ROIS` / `_REFERENCE_SIZE` from `measure` to draw ROI overlays, so renaming them breaks the dashboard.
- `tools/generate_tank_variations.py` and `tools/organize_ai_tank_variations.py` build tank-color test images into `generated/` (consumed by the dashboard). The ROIs hard-coded in `generate_tank_variations.py` currently differ from `measure._TANK_ROIS` despite its comment saying they match.

### Real data vs. synthetic twin

Real and synthetic data share one directory layout so the same loader works on both: `<root>/{vial_fill,tank_level}/images/`, `<root>/<camera>/image_index.csv` (`filename,timestamp,video_id`), and `vial_fill/labels/`. `dataset.frames(root, video_id)` joins the two cameras on **exact timestamp equality**; a mismatch silently drops the frame. Training goes on `synthetic/`, evaluation on `data/` (see `eval/README.md`).

`twin/` is separate from the package:
- `twin/writer.py` is stdlib-only because it must run both under `uv` and inside Isaac Sim's own Python (on Brev). `twin/render.py` (Isaac side) and `BREV.md` are referenced in docs but do not exist yet; `fake_render.py` is the 2D stand-in.
- `calibrate.py` fits a Beer-Lambert color model from video 5 into `calibration.json`; `manifest.py` consumes it. `manifest.py` and `calibrate.py` import `changeover_guard`, so run them via `uv run`.
- `twin/README.md` documents the Isaac Sim USD prim paths, the exact CSV column contracts the loader needs, and the rule to keep measurement regions above y = 450 (real frames have a burned-in timestamp). Read it before touching twin output.
- Log each Brev session in `twin/runs.csv` (`date,gpu,hours,credits,purpose`).

### Tests

`tests/` imports `changeover_guard` directly and `twin/writer.py` via `sys.path`. Some tests need the real `data/` and skip without it.

## Knowledge graph

`graphify-out/` (GRAPH_REPORT.md, graph.json, graph.html) holds a code graph of this repo. Check `GRAPH_REPORT.md` for god nodes and communities before broad searches, and run `graphify update .` after code changes.
