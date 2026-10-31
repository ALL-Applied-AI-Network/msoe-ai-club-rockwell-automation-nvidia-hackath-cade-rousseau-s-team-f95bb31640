# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Changeover guard": a hackathon project (Rockwell Automation + NVIDIA, Fluid Mixing Station 3 dataset). It reads tank and vial colors from camera frames, predicts the color of the first vial after a recipe changeover (leftover liquid from the previous recipe), and explains or advises on failures. Python >= 3.11, managed with `uv`. Final submission is due Nov 1, 2026; the README is to be replaced with a project write-up before then.

## Commands

```bash
uv sync                                   # install deps (incl. dev: pytest, ruff)
uv run pytest                             # all tests (testpaths = tests/, eval/)
uv run pytest tests/test_measure.py::test_name   # single test
uv run ruff check .                       # lint (no ruff config; defaults)
uv run uvicorn api.main:app --reload     # FastAPI backend on :8000; also serves web/out once built
cd web && bun dev                        # Next.js operator UI on :3000 (proxies /api to :8000)
cd web && bun run build                  # static export to web/out
uv run streamlit run dashboard/app.py     # older Streamlit dashboard and ROI calibration tools

# Digital twin pipeline (run from repo root)
uv run python twin/calibrate.py           # data/ video 5 -> twin/calibration.json (committed)
uv run python twin/manifest.py            # every valid recipe x 3 vials -> synthetic/manifest.csv
uv run python twin/fake_render.py         # quick 2D stand-in -> synthetic/ (video_id 100)
uv run python twin/scenarios.py           # labeled robustness/decay/demo sets -> synthetic/scenarios.csv
# Run twin/render.py inside Isaac Sim on Brev; see twin/BREV.md.

graphify update .                         # refresh graphify-out/ code graph (no LLM cost)
```

`data/` (real dataset, unzipped from the event guide; must contain `data/vial_fill/image_index.csv`) and `synthetic/` are gitignored. If `uv run pytest` reports 3 skipped, the data is missing or in a wrapper folder. `*.jpg` and `*.mp4` are gitignored too. The exception is `data/10_29_26/` (October 29 session stills and labels), which is committed; its videos are in the `data-10-29-26` GitHub release (`gh release download data-10-29-26 && unzip 10_29_26_videos.zip`). `analysis/10_29_26/` holds the committed transition-workflow outputs; `uv run --group analysis python scripts/analyze_transitions.py` writes a fresh run to `outputs/transition_analysis/<run>/`, which you copy over them to update.

## Architecture

The pipeline is a chain of modules in `src/changeover_guard/` whose only coupling is the dataclasses in `contracts.py`:

`dataset.frames()` -> `measure` (`TankReading` / `VialMeasurement`) -> `mixing.predict()` (`Prediction`) -> `advise.check()` / `advise.advise()` (`Check`, `Advice`) -> `api/main.py` (FastAPI, JSON shapes in `api/schemas.py`) -> `web/` (Next.js), and `eval/`.

- `contracts.py` is the shared interface and also holds the domain constants: usable tanks (`T100`, `T200`, `T400`; T300 is out of service), dispense order, mL limits enforced by `Recipe.validate()`, the color/vial/tank-fill warning thresholds, and CIEDE2000 helpers for sRGB 0-255. The README requires telling the whole team before changing it.
- Colors are RGB 0-255 tuples everywhere in contracts; OpenCV images are BGR, so `measure.py` flips at the boundary.
- `measure.tanks()` selects either the wide top-view or 4:3 real/front-view ROI calibration by aspect ratio, scales it to the frame, and takes a median over non-shadow, non-glare pixels. `measure.tank_fill_levels()` uses separate four-tank ROIs and a horizontal-transition scanline. `measure.vial()`, `mixing.predict()`, `advise.check()` and `advise.advise()` are still `NotImplementedError` stubs; their signatures are the contract.
- `dashboard/app.py` uses the public `tank_rois()` / `fill_rois()` helpers so overlays match measurement. Its calibration panel also displays the private `_FILL_ROIS` source values and reloads `measure.py` after edits.
- `generated/images/` contains top-view color, front-view color, and tank-fill fixtures plus editable JSON targets. They support tests and the dashboard demonstration; they are not digital-twin training data.

### Real data vs. synthetic twin

Real and synthetic data share one directory layout so the same loader works on both: `<root>/{vial_fill,tank_level}/images/`, `<root>/<camera>/image_index.csv` (`filename,timestamp,video_id`), and `vial_fill/labels/`. `dataset.frames(root, video_id)` joins the two cameras on **exact timestamp equality**; a mismatch silently drops the frame. Training goes on `synthetic/`, evaluation on `data/` (see `eval/README.md`).

`twin/` is separate from the package:
- `twin/writer.py` is stdlib-only because it must run both under `uv` and inside Isaac Sim's own Python (on Brev). `fake_render.py` is the quick 2D stand-in; `render.py` performs the real Isaac Sim render workflow documented in `BREV.md`.
- `calibrate.py` fits a Beer-Lambert color model from video 5 into `calibration.json`; `manifest.py` consumes it. `manifest.py` and `calibrate.py` import `changeover_guard`, so run them via `uv run`.
- `render.py --calibrate` measures the material-to-pixel response into `material_lut.json`; ordinary render runs invert that LUT for requested liquid colors.
- `twin/README.md` documents the Isaac Sim USD prim paths, the exact CSV column contracts the loader needs, and the rule to keep measurement regions above y = 450 (real frames have a burned-in timestamp). Read it before touching twin output.
- `twin/runs.csv` records three Brev sessions, including the completed 462-vial / 1,848-frame render. Add a row for every future session.

### Tests

`tests/` imports `changeover_guard` directly and `twin/writer.py` via `sys.path`. Some tests need the real `data/` and skip without it.

## Knowledge graph

`graphify-out/` (GRAPH_REPORT.md, graph.json, graph.html) holds a code graph of this repo. Check `GRAPH_REPORT.md` for god nodes and communities before broad searches, and run `graphify update .` after code changes.
