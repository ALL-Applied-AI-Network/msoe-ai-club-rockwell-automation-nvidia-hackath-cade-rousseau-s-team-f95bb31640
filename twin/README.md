# Digital twin

Owners: Cade, Vlad.

`render.py` runs inside Isaac Sim's own Python on Brev, where `uv sync` packages aren't available. The other scripts here run locally with `uv run`.

The twin is the organizers' `csi_digital_twin_v1`. Unzip it to `data/csi_digital_twin_v1/` (gitignored, 714 MB). We only use `stations/fill_station/fill_station.usda`: it loads in seconds and has its own camera and lights.

## Run

```bash
uv run python twin/calibrate.py               # liquid colors from video 5 -> twin/calibration.json (committed)
uv run python twin/manifest.py --limit 60     # vials to render -> synthetic/manifest.csv
uv run python twin/fake_render.py             # stand-in 2D renders -> synthetic/ (video_id 100)
```

Then load the output with `dataset.frames("synthetic", 100)`.

## Twin knobs (fill station)

Found with `uv run --with usd-core python twin/inspect_usd.py data/csi_digital_twin_v1/stations/fill_station/fill_station.usda`. All prim paths below start with `/FillStation/FillStation_LocalRoot/`.

- **Camera:** `FillCam/FillCam`, a 640×480 pinhole already matched to the real `vial_fill` camera (see `cameras/FillCam.json`). Nothing to tune.
- **Liquid:** `Fill_liquid/Fill_liquid`, a mesh whose points are animated:
  - time codes 21–368: blue pours
  - 586–661: clear is added
  - 686: finished vial
- **Recipe the animation plays:** the blue-to-total height ratio is 0.72, so it's recipe 7 (50 mL T400 + 20 mL T100 = 70 mL).
- **Liquid size at 686:** 26.4 mm across and 123 mm tall, which is about 0.57 mL per mm of height.
- **Changing the fill level:** scale the `Fill_liquid` Xform in Z by `target_ml / 70`, or pick an earlier time code.
- **Liquid color:** the material `/FillStation/_materials/USD_blue_liquid_approximation` is a plain UsdPreviewSurface with `diffuseColor` (0.006, 0.2, 0.3), `opacity` 0.6, `ior` 1.333 and `roughness` 0.12. There's no volume absorption, so a material color does **not** map directly to a pixel color. That mapping has to be calibrated by rendering (see `BREV.md`).
- **Stream:** `Fill_dispensing_stream` uses the material `USD_jet_approximation` and is visible during the pours.
- **Lights:** these RectLights are the domain-randomization knobs (current intensity in brackets):
  - `Fill_overhead_panel` (557)
  - `V2_base_reflection` (350)
  - `Fill_front_softbox` (207)
  - `Fill_right_reflection` (143)
  - `V2_guard_bounce` (70)

  Plus the DomeLight `/FillStation/env_light` (0.18). All are 6500 K.
- **Semantic labels:** `Fill_vial_active` carries `userProperties:synthetic_class = vial` (id 1).
- **Real frames differ:** they have a timestamp burned into the bottom ~25 px. Synthetic frames don't, so keep measurement regions above y = 450.

## Output

Write output to `synthetic/` (gitignored). Use the same layout as `data/`:

```
synthetic/
  <camera>/images/
  <camera>/image_index.csv     # columns: filename,timestamp,video_id
  vial_fill/labels/
```

## What the loader needs

- The `vial_fill` and `tank_level` rows for the same frame must have identical timestamps. The cameras are joined on exact equality, and any mismatch silently drops the frame.
- The `video_id` column must equal the ID passed to `frames(root, video_id)`.
- Frame labels go in `vial_fill/labels/video{N}_frame_labels.csv` with the columns `filename,in_fill,vial,recipe_id`, keyed by the vial_fill filename. Use `in_fill` 0/1, and leave `vial` and `recipe_id` blank between fills.
- Recipes go in `vial_fill/labels/video5_vial_recipes.csv` with the columns `vial,recipe_id,t100_ml,t200_ml,t400_ml,start_s,end_s`. A blank mL cell means the tank is unused, and every recipe must pass `Recipe.validate()`.

## Brev costs

- Add a line to `runs.csv` for every Brev session: `date,gpu,hours,credits,purpose`.
- Stop the instance when you aren't using it.
