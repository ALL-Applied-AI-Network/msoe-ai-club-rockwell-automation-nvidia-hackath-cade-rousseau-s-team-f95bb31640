# Digital twin

Owners: Cade, Vlad.

The scripts here run inside Isaac Sim's own Python on Brev. They do not use the uv environment, so don't expect `uv sync` packages to be available.

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
