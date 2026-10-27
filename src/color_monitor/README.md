# Vial Color Monitor

A tool that watches the vial-fill videos and checks whether each finished vial is the color its recipe says it should be. It also warns when the recipe changes, because the first vial after a change is often contaminated by leftover liquid from the previous recipe.

## The idea in plain terms

Each recipe mixes liquids from three tanks (T100, T200, T400) in set amounts. If a recipe is made correctly, its vials should always come out about the same color. So the tool:

1. Measures the color of every vial.
2. Learns what each recipe normally looks like from the vials we already have.
3. Flags any vial whose color is too far from its recipe's usual color.

It does not predict colors from theory. A recipe's "expected color" comes only from real vials of that recipe.

## How "color difference" is measured

- Color is stored as three numbers, **L** (lightness), **a** (green to red) and **b** (blue to yellow). This is the CIELAB system.
- The distance between two colors is **ΔE00** (pronounced "delta E"), a standard single number for how different two colors look to a person.
  - About 1 = barely noticeable.
  - Bigger = more visibly different.
- Our tolerance is at least 2.3, or 1.5 times the natural spread seen in a recipe's vials, whichever is larger.

## What the tool does, in order

1. **Reads the videos.** It takes one frame per second. In each frame it looks at a small fixed box at the bottom of the vial and records the median color there. It also checks that the black vial holder is in place, so it knows a vial is present.
2. **Finds each vial.** From the color signal it detects the cycle: empty vial, liquid arrives, color settles, vial is swapped out.
3. **Finds the settled color.** The settled color is the last stretch of at least 5 frames where the color barely changes between frames (step of 1.5 ΔE00 or less). The median color of that stretch is the vial's color. If no such stretch exists, the vial is marked "color measurement uncertain" and no color is guessed.
4. **Matches vials to recipes.** Video 5 is the only video with recipe labels. The nth vial seen in the video is matched to the nth labeled vial, and the timing is checked.
5. **Builds a reference color per recipe.** It averages settled vials of that recipe, skipping the first vial after a recipe change. A recipe needs at least 2 such vials, otherwise it reports "insufficient data". Recipes outside the known catalog are rejected.
6. **Judges every vial.** Each vial is compared to its recipe's reference, built without that vial so it can't be graded against itself. The verdict is one of:
   - **consistent**
   - **OUT OF RANGE**
   - **no reference (insufficient data)**
   - **color measurement uncertain**
7. **Watches changeovers.** When the recipe changes, the first vial gets a "WATCH" note, and the tool records when the color settles back to normal.
8. **Evaluates itself.** On video 5 it counts false alarms, false passes and which recipes look alike.

There is also an optional tank-identification function. It takes three single-tank 50 mL vials and accepts the tank's base color only if the three agree within a tolerance you set. It is written but has not been tested, because there is no data for it yet.

## How to run it

From this folder:

```
python -m color_monitor
```

- The first run reads all the videos and is slow. Later runs reuse a saved copy (`outputs/frames.csv`).
- Add `--refresh` to re-read the videos, for example after changing settings.
- It needs the Python packages `opencv-python`, `numpy`, `pandas` and `scikit-image`.

## What you get

Everything is written to the `outputs/` folder.

| File | What it is |
|---|---|
| `vial_report.csv` | Start here. One row per vial: recipe, measured color, ΔE00 to the reference, verdict and changeover note. |
| `report.txt` | The data audit and the evaluation results. |
| `recipe_references.csv` | Expected color and tolerance for each recipe. |
| `vial_table.csv` | Vials matched to labels, with timing details. |
| `cycles.csv` | Every vial cycle detected in all 5 videos. |
| `roi_overlays.png` | The measurement box drawn on every vial, for a quick visual check. |

## What we found

- **Use the videos, not the still images.** The still images' timestamps don't line up with the labels. The videos do.
- **The first vial after a recipe change is usually wrong.** It is 6 to 17 ΔE00 off, from leftover liquid of the previous recipe. By the second vial, the color is back to normal. The tool flags this correctly.
- **Some wrong colors slip through.** About 4% of the time a vial passes as a different recipe. Recipes 5, 8, 9 and 11 can look alike.

## Limitations (please read)

- **Little data per recipe.** Only 2 clean vials per recipe, so the reference colors and tolerances are weak. The scores for those same vials look better than they would on new vials.
- **Two recipes have no reference.** Recipes 6 and 14 don't have enough usable vials.
- **Only one labeled video.** Only video 5 has recipe labels, so all references and results come from that single run.
- **Video 3 is mostly unusable.** It has a different scene.
- **The measurement box is placed by hand.** It does not adapt if the camera moves.
- **No true good/bad labels.** "False alarm" here means "different from the other vials of the same recipe", not "the vial was actually bad".
- **Recorded video only.** It does not run on a live feed.

The most useful next step is more labeled vials per recipe, ideally from a second labeled run.

## Where the code is

`color_monitor/` holds the code, one file per step:

- `extract.py` and `color.py`: read frames and measure color.
- `segment.py`: find vials and their settled color.
- `audit.py`: match vials to recipe labels.
- `reference.py`: recipe reference colors and the verdicts.
- `changeover.py`: per-vial report and recipe-change warnings.
- `evaluate.py`: self-test results.
- `tank_id.py`: optional tank-identification function.
- `overlay.py`: the visual check image.
- `config.py`: all the adjustable settings.
