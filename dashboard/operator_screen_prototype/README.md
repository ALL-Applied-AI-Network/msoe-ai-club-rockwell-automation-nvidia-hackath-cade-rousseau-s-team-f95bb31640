# Changeover Guard — Operator Screen Prototype

This is a standalone Streamlit UI prototype with simulated data. It does not connect to a PLC, camera, or optimization model.

## Run

From the repository root:

```bash
uv sync
uv run streamlit run dashboard/operator_screen_prototype/operator_screen.py
```

## Model output (JSON)

Tick **Use JSON model output** to show `inspection_result.json` from this folder instead of a demo case. The file is gitignored. Any field you leave out shows as blank, never as demo text. Unknown keys are listed on screen.

```json
{
  "vial": 7,
  "recipe": "7",
  "previous": "6",
  "expected": "Blue",
  "observed": "Green-tinted",
  "expected_rgb": [45, 125, 192],
  "observed_rgb": [108, 152, 102],
  "status": "FAIL",
  "cause": "Leftover orange from Recipe 6",
  "action": "Inspect and segregate the vial.",
  "fill": "68 mL measured / 70 mL target"
}
```

Colors are sRGB 0-255 lists, the same format as `contracts.py`. Any `status` other than `PASS` is shown as flagged.

## Integration plan

- Keep the UI independent of the vision and optimization algorithms.
- For optimization, provide the original recipe list, suggested recipe list, and model-computed improvement metrics.
- Only display verified model results; the example optimization is illustrative.
- Preserve read-only operation until machine integration has been separately reviewed and tested.
