# Changeover Guard — Operator Screen Prototype

This is a standalone Streamlit UI prototype with simulated data. It does not connect to a PLC, camera, or optimization model.

## Run

From a terminal with Python 3.11+:

```bash
pip install streamlit
streamlit run operator_screen.py
```

## Integration plan

- Keep the UI independent of the vision and optimization algorithms.
- Feed it the **active** recipe ID, previous recipe ID, expected RGB, observed RGB, inspection pass/fail, and an optional cause/recommendation.
- For optimization, provide the original recipe list, suggested recipe list, and model-computed improvement metrics.
- Only display verified model results; the example optimization is illustrative.
- Preserve read-only operation until machine integration has been separately reviewed and tested.

For the existing repository, this file can be placed alongside `dashboard/app.py` and run separately. The project already includes Streamlit as a dependency.
