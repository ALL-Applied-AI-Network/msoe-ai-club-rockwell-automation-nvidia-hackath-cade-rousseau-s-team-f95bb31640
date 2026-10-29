# Operator UI

Next.js (React, TypeScript, Tailwind) front end for Changeover Guard. All data comes from the FastAPI backend in `../api/`, which wraps the `changeover_guard` pipeline. Owners: Mat, Vlad.

## Develop

Two terminals, from the repo root:

```bash
uv run uvicorn api.main:app --reload     # API on :8000 (endpoints listed at /docs)
cd web && npm install && npm run dev     # UI on :3000; /api calls are forwarded to :8000
```

The real dataset must be unzipped into `data/` (see the root README).

## Demo / deploy

```bash
cd web && npm run build                  # static files in web/out
uv run uvicorn api.main:app              # serves the UI and the API at http://localhost:8000
```

One process, one URL, so it can run on the CSI workstation or the DGX Spark.

## Layout

- `app/page.tsx`: the three tabs. The open tab is in the URL hash (`#inspection`, `#queue`, `#evidence`).
- `components/`: one file per tab, plus small shared pieces in `ui.tsx`.
- `lib/api.ts`: types matching `api/schemas.py`, and a `useApi` fetch hook.

## Model status

Until `measure.vial()`, `mixing.predict()` and `advise` are implemented, inspection and queue results come back with `source: "pending"` and empty model fields. The UI says so instead of showing demo values. Nothing here changes when they land: the API calls them through the signatures in `contracts.py`.
