# Operator UI

Next.js (React, TypeScript, Tailwind) front end for Changeover Guard. All data comes from the FastAPI backend in `../api/`, which wraps the `changeover_guard` pipeline. Owners: Mat, Vlad.

## Develop

Two terminals, from the repo root:

```bash
uv run uvicorn api.main:app --reload     # API on :8000 (endpoints listed at /docs)
cd web && bun install && bun dev         # UI on :3000; /api calls are forwarded to :8000
```

The real dataset must be unzipped into `data/` (see the root README).

## Demo / deploy

```bash
cd web && bun run build                  # static files in web/out
uv run uvicorn api.main:app              # serves the UI and the API at http://localhost:8000
```

One process, one URL, so it can run on the CSI workstation or the DGX Spark.

## Layout

- `app/page.tsx`: the three tabs. The open tab is in the URL hash (`#inspection`, `#queue`, `#evidence`).
- `components/`: one file per tab, plus small shared pieces in `ui.tsx`.
- `lib/api.ts`: types matching `api/schemas.py`, and a `useApi` fetch hook.

## Model status

Inspection and queue results remain `source: "pending"` until validated model,
camera and fill evidence is supplied. The contract adapters now implement color
measurement and checks, but abstain for unmeasured fill and unsupported first-vial
carryover predictions. The standalone operator notebook provides richer
sequence diagnostics; its exploratory results do not automatically enable API
production decisions. See `docs/operator_diagnostic_system.md`.
