# Rockwell Automation + NVIDIA Innovation Lab · Fall 2026

This is your team's repository for the four-week physical AI hackathon run by Rockwell Automation, NVIDIA, the MSOE AI Club and UWM's Connected Systems Institute. Build everything here. Sponsors can follow your work as it progresses, and your final submission links straight to this repository.

**Full event guide** (datasets, compute, rules, judging, prizes): <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
**Your team workspace** (team, repository access, submissions): <https://dashboard.all-ai-network.org/j/L36HEH>
**Questions:** [join the hackathon Slack](https://join.slack.com/t/rockwellnvidi-wq82211/shared_invite/zt-4bp05nsn9-UW3afFtM9wjzWtZiQoJBHg) and ask in the channel that matches your question. The event guide's "Getting help" section says who to ping for what.

## Key dates

| When | What |
|---|---|
| Thu Oct 8, 4:00–6:00 PM | Kickoff at Rockwell Automation HQ, 1201 S 2nd St, Milwaukee |
| **Wed Oct 14, 11:59 PM CT** | **Team plan due:** [submit it here](https://dashboard.all-ai-network.org/s/V28NHK) |
| **Sun Nov 1, 11:59 PM CT** | **Final submission due:** [submit it here](https://dashboard.all-ai-network.org/s/V28NHK) |
| Thu Nov 5, 5:00–8:00 PM | Finals at MSOE Diercks Hall: live presentations and prizes |

## First week

- [ ] Everyone on the team has an ALL AI Network account and has joined this team in the [workspace](https://dashboard.all-ai-network.org/j/L36HEH).
- [ ] Everyone has connected GitHub and accepted the repository invitation (check your GitHub email, or open the workspace and press the check button).
- [ ] Pick your dataset. We recommend one strong solution around a single dataset.
- [ ] Download the data and make sure everyone who needs it can open it.
- [ ] Read the compute guide and decide what you'll need.
- [ ] Join Slack, open `#hackathon-announcements`, and find your dataset's channel.
- [ ] Submit the [team plan](https://dashboard.all-ai-network.org/s/V28NHK) by Wednesday, October 14. It's a short resource request, not a graded paper, and any teammate can update it before the deadline.

## The three datasets

1. **[Final Vial Inspection](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection):** find missing caps, crooked caps, damaged vials and unexpected empty positions after the robot fills a fixture.
2. **[Fluid Mixing Station 3](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing):** use synchronized camera views to detect fill and color problems caused by recipes and fluid carryover.
3. **[Grasping and Sorting](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Grasping-Sorting):** use a digital twin and multiple cameras to make a standalone FANUC arm grasp and sort objects more reliably.

## Read before your team plan

- **[Compute: Brev, DGX Spark and build.nvidia.com](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Compute).** Brev credits are spent whenever an instance is running, so stop instances you aren't using.
- **[Collect and test your own data](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection):** visits, extra data, digital twins, and booking a supervised test on the real equipment.
- **[Tools and resources](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Resources):** AI coding tools, vision-language models, NVIDIA Cosmos and more. Start with a simple baseline.

## Final submission (Sun Nov 1)

- A video of about five minutes
- A one-page project summary
- A link to this repository
- References and credits for data, models, code and tools

## Working in this repository

### Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the repo (use the URL from the green "Code" button):
   ```bash
   git clone <this repository's URL from the green "Code" button>
   ```
3. Run `uv sync` inside the repo.

### Data

Download the Fluid Mixing dataset from the [event guide link](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing) above ("The three datasets"). Unzip it into `data/` so that `data/vial_fill/image_index.csv` exists. The folders are `vial_fill/`, `tank_level/` and `cap_seal/`, plus `videos.csv` and `README.md`. We don't use `cap_seal`. `data/` is gitignored. If extraction creates a wrapper folder (Google Drive zips and Windows "Extract All" do this), move `vial_fill/` and `tank_level/` up so that `data/vial_fill/image_index.csv` exists. If `uv run pytest` reports 3 skipped, the data isn't where it should be.

### Run

```bash
uv run pytest
uv run uvicorn api.main:app --reload     # API on :8000 (docs at /docs)
cd web && bun install && bun dev         # operator UI on :3000, in a second terminal
```

### Optimization model

The queue optimizer preserves the first submitted batch, supports repeated recipe IDs as distinct
batches, and uses an exact Held-Karp solver over directed empirical carryover costs. Its objective
balances total changeover risk with the worst individual transition. Run synthetic recovery, real
Run 5 exploratory ranking and the 36-scenario sensitivity analysis with:

```bash
uv run python scripts/evaluate_optimization.py
```

The reported reduction is a relative model score under explicit carryover assumptions, not measured
scrap or waste. See `docs/optimization_model.md` for the model, API, validation results and limitations.

For a demo, `cd web && bun run build` once; then `uv run uvicorn api.main:app` alone serves the UI and API at http://localhost:8000. See `web/README.md`. The Streamlit tools in `dashboard/` still run with `uv run streamlit run dashboard/app.py`.

### Tank color initialization

The operator UI opens on **Tank Initialization**. Select a PNG/JPEG tank top view,
review or adjust the four liquid regions, and save the CIELAB baselines for
`T_100`, `T_200`, `T_300`, and `T_400`. Confirmed snapshots persist in ignored
`outputs/baselines/` and can be downloaded as JSON. This flow works without the
event dataset. T_300 is recorded but remains unavailable for recipes.

See [MODEL_PROPOSAL.md](MODEL_PROPOSAL.md) for the volume-aware optical model
design and the **previous recipe run** state. The notebook below implements clean
optical-density fitting and observed recovery analysis; production deployment
still requires camera evidence and validated model artifacts.

### Vial color and transition modeling

Open [`notebooks/transition_analysis.ipynb`](notebooks/transition_analysis.ipynb)
or run `uv run --group analysis python scripts/analyze_transitions.py` after
`uv sync --locked --group analysis`. The workflow accepts the October 29
`vial_filled/labels/vial_recipes.csv` and `vial_fill_frame_labels.csv` schema,
aligns wall-clock labels using `timestamp - video_s`, and exports measured
recovery curves, censored transitions, independent-session counts and supported
optical-density/decay fits. Camera proof and independent clean-model validation
gate production pass/fail; missing data produces explicit pending/null results.
See [configuration, calibration controls and validation](docs/transition_analysis.md).

The October 29 stills and labels are committed under `data/10_29_26/`; the three
session videos are not. Get `10_29_26_videos.zip` from the team chat and unzip it
at the repo root (`unzip 10_29_26_videos.zip`); it restores
`data/10_29_26/*/videos/*.mp4`.

### Who owns what

| Workstream | Owners | Lives in |
|---|---|---|
| Digital twin and synthetic data | Cade, Vlad | `twin/` (Isaac Sim's own Python, on Brev) |
| Camera and color measurement | Braedon, Kal | `src/changeover_guard/dataset.py`, `measure.py` |
| Mixing model and explanations | Adrian, Niko | `src/changeover_guard/mixing.py`, `advise.py` |
| Dashboard and integration | Mat, Vlad | `web/` (Next.js UI), `api/` (FastAPI), `dashboard/` (Streamlit tools) |
| Business case and testing | William | `eval/` |

`contracts.py` is shared. Changes to it need a heads-up to the whole team.

### Git

- Commit early and push often. Mentors and judges see what's here.
- **Don't commit large datasets or model weights.** Keep them out with `.gitignore` and note in this README where to download them.
- Before the finals, replace this README with your project: the problem, your approach, results, how to run it, and credits.

**Can't push or see this repository?** Open the [team workspace](https://dashboard.all-ai-network.org/j/L36HEH) and press the GitHub check button. If it still doesn't work, ping Brett Storoe in `#hackathon-general` on Slack.
