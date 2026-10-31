"""Run the same real-data workflow as notebooks/transition_analysis.ipynb."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from changeover_guard.model_workflow import default_config, run_workflow, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config; relative paths resolve from the repository root",
    )
    parser.add_argument(
        "--write-config", type=Path, help="write a starter config and exit"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.write_config:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.write_config, default_config(root))
        return 0
    config = (
        json.loads(args.config.read_text(encoding="utf-8"))
        if args.config
        else default_config(root)
    )
    plots = not args.no_plots
    if plots and importlib.util.find_spec("matplotlib") is None:
        print("matplotlib not installed; skipping plots (uv sync --group analysis)", file=sys.stderr)
        plots = False
    result = run_workflow(config, root, plots=plots)
    print(json.dumps(result["report"], indent=2))
    return {"measured": 0, "missing_data": 3, "invalid_inputs": 2}.get(
        result["report"]["status"], 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
