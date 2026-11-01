"""Run the same October 29 diagnostic workflow used by the notebook."""

import argparse
import json
from pathlib import Path

from changeover_guard.diagnostic_workflow import run_operator_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tank-image", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_operator_workflow(
        args.root, tank_image=args.tank_image, plots=not args.no_plots
    )
    print(json.dumps(result["summary"], indent=2))
    print("Outputs:", result["output_dir"])
    return 0 if result["summary"]["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
