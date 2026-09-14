from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.experiment import run, save_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the monotonic BNN simulation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--sample-sizes", nargs="+")
    parser.add_argument("--noise-levels", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    filters = {
        key: set(value) for key, value in {
            "models": args.models, "scenarios": args.scenarios,
            "sample_sizes": args.sample_sizes, "noise_levels": args.noise_levels,
        }.items() if value
    }
    raw, summary = run(config, filters)
    save_results(config, raw, summary)
    print(f"Saved {len(raw)} regional result rows to {config['output_dir']}")


if __name__ == "__main__":
    main()

