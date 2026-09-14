from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.prior_checks import run_prior_checks


def parse_args():
    parser = argparse.ArgumentParser(description="Run prior predictive checks for the BNN.")
    parser.add_argument("--config", type=Path, default=Path("configs/prior_check.yaml"))
    parser.add_argument("--c-values", type=float, nargs="+", help="Override c values from the config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.c_values is not None:
        config["c_values"] = args.c_values
    results = run_prior_checks(config)
    print("\nPrior predictive check results")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

