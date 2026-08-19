#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def run(script: str, *args: str) -> None:
    command = [sys.executable, f"scripts/{script}", *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Student B's B0-B6 Chengdu pipeline.")
    parser.add_argument("--skip-data", action="store_true", help="Reuse prepared B1 files.")
    parser.add_argument("--skip-prediction", action="store_true", help="Reuse trained B2 predictions.")
    args = parser.parse_args()
    if not args.skip_data:
        run("prepare_chengdu_data.py")
    if not args.skip_prediction:
        run("train_chengdu_predictions.py")
        run("train_checkpoint_predictions.py")
    run("export_chengdu_supply.py")
    run("run_prediction_diagnostics.py")
    run("run_checkpoint_dispatch.py")
    run("run_distribution_shift.py")
    run("build_ablation_results.py")
    run("run_runtime_scalability.py")
    run("plot_paper_figures.py")
    run("plot_checkpoint_experiment.py")


if __name__ == "__main__":
    main()
