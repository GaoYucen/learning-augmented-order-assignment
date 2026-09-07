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
    parser = argparse.ArgumentParser(description="Prepare Chengdu data and generate the two final figures.")
    parser.add_argument("--skip-data", action="store_true", help="Reuse prepared B1 files.")
    args = parser.parse_args()
    if not args.skip_data:
        run("prepare_chengdu_data.py")
    run("run_final_prediction_experiments.py")


if __name__ == "__main__":
    main()
