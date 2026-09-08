#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rood_dasfaa2019.data import prepare_chengdu_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and prepare the Chengdu order dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--raw-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.raw_data_dir is not None:
        config["data"]["raw_data_dir"] = str(args.raw_data_dir)
    if args.output_dir is not None:
        config["data"]["output_dir"] = str(args.output_dir)

    outputs = prepare_chengdu_dataset(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
