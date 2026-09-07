#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rood_dasfaa2019.experiments import ChengduExperimentContext, run_chengdu_slot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learning-augmented dispatch on prepared Chengdu slots.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--model", default="HGB")
    parser.add_argument("--split", choices=["validation", "test", "shifted_test"], default="test")
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--corruption", choices=["none", "scale", "permutation", "temporal_shift", "scarce_resource"], default="none")
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override experiment seeds.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    slot_ids = (
        context.demand.loc[context.demand["split"] == args.split, ["slot_id", "service_date", "slot_in_day"]]
        .drop_duplicates().sort_values(["service_date", "slot_in_day"])["slot_id"].head(args.slots).tolist()
    )
    rows = []
    for seed in (args.seeds or config["experiment"]["seeds"]):
        for slot_id in slot_ids:
            rows.extend(
                run_chengdu_slot(
                    context,
                    slot_id,
                    args.model,
                    corruption=args.corruption,
                    level=args.level,
                    seed=int(seed),
                    include_baselines=args.include_baselines,
                )
            )
    output = args.output or Path(config["data"]["output_dir"]) / "raw" / "chengdu_experiment.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).groupby(["method", "theta"], dropna=False)["alg_over_opt"].agg(["mean", "std", "count"]))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
