#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rood_dasfaa2019.learning.metrics import summarize_results
from rood_dasfaa2019.learning.runner import run_learning_augmented_instance, run_learning_augmented_slot
from rood_dasfaa2019.learning.synthetic import apply_bottleneck_config, generate_bottleneck_instance
from rood_dasfaa2019.utils.config import load_experiment


def plot_sanity(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    plot_df = df.copy()
    plot_df["curve"] = plot_df.apply(
        lambda row: f"RP-LAIPD theta={row['theta']} (robust)" if row["method"] == "RP-LAIPD" else row["method"],
        axis=1,
    )
    grouped = plot_df.groupby(["curve", "prediction_error"], as_index=False)["alg_over_opt"].mean()
    for curve, sub in grouped.groupby("curve"):
        sub = sub.sort_values("prediction_error")
        ax.plot(sub["prediction_error"], sub["alg_over_opt"], marker="o", linewidth=1.8, label=curve)
    ax.set_title("Synthetic sanity-check: ALG/OPT vs prediction error")
    ax.set_xlabel("Prediction error |scale - 1|")
    ax.set_ylabel("ALG / OPT")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "synthetic_sanity")
    parser.add_argument("--setting", choices=["paper", "bottleneck"], default="paper")
    parser.add_argument(
        "--prediction-scales",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 1.25, 0.5, 1.5],
    )
    parser.add_argument(
        "--corruption",
        choices=["scale", "permutation", "adversarial_concentration"],
        default="scale",
    )
    parser.add_argument(
        "--corruption-strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="Robust IPD resource fractions. theta=1 ignores prediction; theta=0 follows advice.",
    )
    args = parser.parse_args()

    cfg = load_experiment(args.experiment)
    if args.setting == "bottleneck":
        cfg = apply_bottleneck_config(cfg)
    rows = []
    for slot in range(args.slots):
        seed = args.seed + args.experiment * 100 + slot
        if args.setting == "bottleneck":
            stations, orders, buses = generate_bottleneck_instance(cfg, seed)
            rows.extend(
                run_learning_augmented_instance(
                    stations,
                    orders,
                    buses,
                    cfg,
                    seed,
                    args.prediction_scales,
                    args.corruption_strengths,
                    args.corruption,
                    args.thetas,
                    slot,
                )
            )
        else:
            rows.extend(
                run_learning_augmented_slot(
                    cfg=cfg,
                    seed=seed,
                    prediction_scales=args.prediction_scales,
                    corruption_strengths=args.corruption_strengths,
                    corruption=args.corruption,
                    thetas=args.thetas,
                    slot_id=slot,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary = summarize_results(df)
    csv_path = args.output_dir / "synthetic_sanity_results.csv"
    summary_path = args.output_dir / "synthetic_sanity_summary.csv"
    fig_path = args.output_dir / "alg_over_opt_vs_prediction_error.png"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_sanity(df, fig_path)
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
