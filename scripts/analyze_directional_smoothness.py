"""Summarize directional smoothness evidence from eta_case_level.csv.

The input is produced by the theorem-aligned E8-B reconstruction on the server.
This script is intentionally analysis-only: it does not tune VDR-LA or alter
predictions/assignments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pair_report(df, col, a, b, keys, name):
    A = df[df[col] == a][keys + ["calibrated"]].rename(columns={"calibrated": "alg_a"})
    B = df[df[col] == b][keys + ["calibrated"]].rename(columns={"calibrated": "alg_b"})
    M = A.merge(B, on=keys)
    d = M.alg_a - M.alg_b
    return {
        "comparison": name,
        "n": len(M),
        "mean_alg_a": M.alg_a.mean(),
        "mean_alg_b": M.alg_b.mean(),
        "mean_a_minus_b": d.mean(),
        "frac_a_better": (d > 1e-12).mean(),
        "frac_equal": (abs(d) <= 1e-12).mean(),
        "frac_a_worse": (d < -1e-12).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="eta_case_level.csv")
    ap.add_argument("--out", type=Path, default=Path("outputs/smoothness_summary"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    D = pd.read_csv(args.input)
    D["eta_over"] = (D.eta_vdr + D.eta_signed) / 2
    D["eta_under"] = (D.eta_vdr - D.eta_signed) / 2
    keys = ["load", "true_pattern", "hetero", "seed"]

    C = D[(D.pred_mode == "uniform") & (D.cdf_mode == "correct") & (D.value_mode == "exact")].copy()
    scale = C.groupby("scale").agg(
        eta=("eta_vdr", "mean"), eta_under=("eta_under", "mean"), eta_over=("eta_over", "mean"),
        alg=("calibrated", "mean"), frozen=("frozen", "mean"), greedy=("greedy", "mean"),
        n=("calibrated", "size"),
    ).reset_index()
    scale.to_csv(args.out / "controlled_count_scale.csv", index=False)

    pairs = []
    for s in [0.5, 1.5, 2.0]:
        pairs.append(pair_report(C, "scale", 1.0, s, keys, f"perfect(1.0) vs scale={s}"))
        pairs.append(pair_report(C[C.load >= 1.2], "scale", 1.0, s, keys, f"congested perfect(1.0) vs scale={s}"))
    pd.DataFrame(pairs).to_csv(args.out / "controlled_pairwise_perfect.csv", index=False)

    print(scale.round(5).to_string(index=False))
    print("\nPaired comparisons")
    print(pd.DataFrame(pairs).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
