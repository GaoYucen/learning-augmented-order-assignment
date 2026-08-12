#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rood_dasfaa2019.algorithms import (
    greedy_dispatch,
    ipd_dispatch,
    ipd_paper_dispatch,
    offline_opt,
    paper_greedy_dispatch,
    paper_random_dispatch,
    random_dispatch,
)
from rood_dasfaa2019.algorithms.common import candidate_stats
from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.utils.config import load_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=2019)
    p.add_argument('--skip-opt', action='store_true')
    a = p.parse_args()
    rows = []

    for e in range(1, 6):
        cfg = load_experiment(e)
        st, orders, buses = generate_instance(cfg, a.seed + e)
        diag = candidate_stats(orders, buses)
        methods = {
            'PaperRandom': paper_random_dispatch(orders, buses, len(st), cfg, a.seed + e),
            'PaperGreedy': paper_greedy_dispatch(orders, buses, len(st), cfg),
            'RandomFF': random_dispatch(orders, buses, len(st), cfg, a.seed + e),
            'GreedyFF': greedy_dispatch(orders, buses, len(st), cfg),
            'IPD-paper': ipd_paper_dispatch(orders, buses, len(st), cfg),
            'IPD-feasible': ipd_dispatch(orders, buses, len(st), cfg),
        }
        if not a.skip_opt:
            methods['OPT'] = offline_opt(orders, buses, len(st), cfg)

        opt = methods.get('OPT')
        for name, r in methods.items():
            dispatch = r.dispatch if hasattr(r, 'dispatch') else r
            row = {
                'experiment': e,
                'method': name,
                'orders': len(dispatch.accepted),
                'passengers': dispatch.passengers,
                'priority': dispatch.objective,
                'travel_time': dispatch.travel_time,
                'avg_travel_time': dispatch.travel_time / max(dispatch.passengers, 1),
                'avg_candidate_buses': diag.average_candidates,
                'zero_candidate_rate': diag.zero_candidate_rate,
            }
            if hasattr(r, 'diagnostics') and r.diagnostics is not None:
                row['rejected_no_candidate'] = r.diagnostics.rejected_no_candidate
                row['rejected_dual_price'] = r.diagnostics.rejected_dual_price
            if opt is not None:
                opt_dispatch = opt.dispatch if hasattr(opt, 'dispatch') else opt
                row['priority_ratio_to_opt'] = dispatch.objective / max(opt_dispatch.objective, 1e-9)
                row['passenger_ratio_to_opt'] = dispatch.passengers / max(opt_dispatch.passengers, 1e-9)
                row['orders_ratio_to_opt'] = len(dispatch.accepted) / max(len(opt_dispatch.accepted), 1)
            rows.append(row)

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'results.csv', index=False)

    for metric in ['orders', 'passengers', 'priority', 'avg_travel_time']:
        pivot = df.pivot(index='experiment', columns='method', values=metric)
        ax = pivot.plot(kind='bar')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.figure.tight_layout()
        ax.figure.savefig(out / f'{metric}.png', dpi=180)
        plt.close(ax.figure)

    print(df.to_string(index=False))
    print(f'Wrote {out / "results.csv"}')


if __name__ == '__main__':
    main()
