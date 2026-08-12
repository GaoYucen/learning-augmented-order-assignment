#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path

from rood_dasfaa2019.algorithms import (
    ipd_dispatch,
    offline_opt,
    paper_greedy_dispatch,
    paper_random_dispatch,
)
from rood_dasfaa2019.algorithms.common import candidate_stats
from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.utils.config import load_experiment


def main():
    base = load_experiment(1)
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'scan_ipd_configs.csv'
    combos = []
    for overlap in (6, 8, 10, 12):
        for wave_std in (15, 25, 35):
            for route_min, route_max in ((5, 8), (7, 10), (9, 12)):
                cfg = deepcopy(base)
                cfg['route_overlap_factor'] = overlap
                cfg['bus_wave_std_minutes'] = wave_std
                cfg['route_min_stations'] = route_min
                cfg['route_max_stations'] = route_max
                combos.append(cfg)

    fieldnames = [
        'idx', 'overlap', 'wave_std', 'route_min', 'route_max',
        'avg_candidates', 'zero_candidate_rate', 'random_priority',
        'greedy_priority', 'ipd_priority', 'opt_priority',
        'ipd_vs_random_pct', 'ipd_vs_greedy_pct', 'ipd_ratio_to_opt',
    ]
    rows = []
    for idx, cfg in enumerate(combos, 1):
        st, orders, buses = generate_instance(cfg, 2020)
        diag = candidate_stats(orders, buses)
        pr = paper_random_dispatch(orders, buses, len(st), cfg, 2020)
        pg = paper_greedy_dispatch(orders, buses, len(st), cfg)
        ipd = ipd_dispatch(orders, buses, len(st), cfg)
        opt = offline_opt(orders, buses, len(st), cfg)
        row = {
            'idx': idx,
            'overlap': cfg['route_overlap_factor'],
            'wave_std': cfg['bus_wave_std_minutes'],
            'route_min': cfg['route_min_stations'],
            'route_max': cfg['route_max_stations'],
            'avg_candidates': round(diag.average_candidates, 3),
            'zero_candidate_rate': round(diag.zero_candidate_rate, 3),
            'random_priority': round(pr.objective, 3),
            'greedy_priority': round(pg.objective, 3),
            'ipd_priority': round(ipd.objective, 3),
            'opt_priority': round(opt.objective, 3),
            'ipd_vs_random_pct': round((ipd.objective / max(pr.objective, 1e-9) - 1.0) * 100.0, 2),
            'ipd_vs_greedy_pct': round((ipd.objective / max(pg.objective, 1e-9) - 1.0) * 100.0, 2),
            'ipd_ratio_to_opt': round(ipd.objective / max(opt.objective, 1e-9), 3),
        }
        rows.append(row)

    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(','.join(fieldnames))
    for row in rows:
        print(','.join(str(row[k]) for k in fieldnames))
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()