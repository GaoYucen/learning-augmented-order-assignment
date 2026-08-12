#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy
from statistics import mean
import csv
from pathlib import Path

from rood_dasfaa2019.algorithms import (
    ipd_dispatch,
    ipd_paper_dispatch,
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
    out_csv = out_dir / 'evaluate_ipd_candidates.csv'
    candidate_params = [
        {'route_overlap_factor': 8, 'bus_wave_std_minutes': 15, 'route_min_stations': 9, 'route_max_stations': 12},
        {'route_overlap_factor': 10, 'bus_wave_std_minutes': 15, 'route_min_stations': 10, 'route_max_stations': 14},
        {'route_overlap_factor': 12, 'bus_wave_std_minutes': 15, 'route_min_stations': 10, 'route_max_stations': 14},
        {'route_overlap_factor': 14, 'bus_wave_std_minutes': 15, 'route_min_stations': 10, 'route_max_stations': 15},
        {'route_overlap_factor': 12, 'bus_wave_std_minutes': 10, 'route_min_stations': 10, 'route_max_stations': 14},
        {'route_overlap_factor': 14, 'bus_wave_std_minutes': 10, 'route_min_stations': 10, 'route_max_stations': 15},
        {'route_overlap_factor': 12, 'bus_wave_std_minutes': 8, 'route_min_stations': 12, 'route_max_stations': 16},
    ]
    seeds = [2020, 2021, 2022, 2023, 2024]

    fieldnames = [
        'config', 'avg_candidates', 'zero_rate', 'random_priority',
        'greedy_priority', 'ipd_paper_priority', 'ipd_feasible_priority',
        'opt_priority', 'ipdf_vs_random_pct', 'ipdf_vs_greedy_pct',
        'ipdf_ratio_to_opt',
    ]
    rows_out = []

    for params in candidate_params:
        rows = []
        for seed in seeds:
            cfg = deepcopy(base)
            cfg.update(params)
            stations, orders, buses = generate_instance(cfg, seed)
            diag = candidate_stats(orders, buses)
            pr = paper_random_dispatch(orders, buses, len(stations), cfg, seed)
            pg = paper_greedy_dispatch(orders, buses, len(stations), cfg)
            ipd_p = ipd_paper_dispatch(orders, buses, len(stations), cfg)
            ipd_f = ipd_dispatch(orders, buses, len(stations), cfg)
            opt = offline_opt(orders, buses, len(stations), cfg)
            rows.append((
                diag.average_candidates,
                diag.zero_candidate_rate,
                pr.objective,
                pg.objective,
                ipd_p.objective,
                ipd_f.objective,
                opt.objective,
            ))

        avg_candidates, zero_rate, pr_obj, pg_obj, ipdp_obj, ipdf_obj, opt_obj = [mean(col) for col in zip(*rows)]
        ipdf_vs_random = (ipdf_obj / max(pr_obj, 1e-9) - 1.0) * 100.0
        ipdf_vs_greedy = (ipdf_obj / max(pg_obj, 1e-9) - 1.0) * 100.0
        ipdf_ratio_to_opt = ipdf_obj / max(opt_obj, 1e-9)
        rows_out.append({
            'config': str(params),
            'avg_candidates': round(avg_candidates, 3),
            'zero_rate': round(zero_rate, 3),
            'random_priority': round(pr_obj, 3),
            'greedy_priority': round(pg_obj, 3),
            'ipd_paper_priority': round(ipdp_obj, 3),
            'ipd_feasible_priority': round(ipdf_obj, 3),
            'opt_priority': round(opt_obj, 3),
            'ipdf_vs_random_pct': round(ipdf_vs_random, 2),
            'ipdf_vs_greedy_pct': round(ipdf_vs_greedy, 2),
            'ipdf_ratio_to_opt': round(ipdf_ratio_to_opt, 3),
        })

    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(','.join(fieldnames))
    for row in rows_out:
        print(','.join(str(row[k]) for k in fieldnames))
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()