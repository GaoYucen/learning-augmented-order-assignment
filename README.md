# ROOD-DASFAA2019

A clean, runnable **reimplementation** of the conference paper:

> *Real-Time Route Planning and Online Order Dispatch for Bus-Booking Platforms* (DASFAA 2019).

The repository follows the modular `src/` organization used in ST-SACA, while replacing its reinforcement-learning components with the conference paper's original problem setting:

- demand prediction by historical time-slot averages;
- capacity-aware route construction for bus lines;
- Random and Greedy online dispatch baselines;
- Fractional Primal-Dual (FPD) simulation;
- Improved Primal-Dual (IPD) online integer dispatch;
- an offline mixed-integer-programming upper bound (`OPT`) through SciPy MILP;
- the five experimental configurations reported in the paper.

## Important provenance note

The original 2019 source code is unavailable. This repository is reconstructed from the accepted manuscript and uses ST-SACA only as an engineering-style reference. It should therefore be described as a **paper-based reimplementation**, not as the exact historical source code.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick start

Run one experiment:

```bash
python scripts/run_experiment.py --experiment 1 --seed 2019
```

Run all five configurations and create CSV/figures:

```bash
python scripts/reproduce_paper.py --seed 2019
```

Run tests:

```bash
pytest -q
```

## Learning-augmented synthetic sanity-check

This script is the first implementation entry for Student A. It creates a controlled experiment for learning-augmented online dispatch:

```text
slot orders -> synthetic prediction -> predictive LP advice
-> online dispatch algorithms -> offline OPT comparison -> CSV/figure outputs
```

Run the bottleneck setting:

```bash
python scripts/synthetic_sanity_check.py --setting bottleneck --slots 5 --prediction-scales 1.0 0.75 1.25 0.5 1.5 --thetas 0.2 0.4 0.6 0.8 --output-dir outputs/synthetic_sanity_bottleneck
```

The script compares `Random`, `Greedy`, `IPD`, `Prediction-only`, and `RP-LAIPD`. `Prediction-only` follows predictive LP advice directly. `RP-LAIPD` mixes the prediction advice with the original IPD idea; larger `theta` means the algorithm trusts prediction more.

Generated files:

- `synthetic_sanity_results.csv`: raw result of every slot/method/prediction setting.
- `synthetic_sanity_summary.csv`: grouped mean/std/95% CI, easier to read and report.
- `alg_over_opt_vs_prediction_error.png`: trend figure of `ALG/OPT` against prediction error.

For Student A's A1/A2 tasks, this is a synthetic sanity-check, not the final real-data predictor experiment. Its purpose is to verify that the experiment pipeline works and that learning-augmented methods behave reasonably when prediction quality changes.

## Repository layout

```text
configs/                  Five paper experiment settings
data/stations/             Synthetic 30-station airport benchmark
src/rood_dasfaa2019/
  algorithms/              Random, Greedy, FPD, IPD, offline OPT
  routing/                 Demand prediction and capacity-aware routing
  simulation/              Entities, instance generation, metrics
  utils/                   Configuration and reproducibility helpers
scripts/                   Experiment entry points
outputs/                   Generated CSV files and figures
```

## Experimental assumptions

The manuscript specifies 30 hot stations, randomly generated buses/orders, random priorities in `(0, 1]`, ten-minute bus waiting windows, and four resource constraints. It does not publish the precise station coordinates, random generator, CVRP solver settings, or all numerical constants. This implementation makes those assumptions explicit in YAML files and uses deterministic synthetic coordinates and seeded simulation.

## Citation

```bibtex
@inproceedings{zhou2019realtime,
  title={Real-Time Route Planning and Online Order Dispatch for Bus-Booking Platforms},
  author={Zhou, Hao and Gao, Yucen and Gao, Xiaofeng and Chen, Guihai},
  booktitle={Database Systems for Advanced Applications},
  year={2019}
}
```
