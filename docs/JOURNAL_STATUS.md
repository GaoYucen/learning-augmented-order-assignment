# Journal Extension Status — 2026-09-09

## Current algorithm

The frozen experimental implementation is **VDR-LA with One-Sided Online Release**:

1. value-aware predictive advice;
2. Value-Dominance Reservation (reserve only for predicted future requests with strictly higher value);
3. one-sided online release for statistically supported overprediction / arrival shift;
4. strict feasibility gate, with a work-conserving Greedy-like branch when effective demand is no longer congested.

The formal implementation is `src/rood_dasfaa2019/algorithms/vdr.py`. The corrected future-reservation rule applies quota calibration exactly once. Before consolidation into `main`, the formal implementation passed 5/5 algorithm/VDR tests and matched the official corrected E8-B implementation in 10/10 representative cases for both objective and accepted set.

## Official corrected E8-B

14,400 paired cases, no further calibration tuning:

- Frozen VDR mean ALG/OPT: **0.9331**
- Calibrated VDR mean ALG/OPT: **0.9383**
- Perfect matched: **0.9703 -> 0.9703**
- CDF shift: **0.9212 -> 0.9306**
- Severe shifted count error: **0.9063 -> 0.9186**
- Worst regime-mean gap vs Greedy: **-0.1672 -> -0.1214**
- Strict feasibility violations: **0**

Paper-valid numbers are in `docs/evidence/E8B_REPORT.txt`; large raw result grids remain under the server workspace and are intentionally not tracked.

## Baseline interpretation

In the controlled E8 bottleneck benchmark, low loads (`0.8`, `1.0`) are easy: Random feasible scan and Greedy can both reach OPT. At higher loads the gap becomes meaningful. For example, at load 2.0, Random/Greedy are about 0.7125 ALG/OPT while VDR-LA is about 0.8919. Paper-IPD can exceed 1.0 ALG/OPT because its legacy theorem/implementation permits resource augmentation; it must not be interpreted as a strict-feasible OPT ratio.

## Theory direction

The journal theory is organized into two parts.

### 1. IPD competitive theory

Retain the DASFAA 2019 bicriteria competitive guarantee as a legacy result, and separately analyze the strict-feasible IPD used by the journal version. The strict-feasible result should not inherit the conference `(1-epsilon)` guarantee without proof because the conference result is tied to resource augmentation / constraint violation. A strict-feasible IPD guarantee is also the natural robustness backbone for the learning-augmented algorithm.

### 2. Learning-augmented consistency, robustness, smoothness

The target is a standard three-part theory:

- **Consistency:** accurate/perfect advice approaches the predictive optimum / OPT under explicit assumptions.
- **Robustness:** arbitrary prediction error is lower-bounded by a strict-feasible IPD safety backbone.
- **Smoothness:** performance degrades with decision-relevant prediction error, but the error must be directional/asymmetric.

The latest experiment rules out a naive symmetric absolute-error smoothness statement. Define future high-value demand errors separately:

- `eta-`: missed / underpredicted future high-value demand;
- `eta+`: excess / overpredicted future high-value demand.

A target form is therefore an asymmetric bound such as

`ALG >= max{rho_robust, 1-delta-L_- eta- - L_+ eta+} OPT`,

with `L_- > L_+` expected from the mechanism and experiments. This is a target theorem, not yet a proved result.

## Directional smoothness evidence

On the controlled count-prediction family (CDF and value kept exact):

| count scale | theory-aligned eta | mean ALG/OPT |
|---:|---:|---:|
| 0.5 | 0.2862 | 0.9072 |
| 1.0 | 0.0000 | 0.9703 |
| 1.5 | 0.2862 | 0.9679 |
| 2.0 | 0.5723 | 0.9524 |

At load >= 1.2, perfect prediction beats 0.5x underprediction by about **10.48 percentage points**, and is not worse in **99.63%** of paired cases. Moderate 1.5x overprediction can be as good as or slightly better than perfect prediction, so under- and overprediction are not symmetric. Correct vs shifted arrival CDF is **0.9703 vs 0.9302**; exact vs noisy value is **0.9703 vs 0.9660**.

Compact CSVs and figures are in `docs/evidence/smoothness/`.

## Next work

1. derive a strict-feasible IPD competitive guarantee and identify the correct lower-bound/tightness relationship;
2. formalize a VDR-IPD safety backbone for global robustness;
3. prove consistency and asymmetric smoothness with `eta-` / `eta+` or a piecewise overprediction tolerance;
4. run a harder synthetic benchmark with non-equivalent buses/routes/candidate sets and focus on load >= 1.2;
5. add one real/public mobility trace;
6. build final paper tables, confidence intervals, ablations and failure-regime figures.
