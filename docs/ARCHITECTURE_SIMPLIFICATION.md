# LaGraph Architecture Simplification Plan

Date: 2026-05-24
Branch: `codex/5070-env-migration`

## Goal

Reduce module stacking while keeping the experimental path explainable and reproducible.

The final runtime profile remains `full`, but its meaning has changed after the
recent simplification: it is now the compact final architecture, not the old
all-module version.

## Current Final Architecture

The retained modules are:

- decomposition backbone
- channel graph refinement
- temporal graph refinement
- temporal encoder with dynamic scale selection
- dual-path VQ bottleneck
- reconstruction anomaly scoring

The removed modules are:

- BoundaryDetector
- multi-scale/VQ anomaly scorer from the main path
- dead auxiliary switches from the single-run entry point:
  `use_prediction_head`, `lambda_pred`, `use_contrastive`, `use_freq_loss`,
  and `use_prototype`

The auxiliary switches above were removed because the current LaGraph
implementation does not consume them. A 15-epoch MSL/SWaT prediction-head check
produced bit-identical metrics to the main run, confirming that these switches
should not be treated as valid modules or ablations.

## Runtime Profiles

| Profile | Channel graph | Temporal graph | VQ bypass | Boundary detector | Multi-scale scorer | Intended use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | on | on | on | removed | off | Main compact architecture |
| `dynamic-temporal` | on | dynamic | on | removed | off | Candidate architecture with content-adaptive temporal graph |
| `dynamic-temporal-gated` | on | dynamic + residual gate | on | removed | off | Current main candidate for dual-graph narrative |
| `dynamic-temporal-gated-vqscore` | on | dynamic + residual gate | on | removed | off | Rejected direct VQ-score fusion profile |
| `parallel-dual` | on | fixed parallel branch | on | removed | off | Rejected dual-graph fusion profile |
| `parallel-dual-time` | on | fixed time-gated parallel branch | on | removed | off | Rejected dual-graph fusion profile |
| `residual-dual` | on | fixed residual parallel correction | on | removed | off | Rejected conservative dual-graph fusion profile |
| `residual-dual-time` | on | fixed time-gated residual correction | on | removed | off | Rejected conservative dual-graph fusion profile |
| `graph-shift` | on | on | on | removed | off | Rejected normal-graph deviation scoring profile |
| `graph-shift-lite` | on | on | on | removed | off | Rejected low-weight graph-shift profile |
| `graph-shift-strong` | on | on | on | removed | off | Rejected high-weight graph-shift profile |
| `causal-lag` | on | on | on | removed | off | Lagged mechanism branch, score disabled |
| `causal-lag-score` | on | on | on | removed | off | Rejected lagged mechanism score profile |
| `causal-lag-score-strong` | on | on | on | removed | off | Stronger lagged mechanism score profile |
| `dynamic-temporal-regularized` | on | dynamic + residual gate + temporal graph regularization | on | removed | off | Rejected graph smoothness/locality regularization profile |
| `dynamic-temporal-robust` | on | dynamic + residual gate | on | removed | off | Rejected 5% trimmed reconstruction loss profile |
| `dynamic-temporal-robust-lite` | on | dynamic + residual gate | on | removed | off | Rejected 2% trimmed reconstruction loss profile |
| `dynamic-temporal-channelnorm` | on | dynamic + residual gate | on | removed | off | Rejected robust channel-normalized scoring profile |
| `dynamic-temporal-channelnorm-scale` | on | dynamic + residual gate | on | removed | off | Rejected scale-only channel-normalized scoring profile |
| `dynamic-temporal-aff` | on | dynamic + time/channel gate | on | removed | off | Rejected affiliation-oriented post-processing profile |
| `dynamic-temporal-aff-lite` | on | dynamic + time/channel gate | on | removed | off | Rejected conservative affiliation post-processing profile |
| `dynamic-temporal-aff-peak` | on | dynamic + residual gate | on | removed | off | Rejected local max-score expansion profile |
| `dynamic-temporal-aff-wide` | on | dynamic + residual gate | on | removed | off | Rejected wider channel-score aggregation profile |
| `with-scorer` | on | on | on | removed | on | Old multi-scale/VQ scoring path for ablation |
| `no-boundary` | on | on | on | removed | on | Compatibility alias for older commands |
| `no-vq` | on | on | off | removed | on | VQ ablation |
| `core` | on | on | off | removed | on | Legacy simplified profile, equivalent to no-vq for current path |
| `channel-only` | on | off | on | removed | off | Channel graph contribution ablation |
| `temporal-only` | off | on | on | removed | off | Temporal graph contribution ablation |
| `reconstruction` | off | off | off | removed | off | Lower-bound reconstruction baseline |

## Default VQ Setting

The unified VQ default is:

```text
vq_cooldown_epochs = 10
lambda_vq = 0.1
vq_score_weight = 0.3
```

This setting was selected because it gave the best balanced behavior across MSL
and SWaT among the tested non-dataset-specific configurations. Detailed tuning
results are recorded in `docs/VQ_TUNING_LOG.md`.

## Experimental Recommendation

Use `full` as the stable baseline in future experiments.

Use `dynamic-temporal-gated` as the strongest adaptive-temporal candidate, not
as the default replacement for `full` yet. It replaces the fixed-position
temporal graph with a content-adaptive temporal adjacency, adds a learnable
residual gate to avoid overwriting the stronger channel graph signal, and trains
the dynamic temporal graph at the base learning rate. Use `dynamic-temporal`
only as the ungated candidate ablation. Use `with-scorer` to report the old
multi-scale/VQ scoring path as an ablation,
because the scorer was not consistently better than direct reconstruction
scoring. Use `no-vq` as the primary VQ ablation, `channel-only` and
`temporal-only` to justify the dual-graph design, and `reconstruction` as the
lower-bound baseline. Keep `core` only for backward compatibility with earlier
commands.

Do not present dataset-specific VQ settings as the main method. If a dataset
benefits from a different VQ score weight, report it only in parameter
sensitivity analysis.

## Three-Seed Stability Check

After promoting direct reconstruction scoring to the main `full` profile, a
three-seed check was run on MSL and SWaT with 15 epochs and the unified VQ
defaults. Reported values are the best values over the standard anomaly-ratio
grid for each metric.

| Dataset | Metric | Seed values | Mean | Std | Best ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| MSL | raw F1 | 0.1109 / 0.1095 / 0.1029 | 0.1078 | 0.0043 | 15% |
| MSL | adjusted F1 | 0.8576 / 0.8529 / 0.8579 | 0.8561 | 0.0028 | 1% |
| MSL | affiliation F1 | 0.7006 / 0.6933 / 0.7001 | 0.6980 | 0.0041 | 2% |
| SWaT | raw F1 | 0.3169 / 0.3162 / 0.3082 | 0.3138 | 0.0048 | 5% |
| SWaT | adjusted F1 | 0.9361 / 0.9261 / 0.9266 | 0.9296 | 0.0056 | 2% |
| SWaT | affiliation F1 | 0.8458 / 0.8474 / 0.8391 | 0.8441 | 0.0044 | 5% |

Conclusion: the final compact `full` profile is stable across seeds. The main
remaining risk is not seed variance; it is whether the raw point-wise F1 is
competitive enough against baselines. The paper should emphasize event-adjusted
and affiliation metrics, while still reporting raw F1 transparently.

## Single-Graph Ablation Check

A seed-2021 single-graph ablation was run on MSL and SWaT to test whether the
dual-graph design is necessary. Reported values are the best values over the
standard anomaly-ratio grid for each metric.

| Profile | Dataset | Raw F1 | Adjusted F1 | Affiliation F1 | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `full` | MSL | 0.1109 | 0.8576 | 0.7006 | Dual graph + VQ |
| `channel-only` | MSL | 0.1162 | 0.8584 | 0.6936 | Better raw/adjusted, lower affiliation |
| `temporal-only` | MSL | 0.1106 | 0.8580 | 0.6996 | Nearly identical to `full` |
| `full` | SWaT | 0.3169 | 0.9361 | 0.8458 | Dual graph + VQ |
| `channel-only` | SWaT | 0.3510 | 0.9216 | 0.8594 | Better raw/affiliation, lower adjusted |
| `temporal-only` | SWaT | 0.3144 | 0.9360 | 0.8439 | Nearly identical to `full` |

Interpretation: the current evidence does not justify a strong claim that the
dual-graph combination is consistently superior. The temporal graph appears to
recover almost all of the `full` behavior, while the channel-only profile may be
a stronger compact candidate when raw F1 and affiliation F1 are prioritized.
Before promoting `channel-only`, it needs the same multi-seed stability check as
`full`.

## Dynamic Temporal Graph Check

A seed-2021 dynamic temporal graph check was run on MSL and SWaT after the
single-graph result showed that the original temporal graph did not justify a
strong dual-graph claim.

| Profile | Dataset | Raw F1 | Adjusted F1 | Affiliation F1 | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `full` | MSL | 0.1109 | 0.8576 | 0.7006 | Fixed-position temporal graph |
| `dynamic-temporal` | MSL | 0.1117 | 0.8574 | 0.6983 | Ungated dynamic temporal graph |
| `dynamic-temporal-gated` | MSL | 0.1139 | 0.8577 | 0.6940 | Current-code rerun; improves raw but lowers affiliation |
| `channel-only` | MSL | 0.1162 | 0.8584 | 0.6936 | Strong raw F1, weaker affiliation |
| `full` | SWaT | 0.3169 | 0.9361 | 0.8458 | Fixed-position temporal graph |
| `dynamic-temporal` | SWaT | 0.3259 | 0.9267 | 0.8492 | Improves raw/affiliation, lower adjusted |
| `dynamic-temporal-gated` | SWaT | 0.3571 | 0.9196 | 0.8614 | Better than `full` on raw/affiliation |
| `channel-only` | SWaT | 0.3510 | 0.9216 | 0.8594 | Still strongest on raw/affiliation |

Interpretation: the ungated dynamic temporal graph is not enough. The gated
variant improves `full` on MSL raw F1 and SWaT raw/affiliation F1, but it lowers
MSL affiliation F1 and SWaT adjusted F1 in the current-code rerun. The defensible
paper claim is therefore not "both graphs improve every metric"; it is that a
channel graph provides the stable cross-variable structure, while a gated
dynamic temporal graph is an adaptive temporal refinement that can improve
point-wise ranking and SWaT event coverage. At this stage, `full` remains the
safer main method and `dynamic-temporal-gated` should be reported as the
strongest adaptive-temporal candidate.

## Dynamic-Gated Stability Check

A three-seed follow-up was run for `dynamic-temporal-gated` with 15 epochs on
MSL and SWaT. Reported values are the best values over the standard
anomaly-ratio grid for each metric.

| Dataset | Metric | Seed values | Mean | Std | Best ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| MSL | raw F1 | 0.1139 / 0.1170 / 0.1164 | 0.1158 | 0.0016 | 10% / 15% / 10% |
| MSL | adjusted F1 | 0.8577 / 0.8525 / 0.8575 | 0.8559 | 0.0029 | 1% / 1% / 1% |
| MSL | affiliation F1 | 0.6940 / 0.6946 / 0.6917 | 0.6934 | 0.0015 | 1% / 1% / 2% |
| SWaT | raw F1 | 0.3571 / 0.3404 / 0.3275 | 0.3417 | 0.0148 | 5% / 5% / 5% |
| SWaT | adjusted F1 | 0.9196 / 0.9210 / 0.9154 | 0.9187 | 0.0029 | 2% / 2% / 2% |
| SWaT | affiliation F1 | 0.8614 / 0.8471 / 0.8500 | 0.8528 | 0.0075 | 5% / 5% / 5% |

Compared with `full`, `dynamic-temporal-gated` improves mean raw F1 but lowers
mean MSL affiliation F1 and SWaT adjusted F1. SWaT stability was rerun with the
current default `num_workers=2 --prefetch-factor=2`, so the table now uses a
single worker setting. The current conclusion is that gated dynamic temporal
modeling is useful, especially for raw F1 and SWaT affiliation F1, but not a
uniformly dominant replacement for `full`.

## Parallel Fusion and Graph-Shift Check

After finding that the serial dual-graph path did not consistently outperform
single-graph ablations, several alternatives were tested on MSL with seed 2021,
15 epochs, `num_workers=2`, and `prefetch_factor=2`.

| Profile | Main change | Raw F1 | Adjusted F1 | Affiliation F1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `full` | Current compact architecture | 0.1109 | 0.8576 | 0.7006 | Keep as main |
| `parallel-dual-time` | Channel and temporal branches fused by time-wise gate | 0.1189 | 0.8579 | 0.6938 | Reject |
| `residual-dual-time` | `full` plus small time-wise parallel residual | 0.1081 | 0.8577 | 0.6998 | Reject |
| `graph-shift` | Add deviation from normal channel graph to score | 0.1097 | 0.8572 | 0.7001 | Reject |
| `graph-shift-lite` | Lower graph-shift score weight | 0.1105 | 0.8576 | 0.6959 | Reject |

Interpretation: the problem is not solved by changing serial dual-graph
composition into parallel fusion. The gate is still trained by reconstruction
on normal windows, so it has no direct signal for anomaly-time graph selection.
Graph-shift scoring is interpretable, but the same-time channel graph is too
stable on MSL to improve the anomaly ranking. Keep these profiles only for
reproducibility and negative-ablation reporting.

## Lagged Causal Prototype Check

A first lag-constrained causal prototype was added after the graph-fusion
experiments. The module predicts `X_i(t)` from lagged parents `X_j(t-k)` with
`k in {1, 2, 4}` and top-5 sparse parents. To avoid disturbing the main
reconstruction backbone, the current implementation trains this branch on
`resid.detach()` by default. This makes it an explanatory/scoring branch rather
than a module that changes the main representation.

MSL seed-2021, 15 epochs:

| Profile | Main change | Raw F1 | Adjusted F1 | Affiliation F1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `full` | Current compact architecture | 0.1109 | 0.8576 | 0.7006 | Keep as main |
| `causal-lag` | Lagged mechanism branch, no causal score | 0.1101 | 0.8530 | 0.6970 | Reject as main |
| `causal-lag-score` | Detached lagged mechanism score, weight 0.05 | 0.1102 | 0.8574 | 0.6983 | Reject as main |

Interpretation: temporal precedence is now represented in code, which helps the
paper direction, but the first mechanism-residual score is not discriminative
enough to improve affiliation F1. The next causal attempt should use
counterfactual parent masking or intervention-style edge validation instead of
simply adding lagged prediction error to the anomaly score.

## Affiliation-Oriented Scoring Check

Several affiliation-oriented inference variants were tested on MSL with
seed-2021. These were intended to improve event coverage without changing the
main training objective.

| Profile | Main change | Raw F1 | Adjusted F1 | Affiliation F1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `dynamic-temporal-gated` | Baseline gated dynamic temporal graph | 0.1221 | 0.8584 | 0.7014 | Keep |
| `dynamic-temporal-aff` | Time/channel gate + mean smoothing + segment shaping | 0.1156 | 0.6677 | 0.6819 | Reject |
| `dynamic-temporal-aff-lite` | Time/channel gate + light smoothing/gap filling | 0.1099 | 0.7827 | 0.6873 | Reject |
| `dynamic-temporal-aff-peak` | Local max-score expansion | 0.1153 | 0.6973 | 0.6851 | Reject |
| `dynamic-temporal-aff-wide` | Top-8 channel scoring instead of top-5 | 0.1115 | 0.8540 | 0.6974 | Reject |

Interpretation: MSL affiliation quality is hurt by simple segment-level
post-processing and naive score expansion. The current best path remains
`dynamic-temporal-gated`. Future affiliation improvements should target the
learned representation or threshold calibration, not generic smoothing,
dilation, or wider top-k scoring.

## Affiliation Optimization Attempts

After the gated dynamic temporal graph became the best dual-graph candidate,
additional seed-2021 MSL experiments tested whether affiliation F1 could be
improved by graph regularization, robust contaminated-training objectives, and
channel-aware scoring. Reported values are best over the standard anomaly-ratio
grid.

| Profile / setting | Main change | Raw F1 | Adjusted F1 | Affiliation F1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `dynamic-temporal-gated` | Baseline gated dynamic temporal graph | 0.1221 | 0.8584 | 0.7014 | Keep |
| `dynamic-temporal-regularized` | Temporal graph smoothness + locality loss | 0.1139 | 0.8577 | 0.6940 | Reject |
| `dynamic-temporal-robust` | 5% trimmed reconstruction loss after VQ warmup | 0.1179 | 0.8530 | 0.6932 | Reject |
| `dynamic-temporal-robust-lite` | 2% trimmed reconstruction loss after VQ warmup | 0.1163 | 0.8579 | 0.6963 | Reject |
| `dynamic-temporal-channelnorm` | Robust z-score channel-normalized scoring | 0.1176 | 0.8542 | 0.6996 | Reject |
| `dynamic-temporal-channelnorm-scale` | Scale-only channel-normalized scoring | 0.1176 | 0.8542 | 0.6996 | Reject |
| `dynamic-temporal-gated --score-topk-k 3` | Top-3 channel aggregation instead of default top-5 | 0.1156 | 0.8573 | 0.6961 | Reject |
| `dynamic-temporal-gated-vqscore` | Direct VQ score fusion, weight 0.10 | 0.1160 | 0.8583 | 0.6941 | Reject |
| `dynamic-temporal-gated-vqscore --vq-score-weight 0.02` | Direct VQ score fusion, weight 0.02 | 0.1152 | 0.8577 | 0.6958 | Reject |
| `dynamic-temporal-gated --dynamic-temporal-residual-init 0.05` | Weaker dynamic temporal residual | 0.1134 | 0.8578 | 0.6951 | Reject |
| `dynamic-temporal-gated --dynamic-temporal-residual-init 0.20` | Stronger dynamic temporal residual | 0.1153 | 0.8529 | 0.6941 | Reject |
| `dynamic-temporal-gated --dynamic-temporal-topk 8` | Sparser dynamic temporal adjacency | 0.1136 | 0.8577 | 0.6961 | Reject |
| `dynamic-temporal-gated --dynamic-temporal-topk 50` | Denser dynamic temporal adjacency | 0.1129 | 0.8577 | 0.6948 | Reject |

Interpretation: the current shortfall is not solved by adding regularizers or
score calibration layers. The graph regularizer lowered ranking quality, robust
training reduced reconstruction loss without improving anomaly separability, and
channel normalization traded recall for precision. Direct VQ-score fusion also
hurt ranking quality, so VQ should remain a training-side bottleneck rather than
the main anomaly score. The residual-gate and dynamic-topk sweeps indicate that
the default gated temporal graph is already near the best MSL seed-2021 setting
among the tested local variants. For the paper, do not claim these as final
modules. Treat them as negative ablations showing that the selected
`dynamic-temporal-gated` design is not the result of unchecked module stacking.

## Commands

Main unified configuration, using defaults:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile full --save-dir label/LaGraph_main_15ep
```

Equivalent explicit command:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile full --vq-cooldown-epochs 10 --lambda-vq 0.1 --vq-score-weight 0.3 --save-dir label/LaGraph_main_15ep
```

Three-seed final configuration:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv swat.csv --arch-profile full --seed 2021 --save-dir label/LaGraph_main_15ep
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv swat.csv --arch-profile full --seed 2022 --save-dir label/LaGraph_main_15ep_seed2022
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv swat.csv --arch-profile full --seed 2023 --save-dir label/LaGraph_main_15ep_seed2023
```

Key ablations:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile with-scorer --save-dir label/LaGraph_ablation_with_scorer
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile dynamic-temporal --save-dir label/LaGraph_candidate_dynamic_temporal
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile dynamic-temporal-gated --save-dir label/LaGraph_candidate_dynamic_temporal_gated
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile no-vq --save-dir label/LaGraph_ablation_no_vq
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile channel-only --save-dir label/LaGraph_ablation_channel_only
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile temporal-only --save-dir label/LaGraph_ablation_temporal_only
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile reconstruction --save-dir label/LaGraph_ablation_reconstruction
```

Single-dataset sanity checks:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv --arch-profile full --save-dir label/LaGraph_msl_15ep
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets swat.csv --arch-profile full --save-dir label/LaGraph_swat_15ep
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv --arch-profile dynamic-temporal-gated --seed 2021 --save-dir label/LaGraph_candidate_dynamic_temporal_gated_msl
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets swat.csv --arch-profile dynamic-temporal-gated --seed 2021 --num-workers 0 --prefetch-factor 2 --save-dir label/LaGraph_candidate_dynamic_temporal_gated_swat
```

## Reporting Guidance

For a CCF-A style paper, report:

- main results with one unified hyperparameter setting;
- VQ ablation and multi-scale scorer ablation;
- channel-only and temporal-only graph ablations;
- gated dynamic temporal graph ablation;
- reconstruction-only lower bound;
- parameter sensitivity for `vq_score_weight` and `vq_cooldown_epochs`;
- training time and parameter count;
- multiple seeds for the final configuration.

The central claim should be that LaGraph benefits from graph-structured temporal
representation and VQ-regularized representation learning, not from stacking many
loosely justified scoring modules.
