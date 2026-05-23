# LaGraph Architecture Simplification Plan

Date: 2026-05-23
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

Use `full` as the main method in future experiments.

Use `with-scorer` to report the old multi-scale/VQ scoring path as an ablation,
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
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile no-vq --save-dir label/LaGraph_ablation_no_vq
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile channel-only --save-dir label/LaGraph_ablation_channel_only
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile temporal-only --save-dir label/LaGraph_ablation_temporal_only
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --arch-profile reconstruction --save-dir label/LaGraph_ablation_reconstruction
```

Single-dataset sanity checks:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv --arch-profile full --save-dir label/LaGraph_msl_15ep
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark/run_single.py --epochs 15 --datasets swat.csv --arch-profile full --save-dir label/LaGraph_swat_15ep
```

## Reporting Guidance

For a CCF-A style paper, report:

- main results with one unified hyperparameter setting;
- VQ ablation and multi-scale scorer ablation;
- channel-only and temporal-only graph ablations;
- reconstruction-only lower bound;
- parameter sensitivity for `vq_score_weight` and `vq_cooldown_epochs`;
- training time and parameter count;
- multiple seeds for the final configuration.

The central claim should be that LaGraph benefits from graph-structured temporal
representation and VQ-regularized representation learning, not from stacking many
loosely justified scoring modules.
