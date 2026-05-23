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
- contrastive auxiliary branch by default
- frequency auxiliary loss by default
- prototype branch by default
- prediction head by default

Auxiliary branches remain available only for explicit ablation or diagnostic runs.

## Runtime Profiles

| Profile | Channel graph | Temporal graph | VQ bypass | Boundary detector | Multi-scale scorer | Intended use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | on | on | on | removed | off | Main compact architecture |
| `with-scorer` | on | on | on | removed | on | Old multi-scale/VQ scoring path for ablation |
| `no-boundary` | on | on | on | removed | on | Compatibility alias for older commands |
| `no-vq` | on | on | off | removed | on | VQ ablation |
| `core` | on | on | off | removed | on | Legacy simplified profile, equivalent to no-vq for current path |
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
scoring. Use `no-vq` as the primary VQ ablation and `reconstruction` as the
lower-bound baseline. Keep `core` only for backward compatibility with earlier
commands.

Do not present dataset-specific VQ settings as the main method. If a dataset
benefits from a different VQ score weight, report it only in parameter
sensitivity analysis.

## Commands

Main unified configuration, using defaults:

```powershell
python ts_benchmark/run_single.py --epochs 15 --arch-profile full --save-dir label/LaGraph_main_15ep
```

Equivalent explicit command:

```powershell
python ts_benchmark/run_single.py --epochs 15 --arch-profile full --vq-cooldown-epochs 10 --lambda-vq 0.1 --vq-score-weight 0.3 --save-dir label/LaGraph_main_15ep
```

Key ablations:

```powershell
python ts_benchmark/run_single.py --epochs 15 --arch-profile with-scorer --save-dir label/LaGraph_ablation_with_scorer
python ts_benchmark/run_single.py --epochs 15 --arch-profile no-vq --save-dir label/LaGraph_ablation_no_vq
python ts_benchmark/run_single.py --epochs 15 --arch-profile reconstruction --save-dir label/LaGraph_ablation_reconstruction
```

Single-dataset sanity checks:

```powershell
python ts_benchmark/run_single.py --epochs 15 --datasets MSL.csv --arch-profile full --save-dir label/LaGraph_msl_15ep
python ts_benchmark/run_single.py --epochs 15 --datasets swat.csv --arch-profile full --save-dir label/LaGraph_swat_15ep
```

## Reporting Guidance

For a CCF-A style paper, report:

- main results with one unified hyperparameter setting;
- VQ ablation and multi-scale scorer ablation;
- reconstruction-only lower bound;
- parameter sensitivity for `vq_score_weight` and `vq_cooldown_epochs`;
- training time and parameter count;
- multiple seeds for the final configuration once the main setting is stable.

The central claim should be that LaGraph benefits from graph-structured temporal
representation and VQ-regularized representation learning, not from stacking many
loosely justified scoring modules.
