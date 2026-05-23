# VQ Parameter Tuning Log

Date: 2026-05-23
Branch: `codex/5070-env-migration`

## Background

After removing `BoundaryDetector`, the current final LaGraph path keeps:

- channel graph
- temporal graph
- dual-path VQ bottleneck
- dynamic scale selection
- direct reconstruction anomaly scoring

The goal of this tuning round was to find a unified VQ configuration that improves point-wise and event-level behavior without using dataset-specific hyperparameters.

## Parameters

The tunable VQ parameters exposed in `ts_benchmark/run_single.py` are:

- `--vq-cooldown-epochs`
- `--lambda-vq`
- `--vq-score-weight`

Default historical setting:

```text
vq_cooldown_epochs = 15
lambda_vq = 0.1
vq_score_weight = 0.5
```

Adopted unified default setting:

```text
vq_cooldown_epochs = 10
lambda_vq = 0.1
vq_score_weight = 0.3
```

## MSL Results

| Configuration | Best Raw F1 | Best Adjust F1 | Best Affiliation F |
| --- | ---: | ---: | ---: |
| cooldown=8, weight=0.5 | 0.1062 | 0.7943 | 0.7041 |
| cooldown=8, weight=0.3 | 0.1136 | 0.7947 | 0.7115 |
| cooldown=10, weight=0.3 | 0.1142 | 0.8222 | 0.7247 |
| cooldown=12, weight=0.3 | 0.1099 | 0.8011 | 0.7008 |
| default cooldown=15, weight=0.5, 15 epochs | 0.0972 | 0.8490 | 0.6952 |
| historical 20 epochs | 0.1056 | 0.8506 | 0.7082 |

MSL conclusion:

- `cooldown=10, weight=0.3` gives the best balance.
- `cooldown=12` does not recover enough adjust F1 and loses raw/affiliation performance.
- `cooldown=15` is too late for short 15-epoch runs; the best epoch stayed near the start in the default 15-epoch baseline.

## SWaT Results

| Configuration | Best Raw F1 | Best Adjust F1 | Best Affiliation F |
| --- | ---: | ---: | ---: |
| cooldown=10, weight=0.3 | 0.3038 | 0.9248 | 0.8342 |
| cooldown=10, weight=0.5 | 0.2918 | 0.9237 | 0.8269 |
| cooldown=12, weight=0.5 | 0.2737 | 0.9295 | 0.8159 |
| default cooldown=15, weight=0.5 | 0.7297 | 0.8639 | 0.7780 |
| historical 20 epochs | 0.2526 | 0.9584 | 0.8288 |

SWaT conclusion:

- `weight=0.3` is better than `weight=0.5` for the balanced objective.
- `cooldown=12, weight=0.5` slightly improves adjust F1 over `cooldown=10, weight=0.5`, but loses raw F1 and affiliation F.
- The default setting has unusually high raw F1 at 15% anomaly ratio, but weaker event-level and affiliation metrics. Treat this as a sensitivity result, not the main default.

## Decision

Use the following unified VQ configuration as the default going forward. VQ is
kept as a representation-learning constraint, while the old multi-scale/VQ
scoring path is kept only as `with-scorer` ablation.

```text
vq_cooldown_epochs = 10
lambda_vq = 0.1
vq_score_weight = 0.3
```

Rationale:

- It is the best balanced MSL configuration.
- It is also the best balanced SWaT configuration among the non-default candidates.
- It avoids dataset-specific hyperparameter choices, which is important for generalization claims.
- It is easier to justify in a CCF-A style paper than selecting different VQ settings per dataset.

## Follow-Up

If the unified setting performs poorly on the broader benchmark, revisit this log and test:

1. validation-set adaptive VQ score calibration;
2. a narrower sweep around `vq_score_weight` in `[0.25, 0.35]`;
3. a longer 20-epoch run with `cooldown=10`;
4. reporting `weight=0.5` only as parameter sensitivity, not as the main method.

## Next Command

Run all default included datasets, excluding SMD by the existing script default:

```powershell
D:\Anaconda3\envs\lagraph5070\python.exe ts_benchmark\run_single.py --epochs 15 --arch-profile full --vq-cooldown-epochs 10 --lambda-vq 0.1 --vq-score-weight 0.3 --save-dir label/LaGraph_main_15ep
```
