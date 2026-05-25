# Root-Cause Analysis Experiment Plan

## Goal

Upgrade LaGraph from a pure anomaly detector to an industrial diagnosis framework:

1. Detect anomalous time intervals.
2. Rank likely root-cause variables or subsystems for each anomaly event.
3. Quantify root-cause ranking quality with ranking metrics.

## Task Split

### Anomaly Detection

Question: when does an anomaly happen?

Metrics:

- raw F1
- adjusted F1
- Affiliation-F1
- AUROC / AUPRC
- detection delay
- training and inference cost

### Root-Cause Tracking

Question: which variable or subsystem is most responsible for the anomaly?

Metrics:

- Hit@1 / Hit@3 / Hit@5
- MRR
- NDCG@K
- Precision@K
- Recall@K
- root-cause delay, added later

## Current Implementation

The first implemented RCA signal is channel-wise reconstruction contribution.

For each true anomaly event, LaGraph now exports:

- channel ranking: variables sorted by mean channel reconstruction error inside the event
- group ranking: subsystem-level aggregation, currently based on HAI prefixes such as `P1`, `P2`, `P3`

Export command example:

```powershell
python ts_benchmark/run_single.py --epochs 1 --datasets HAI_21_03_test1.csv --eval-config unfixed_detect_label_config.json --export-rca --num-workers 2 --prefetch-factor 2
```

RCA evaluation example:

```powershell
python scripts/evaluate_rca.py --rca result/rca/HAI_21_03_test1/<timestamp>_rca.json --scope group
```

## HAI RCA Ground Truth

For `HAI_21_03_test1.csv`, ground truth is built from `attack_P1`, `attack_P2`, and `attack_P3`.
The same builder supports `test1.csv.gz` through `test5.csv.gz`.

Current task level: subsystem-level RCA.

- `attack_P1 = 1` means root group `P1`
- `attack_P2 = 1` means root group `P2`
- `attack_P3 = 1` means root group `P3`

This is suitable for the first RCA experiment because HAI has explicit process-area attack labels.

Current converted HAI files:

- `HAI_21_03_test1.csv`: 5 attack events
- `HAI_21_03_test2.csv`: 20 attack events
- `HAI_21_03_test3.csv`: 8 attack events
- `HAI_21_03_test4.csv`: 5 attack events
- `HAI_21_03_test5.csv`: 12 attack events

This gives 50 subsystem-level RCA events in total.

## Baselines

Subsystem-level RCA should not be reported without baselines. Current baselines:

- random subsystem ranking
- train-normal z-score deviation aggregated by subsystem

Example commands:

```powershell
python scripts/evaluate_rca.py --rca result/rca/HAI_21_03_test2/<timestamp>_rca.json --scope group --baseline random
python scripts/evaluate_hai_zscore_rca.py --test-file test2.csv.gz
python scripts/summarize_hai_rca_results.py
```

## TE RCA Ground Truth

Current TE converted files use only the first 53 process variables and exclude the disturbance variables. Therefore, strict variable-level root-cause labels are not yet available for TE.

TE can still be used for:

- anomaly detection evaluation
- qualitative variable ranking visualization

For strict RCA evaluation on TE, the next step is to build a fault-ID to affected observed-variable mapping from TE process knowledge or expert annotation.

## Current Limitations

1. RCA ranking currently uses reconstruction contribution only.
2. It does not yet use causal lag direction or graph propagation strength.
3. HAI evaluation is currently subsystem-level, not exact actuator/sensor-level.
4. TE variable-level RCA requires a reliable fault-to-variable mapping.

## Next Steps

1. Add graph influence into RCA score:

```text
rca_i = reconstruction_contribution_i
      + graph_shift_weight * graph_influence_i
      + causal_weight * outgoing_lagged_influence_i
```

2. Add root-cause delay:

```text
first time root subsystem enters Top-K - anomaly start time
```

3. Build variable-level labels for HAI where attack descriptions allow exact component mapping.

4. Build TE fault-to-observed-variable mapping before claiming strict TE RCA.

5. Compare against simple RCA baselines:

- raw reconstruction error
- z-score deviation
- integrated gradients, if practical
- graph-centrality weighted reconstruction contribution
