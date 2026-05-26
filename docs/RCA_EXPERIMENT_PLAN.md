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

The first implemented RCA signal is channel-wise reconstruction contribution. The current improved RCA signal adds a local counterfactual contrast term:

```text
score_i = event_error_i
        + graph_weight * graph_propagated_error_i
        + contrast_weight * max(event_error_i - pre_event_baseline_i, 0)
```

The current recommended HAI setting is:

- `rca_contrast_window = 1000`
- `rca_contrast_weight = 0.75`
- `rca_graph_weight = 0.0`

The graph propagation path is implemented and exported, but the first HAI test showed no stable subsystem-level ranking gain from graph propagation alone. It should be treated as an interpretable component to keep testing, not as the main current source of improvement.

For each true anomaly event, LaGraph now exports:

- channel ranking: variables sorted by mean channel reconstruction error inside the event
- group ranking: subsystem-level aggregation, currently based on HAI prefixes such as `P1`, `P2`, `P3`

Export command example:

```powershell
python ts_benchmark/run_single.py --epochs 1 --datasets HAI_21_03_test1.csv --eval-config unfixed_detect_label_config.json --export-rca --rca-contrast-window 1000 --rca-contrast-weight 0.75 --num-workers 2 --prefetch-factor 2
```

RCA evaluation example:

```powershell
python scripts/evaluate_rca.py --rca result/rca/HAI_21_03_test1/<timestamp>_rca.json --scope group
python scripts/evaluate_rca.py --rca result/rca/HAI_21_03_test1/<timestamp>_rca.json --scope group --score-mode components --component-contrast-weight 0.75
python scripts/evaluate_rca.py --rca result/rca/HAI_21_03_test1/<timestamp>_rca.json --scope group --event-source predicted --prediction-key pot
```

RCA JSON now stores two event sources:

- `events`: RCA over true anomaly intervals. This isolates localization quality from detection quality.
- `predicted_events_by_key`: RCA over model-predicted anomaly intervals for each threshold key such as `pot`, `0.5`, `1.0`, `2`, and `5`. This reflects deployment behavior.

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
python scripts/evaluate_rca.py --rca result/rca/HAI_21_03_test2/<timestamp>_rca.json --scope group --baseline random --random-trials 1000
python scripts/evaluate_hai_zscore_rca.py --test-file test2.csv.gz
python scripts/summarize_hai_rca_results.py
```

The random baseline should use multiple shuffled trials because HAI subsystem RCA has a small candidate space. A single random shuffle can look artificially strong on events with multiple true root groups.

## Current HAI RCA Results

The current 5-file HAI subsystem-level RCA evaluation covers 50 attack events.

Mean metrics across `HAI_21_03_test1` through `HAI_21_03_test5`:

| Method | MRR | Hit@1 | NDCG@3 | NDCG@5 |
|---|---:|---:|---:|---:|
| LaGraph-contrast | 0.9867 | 0.9733 | 0.9462 | 0.9717 |
| LaGraph | 0.9867 | 0.9733 | 0.9275 | 0.9680 |
| z-score | 0.9667 | 0.9433 | 0.9299 | 0.9554 |
| random | 0.5924 | 0.3386 | 0.5759 | 0.6917 |

Interpretation:

- The local contrast term improves ranking quality mainly on multi-root events, especially NDCG@3.
- It does not improve Hit@1 over the original LaGraph on average, but it keeps Hit@1 stable while improving ranking order.
- Compared with z-score, LaGraph-contrast has higher average MRR, Hit@1, NDCG@3, and NDCG@5 on the current HAI subsystem RCA benchmark.
- The gain is useful for paper framing, but it is not yet enough to claim strong variable-level causal root-cause localization.

## Predicted-Event RCA Protocol

Predicted-event RCA answers a stricter question:

```text
Can the deployed detector both cover the true event and rank the correct root subsystem?
```

Evaluation is done per true attack event. For each true event, the evaluator finds the overlapping predicted event with maximum overlap. If no predicted event overlaps, the RCA metrics for that true event are zero. This means `matched` is the event coverage rate under the selected detection threshold.

Additional metric:

```text
RCA_Delay@K = max(0, predicted_event_start - true_event_start)
```

The delay is reported only when the root subsystem appears in Top-K.

Predicted-event RCA averaged across all five HAI test files:

| Prediction key | matched | MRR | Hit@1 | NDCG@3 | NDCG@5 | RCA_Delay@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9900 | 0.9375 | 0.8933 | 0.8860 | 0.9308 | 18.40 |
| 5 | 1.0000 | 0.9425 | 0.8933 | 0.8846 | 0.9338 | 18.00 |
| 1.0 | 0.9800 | 0.9275 | 0.8833 | 0.8760 | 0.9208 | 19.62 |
| 0.5 | 0.9600 | 0.9275 | 0.9000 | 0.8709 | 0.9157 | 23.52 |
| pot | 0.8867 | 0.8617 | 0.8367 | 0.8321 | 0.8514 | 24.43 |

Interpretation:

- Predicted-event RCA is dominated by event coverage. POT is conservative and misses events on HAI test2/test5.
- Threshold key `2` is the current best balance on NDCG@3 and delay; key `5` gives complete event coverage but likely costs more detection precision.
- Wider thresholds improve `matched`, but the detection false-positive cost must be reported separately through anomaly-detection metrics.
- For the paper, report both protocols: true-event RCA as localization upper bound, and predicted-event RCA as deployment-oriented diagnosis.

## TE RCA Ground Truth

Current TE converted files use only the first 53 process variables and exclude the disturbance variables. Therefore, strict variable-level root-cause labels are not yet available for TE.

TE can still be used for:

- anomaly detection evaluation
- qualitative variable ranking visualization

For strict RCA evaluation on TE, the next step is to build a fault-ID to affected observed-variable mapping from TE process knowledge or expert annotation.

## Current Limitations

1. RCA is currently validated at subsystem level, not exact sensor/actuator level.
2. Graph-propagated attribution is implemented, but its current subsystem-level gain is weak.
3. The current RCA evaluation uses true anomaly intervals; predicted-event RCA and delay analysis still need to be added.
4. TE variable-level RCA requires a reliable fault-to-variable mapping.

## Next Steps

1. Continue testing graph and lagged-causal influence in RCA score:

```text
rca_i = reconstruction_contribution_i
      + graph_shift_weight * graph_influence_i
      + causal_weight * outgoing_lagged_influence_i
```

2. Add root-cause delay:

```text
first time root subsystem enters Top-K - anomaly start time
```

3. Run predicted-event RCA for all HAI test files and report threshold sensitivity.

4. Build variable-level labels for HAI where attack descriptions allow exact component mapping.

5. Build TE fault-to-observed-variable mapping before claiming strict TE RCA.

6. Compare against simple RCA baselines:

- raw reconstruction error
- z-score deviation
- integrated gradients, if practical
- graph-centrality weighted reconstruction contribution
