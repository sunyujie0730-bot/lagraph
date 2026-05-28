# WADI A2 Smoke Test - 2026-05-28

## Purpose

This test checks whether the downloaded WADI A2 2019 data can be used by the
current LaGraph pipeline and whether it is suitable for later detection + RCA
experiments.

The test is not a final benchmark. All runs use only 5 epochs on the
downsampled `ds10` version.

## Data Variants

| Dataset file | Features | Processing |
|---|---:|---|
| `WADI_A2_2019_ds10.csv` | 127 | Official normal + attack files converted to LaGraph long format |
| `WADI_A2_2019_ds10_stable.csv` | 93 | Drops normal-training near-zero-variance channels (`std < 1e-4`) |
| `WADI_A2_2019_ds10_stable_drop2B002.csv` | 92 | Stable version plus drops `2B_AIT_002_PV` |

All variants keep the same time length:

```text
length=95,739
train_lens=78,458
attack-file length=17,281
attack points=997
```

## Key Data Finding

The raw WADI A2 attack file contains a major normal-label distribution shift in
`2B_AIT_002_PV`.

In the converted `ds10` test segment:

```text
2B_AIT_002_PV > 1000 for 9,423 / 17,281 test points
Only 392 of those 9,423 points are labeled as attacks
The high-scale segment starts around test index 7,858 and lasts to the end
```

Training-normal statistics for the same variable:

```text
train mean ~= 9.10
train std  ~= 0.125
```

Attack-file statistics:

```text
attack-file mean ~= 4,427.86
attack-file std  ~= 4,035.36
max value ~= 8,128
```

This single variable causes systematic false positives under train-only
standardization. This is a dataset quality / operating-regime issue, not merely
a model-capacity issue.

## Smoke-Test Results

Best rows are selected separately by raw F1, adjusted F1, and affiliation F.

| Setting | Best metric | Ratio | F1 | Precision | Recall | Adjusted F1 | Affiliation F | Aff. Precision | Aff. Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw ds10 | F1 | 15.0 | 0.1271 | 0.0689 | 0.8245 | 0.1521 | 0.7119 | 0.5529 | 0.9991 |
| raw ds10 | adjusted F1 | 2.0 | 0.1144 | 0.0629 | 0.6299 | 0.1758 | 0.7037 | 0.5454 | 0.9916 |
| raw ds10 | affiliation F | 1.0 | 0.1072 | 0.0590 | 0.5797 | 0.1683 | 0.7432 | 0.5964 | 0.9860 |
| stable ds10 | F1 | 5.0 | 0.1214 | 0.0666 | 0.6891 | 0.1715 | 0.7019 | 0.5427 | 0.9932 |
| stable ds10 | adjusted F1 | 2.0 | 0.1145 | 0.0630 | 0.6289 | 0.1761 | 0.7025 | 0.5444 | 0.9899 |
| stable ds10 | affiliation F | 2.0 | 0.1145 | 0.0630 | 0.6289 | 0.1761 | 0.7025 | 0.5444 | 0.9899 |
| drop `2B_AIT_002_PV` | F1 | 5.0 | 0.3139 | 0.2441 | 0.4393 | 0.5578 | 0.7079 | 0.5633 | 0.9527 |
| drop `2B_AIT_002_PV` | adjusted F1 | 2.0 | 0.2987 | 0.3393 | 0.2668 | 0.6473 | 0.6551 | 0.5528 | 0.8038 |
| drop `2B_AIT_002_PV` | affiliation F | 15.0 | 0.2081 | 0.1257 | 0.6038 | 0.3011 | 0.7117 | 0.5572 | 0.9847 |

## Interpretation

The raw and stable variants over-report anomalies because the test distribution
contains a long high-scale segment in one variable that is mostly labeled
normal. Removing only this suspicious variable changes the behavior
substantially:

```text
raw best F1:      0.1271
cleaned best F1:  0.3139
```

Near-zero-variance filtering alone does not help, so the dominant issue is not
constant channels. The dominant issue is the single high-scale normal-label
shift in `2B_AIT_002_PV`.

## Paper-Relevant Decision

WADI A2 should not be introduced as a final benchmark until the preprocessing
rule is fixed and justified. A defensible route is:

1. Keep raw WADI A2 as a data-quality diagnostic.
2. Use the stable/drop version only if the paper explicitly reports the sensor
   exclusion rule and explains it as an official-data scale inconsistency.
3. Prefer WADI for robustness and RCA validation after attack target labels are
   reconciled from `table_WADI.pdf`.

The current evidence supports using WADI as a valuable industrial stress test,
but not yet as a headline performance benchmark.
