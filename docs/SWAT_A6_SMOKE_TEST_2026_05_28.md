# SWaT A6 Dec 2019 Smoke Test - 2026-05-28

## Purpose

This note records the inspection and first-use result for the SWaT files placed
under `D:\google\2\SWAT`.

The downloaded package is SWaT A6 Dec 2019, not the common SWaT A1/A2 benchmark
used by most anomaly-detection papers. It contains historian values, pcap
captures, and a `Log.docx` attack timeline.

## Files Used

Only the files needed for time-series experiments were copied into the project
raw-data area:

```text
D:\la_v12\dataset\anomaly_detect\data\raw\SWAT\SWaT.A6_Dec_2019\Dec2019.xlsx
D:\la_v12\dataset\anomaly_detect\data\raw\SWAT\SWaT.A6_Dec_2019\Log.docx
D:\la_v12\dataset\anomaly_detect\data\raw\SWAT\SWaT.A6_Dec_2019\SWaT equipment list & glossary.pdf
```

The pcap files and 9GB zip were not copied because the current LaGraph pipeline
uses historian time series, not packet traces.

## Label Decision

`Log.docx` contains both cyber-only and process-disruption phases:

```text
10:30-11:20: historian data exfiltration cycles
12:30-13:25: disrupt sensor and actuator cycles
```

For process time-series anomaly detection, only the five "Disrupt Sensor and
Actuator" windows are labeled as anomalies:

| Event | Test-relative start | Test-relative end | Clock time |
|---:|---:|---:|---|
| 1 | 0 | 179 | 12:30:00-12:32:59 |
| 2 | 780 | 959 | 12:43:00-12:45:59 |
| 3 | 1560 | 1739 | 12:56:00-12:58:59 |
| 4 | 2340 | 2519 | 13:09:00-13:11:59 |
| 5 | 3120 | 3299 | 13:22:00-13:24:59 |

Earlier exfiltration periods are not labeled because they are cyber/data-theft
events and may not be visible in process sensor values.

## Converted Dataset

Converted file:

```text
D:\la_v12\dataset\anomaly_detect\data\SWAT_A6_Dec2019_process.csv
```

Metadata:

```text
length=13,201
train_lens=8,700
features=81
attack points=900
freq=second
```

The converter is:

```text
D:\la_v12\scripts\convert_swat_a6_dataset.py
```

## Smoke-Test Results

All runs use 5 epochs and `mechanism-prior-graph`.

| Setting | Best metric | Ratio | F1 | Precision | Recall | Adjusted F1 | Affiliation F | Aff. Precision | Aff. Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWaT A6 process | F1 | 15.0 | 0.3168 | 0.1912 | 0.9233 | 0.3387 | 0.6869 | 0.5234 | 0.9989 |
| SWaT A6 process | adjusted F1 | 5.0 | 0.3108 | 0.1941 | 0.7800 | 0.3818 | 0.6620 | 0.5086 | 0.9480 |
| SWaT A6 process | affiliation F | 15.0 | 0.3168 | 0.1912 | 0.9233 | 0.3387 | 0.6869 | 0.5234 | 0.9989 |
| SWaT A6 + robust clip `q=0.01/0.99` | F1 | 5.0 | 0.3374 | 0.3393 | 0.3356 | 0.3896 | 0.6173 | 0.6250 | 0.6098 |
| SWaT A6 + robust clip `q=0.01/0.99` | adjusted F1 | 2.0 | 0.1917 | 0.7444 | 0.1100 | 0.5564 | 0.5973 | 0.7528 | 0.4950 |
| SWaT A6 + robust clip `q=0.01/0.99` | affiliation F | 15.0 | 0.3302 | 0.2363 | 0.5478 | 0.4482 | 0.7045 | 0.6315 | 0.7966 |

## Interpretation

The dataset is usable by the project, but it should not replace the common
`swat.csv` benchmark:

1. It is a short A6 scenario with only 13,201 seconds.
2. The official log provides process-disruption intervals but not exact
   attacked sensor/actuator tags.
3. The early attack phase is cyber-only exfiltration, so labeling it as a
   process anomaly would be scientifically weak.
4. Robust input clipping improves raw and adjusted behavior in some threshold
   regimes, but the result is still a supplemental stress test rather than
   headline evidence.

Recommended paper use:

```text
Use SWaT A6 as supplemental evidence for robustness on log-derived process
disruption events. Do not use it as verified RCA evidence unless exact attacked
tags are confirmed from source documentation.
```
