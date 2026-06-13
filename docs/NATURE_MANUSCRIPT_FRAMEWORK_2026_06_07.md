# Nature-style manuscript framework for LaGraph

Date: 2026-06-07

## Working positioning

This manuscript should be framed as an industrial anomaly diagnosis paper, not
as a universal time-series anomaly detection leaderboard paper.

One-sentence argument:

> In industrial multivariate time series, anomaly diagnosis fails when source
> variables and propagated response variables are ranked by raw deviation alone.
> We introduce a source-bottleneck graph diagnosis framework that separates
> anomaly-source evidence from propagation evidence, supported by predicted-event
> variable-level RCA on WADI and SWaT and coarse-grained industrial RCA checks on
> HAI, while avoiding claims of strict causal discovery.

Recommended paper type:

- Method paper with industrial cyber-physical-system validation.
- Main task: anomaly detection plus root-cause ranking.
- Main claim level: source/propagation disentanglement for root-cause diagnosis.
- Boundary: verified event-level industrial datasets; not universal SOTA; not
  causal identification.

## Title candidates

1. Source-bottleneck graph diagnosis for industrial multivariate time-series anomalies
2. Separating anomaly sources from propagated responses in industrial time series
3. Graph-guided source bottlenecking for root-cause diagnosis in industrial systems

Best current title:

> Source-bottleneck graph diagnosis for industrial multivariate time-series anomalies

## Abstract skeleton

Industrial anomaly detection models often identify when a process deviates from
normal operation, but deployment-oriented diagnosis also requires identifying
which variables initiated the event. This is difficult because faults propagate
through coupled process variables, so the largest reconstruction errors may
belong to downstream responses rather than root causes.

Here we introduce LaGraph, a source-bottleneck graph diagnosis framework for
industrial multivariate time series. The method combines reconstruction-based
event detection with graph-guided channel evidence, onset evidence, mechanism
residuals, source gating, and event-specificity suppression to separate likely
source variables from propagated responses.

On predicted anomaly events, the current source-bottleneck-specificity profile
achieves variable-level RCA of MRR 0.5960 and Hit@1 0.5000 on WADI, and MRR
0.5087 and Hit@1 0.3939 on SWaT. At subsystem level, the same line reaches MRR
0.7917 on WADI and 0.8102 on SWaT, indicating reliable localization of the
affected process region before variable-level diagnosis. HAI provides additional
coarse-grained industrial RCA support, but its subsystem labels should be used
as supporting rather than decisive evidence.

These results suggest that source-oriented graph diagnosis can improve
deployment-oriented root-cause ranking beyond simple deviation ranking, while
remaining bounded to event-level industrial RCA rather than strict causal
discovery.

## Contribution set

Contribution 1: Source/propagation diagnosis formulation

- Problem: after an industrial event, many variables respond together.
- Method: distinguish source_score from propagation_score rather than ranking
  raw residuals only.
- Evidence: early source-propagation trials improved the explanation interface
  but were weaker than robust baselines, motivating a stronger bottleneck.
- Claim wording: "We formulate industrial RCA as separating source variables
  from propagated responses."

Contribution 2: Source-bottleneck graph RCA

- Problem: residual ranking tends to over-rank high-response variables.
- Method: source gate, sparse/ranking constraints, mechanism residual, onset
  score, graph response penalty.
- Evidence: source-bottleneck-specificity is the current best main profile:
  WADI variable MRR 0.5960 / Hit@1 0.5000; SWaT variable MRR 0.5087 / Hit@1
  0.3939.
- Claim wording: "A source bottleneck encourages a small set of variables to
  explain event-level abnormality, improving root-variable ranking."

Contribution 3: Event-specificity suppression

- Problem: some variables repeatedly rank high across many predicted events and
  behave like generic responders.
- Method: suppress over-general high-response variables at export/evaluation
  time without manually specifying dataset-specific variables.
- Evidence: WADI improves to MRR 0.5960 from the lower source-bottleneck line;
  SWaT remains at the source-bottleneck level.
- Claim wording: "Event-specificity reduces generic response-variable dominance
  without degrading SWaT in the current evidence."

Contribution 4: Deployment-oriented RCA evaluation

- Problem: true-event RCA is optimistic because it assumes the anomaly interval
  is already known.
- Method: report predicted-event RCA with matched rate, coverage/IoU checks,
  MRR, Hit@K, MAP@K, NDCG@K, and hierarchical subsystem/variable views.
- Evidence: predicted-event variable and subsystem results are available for
  WADI and SWaT; HAI supports coarse subsystem validation.
- Claim wording: "We evaluate diagnosis after model-predicted events, not only
  under oracle event intervals."

## Section outline

### Introduction

Paragraph 1: Industrial motivation

Industrial cyber-physical systems are monitored by many coupled sensors and
actuators. Detection alone answers when an event occurs, but operation and
incident response require identifying the variables or process units most
responsible for the event.

Paragraph 2: Technical bottleneck

In multivariate systems, abnormality propagates. A source variable can trigger
downstream variables, and downstream variables can have larger deviations than
the source. Therefore, raw residual magnitude or z-score ranking is not a
reliable root-cause explanation.

Paragraph 3: Prior-work pressure

Existing reconstruction models, graph neural time-series models, and dual graph
detectors can model temporal and channel dependencies, but a dual graph alone is
not a sufficient novelty claim. Many graph-based TSAD methods already combine
feature and temporal structure, and learned graph edges are not automatically
causal or faithful explanations.

Paragraph 4: Gap

The unresolved problem is deployment-oriented root-cause diagnosis: after a
model predicts an anomaly event, can it rank the event-specific source variables
rather than generic response variables, under a transparent evaluation protocol?

Paragraph 5: Present study

Introduce LaGraph as a source-bottleneck graph diagnosis framework. State the
three technical elements: source/propagation decomposition, source-bottleneck
scoring, and event-specificity suppression. End with bounded evidence on WADI,
SWaT, and HAI.

### Methods

1. Problem definition: predicted-event RCA for industrial multivariate time series.
2. Backbone: reconstruction-based event detector with channel and temporal graph evidence.
3. Source/propagation evidence:
   - base residual
   - source score
   - propagation score
   - mechanism residual
   - onset evidence
4. Source-bottleneck mechanism:
   - source gate
   - sparsity/ranking pressure
   - source-oriented final score
5. Event-specificity suppression:
   - identify generic high-response variables across predicted events
   - suppress over-general responders
   - keep this framed as an export-stage constraint unless further end-to-end
     training evidence is added
6. RCA evaluation:
   - predicted-event and true-event settings
   - variable-level and subsystem-level metrics
   - matched, coverage@10, IoU@10/30, MRR, Hit@K, MAP@K, NDCG@K

### Results

Result 1: Detection is sufficient but not the main novelty

- Show WADI/SWaT event detection quality and threshold sensitivity.
- Message: detection does not collapse, but the paper is not a detector-only
  SOTA claim.
- Figure/table: detection metrics across threshold keys, including raw F1,
  adjusted F1, Affiliation F1, AUROC/AUPRC where available.

Result 2: Raw deviation and early source-propagation scoring are insufficient

- Show why simple residual/source-propagation trials do not solve variable RCA.
- Message: source/propagation decomposition is conceptually necessary, but
  requires a stronger bottleneck.
- Figure/table: robust baseline, z-score/static baselines, early source-only
  and source-propagation variants.

Result 3: Source-bottleneck-specificity improves variable-level predicted-event RCA

- Main table:
  - WADI: MRR 0.5960, Hit@1 0.5000, Hit@3 0.7143, Hit@5 0.7143.
  - SWaT: MRR 0.5087, Hit@1 0.3939, Hit@3 0.5758, Hit@5 0.6667.
- Compare against random, z-score, correlation prior, random graph prior, and
  earlier LaGraph profiles.
- Message: exact root-variable ranking is the strongest paper result.

Result 4: Hierarchical RCA localizes process regions before variables

- Main table:
  - WADI subsystem: MRR 0.7917, Hit@1 0.6429, Hit@3 0.9286, Hit@5 1.0000.
  - SWaT subsystem: MRR 0.8102, Hit@1 0.7097, Hit@3 0.9032, Hit@5 0.9677.
  - HAI subsystem predicted-event and true-event results as supplemental support.
- Message: subsystem RCA is reliable but easier than variable-level RCA; do not
  make it the only claim.

Result 5: Ablations show which components are necessary

Required ablation table:

- full source-bottleneck-specificity
- no source gate
- no source bottleneck
- no mechanism residual
- no onset evidence
- no propagation suppression
- no event-specificity
- source-bottleneck-corefine as a deeper-coupling negative/secondary attempt

Message: the final architecture is selected by evidence, not module stacking.

Result 6: Case studies make the source/response distinction inspectable

Recommended case study panels:

- one WADI event where root variable becomes top-1 after specificity
- one SWaT event where subsystem is correct but variable ranking remains hard
- one failure case where propagation or boundary mismatch explains the error

Each case should show:

- anomaly score and predicted event boundary
- top variables by residual vs final source-bottleneck score
- source/onset/mechanism/propagation components
- subsystem ranking

### Discussion

Opening:

The central advance is not a larger anomaly detector, but a diagnostic
formulation that separates likely source variables from propagated responses in
industrial event-level RCA.

Interpretation:

- Source bottleneck converts "many variables are abnormal" into "which few
  variables best explain the event."
- Event-specificity handles generic responders.
- Hierarchical RCA reflects industrial troubleshooting practice.

Relation to prior work:

- Position against graph TSAD and dual graph models.
- Emphasize that feature/temporal graph combinations already exist.
- Differentiate through source/propagation diagnosis and predicted-event RCA.

Limitations:

- Not strict causal discovery.
- WADI/SWaT label protocols differ.
- HAI is mainly subsystem-level.
- Event-specificity is currently export-stage, not fully end-to-end.
- More seed stability and final ablations are needed before final submission.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Detection alone is not the strongest paper story | Prior reports show event metrics are usable but detector-only gains are limited | Supported |
| Source/propagation separation is the right RCA formulation | Early source-propagation experiments show interpretability value but weak ranking, motivating bottleneck | Supported as motivation |
| Source-bottleneck-specificity is current best main model | WADI variable MRR 0.5960; SWaT variable MRR 0.5087 | Supported, needs final ablation/seed checks |
| Event-specificity improves WADI without harming SWaT in current evidence | WADI improves; SWaT remains at source-bottleneck level | Supported, but export-stage |
| Hierarchical RCA is reliable at process-region level | WADI subsystem MRR 0.7917; SWaT subsystem MRR 0.8102; HAI coarse RCA support | Supported, but not sufficient alone |
| The method performs causal discovery | No intervention or validated causal assumptions yet | Do not claim |
| The dual graph universally outperforms single-graph variants | Prior ablations show mixed results | Do not claim |
| The method is universal SOTA anomaly detection | Evidence and framing do not support this | Do not claim |

## Figures and tables

Figure 1: Problem and framework

- left: source variable causing propagated responses
- middle: channel-temporal graph reconstruction
- right: source-bottleneck RCA output

Figure 2: Evaluation protocol

- anomaly score to predicted event
- predicted-event RCA
- variable and subsystem ranking
- strict coverage/IoU checks

Table 1: Dataset and label protocol

- WADI, SWaT, HAI, optional TE
- detection label, RCA granularity, verified source, role in paper

Table 2: Detection metrics

- WADI and SWaT
- threshold key 15 and best Affiliation F1 row
- raw F1, adjusted F1, Affiliation F1, AUROC, AUPRC

Table 3: Main predicted-event variable RCA

- source-bottleneck-specificity vs random, z-score, correlation prior, random
  graph prior, earlier LaGraph variants

Table 4: Hierarchical subsystem RCA

- WADI, SWaT, HAI
- subsystem MRR, Hit@1, Hit@3, Hit@5

Table 5: Ablation

- component removal and negative architecture attempts

Figure 3: Case study

- residual ranking vs source-bottleneck ranking
- evidence decomposition for top variables

## Assumptions and missing inputs

Critical missing inputs before manuscript drafting:

1. Final frozen profile name and command for the main model.
2. Final detection table for the same runs used in RCA tables.
3. Standard ablations for source gate, bottleneck, mechanism residual, onset,
   propagation suppression, and event-specificity.
4. Multi-seed or epoch-stability table for the main profile.
5. Exact baseline table for WADI variable-level RCA under the same predicted
   events as the main model.
6. A small set of inspectable case studies with event IDs and top-K variables.
7. Final wording decision: whether to use "LaGraph" as the model name or a new
   name such as "Source-Bottleneck LaGraph".

## Recommended manuscript next step

Freeze the manuscript around:

> source-bottleneck-specificity-rca as the main model, with source-bottleneck
> corefinement and stronger causal variants treated as negative or secondary
> architecture attempts.

Do not add major modules before the first full draft. The next useful work is
to finish ablations, seed stability, and case-study evidence for the current
mainline.

