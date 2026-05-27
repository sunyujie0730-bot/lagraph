# Paper Validity Check: HAI RCA Baselines

Date: 2026-05-27

This note records a reviewer-oriented check before freezing the current
architecture. The goal is to test whether the current LaGraph RCA story is
strong enough, or whether the architecture still needs a stronger mechanism
before full ablation.

## Question

The key reviewer question is:

```text
Is LaGraph's RCA capability genuinely stronger than simple reconstruction,
z-score deviation, or a static normal-correlation prior?
```

If the answer is no, then full ablation is premature because the main scientific
claim is not yet strong enough.

## Compared Methods

| Method | Description |
|---|---|
| `random` | Random subsystem ranking |
| `base_only` | LaGraph channel reconstruction contribution only |
| `mechanism_only` | Channel mechanism-violation contribution only |
| `base_plus_mechanism` / `exported` | Current exported LaGraph RCA score |
| `zscore` | Static normal-training z-score deviation baseline |
| `correlation_prior` | Static z-score evidence propagated through normal-training absolute correlation graph |

## True-Event RCA

True-event RCA uses ground-truth anomaly intervals. This setting tests ranking
quality after the anomaly segment is already known.

| Method | MRR | Hit@1 | Hit@3 | NDCG@3 |
|---|---:|---:|---:|---:|
| LaGraph exported | 0.9700 | 0.9400 | 1.0000 | 0.9421 |
| LaGraph base only | 0.9700 | 0.9400 | 1.0000 | 0.9344 |
| LaGraph mechanism only | 0.8850 | 0.8200 | 0.9400 | 0.8273 |
| z-score | 0.9400 | 0.9000 | 1.0000 | 0.9256 |
| correlation prior | 0.9333 | 0.8800 | 1.0000 | 0.9192 |
| random | 0.5766 | 0.3188 | 0.8209 | 0.5664 |

Interpretation:

LaGraph is clearly better than random and slightly better than static z-score on
true-event RCA. However, the gap is not large enough to claim that the current
mechanism module is the main reason for strong RCA. HAI subsystem-level labels
are also relatively easy because many events produce direct subsystem deviation.

## Predicted-Event RCA

Predicted-event RCA is stricter: the method must first produce an event interval,
then rank root causes on that predicted interval.

| Method | Matched | MRR | Hit@1 | Hit@3 | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| LaGraph exported | 0.8800 | 0.8250 | 0.7800 | 0.8600 | 0.7931 |
| LaGraph base only | 0.8800 | 0.8250 | 0.7800 | 0.8600 | 0.7931 |
| LaGraph mechanism only | 0.8800 | 0.7717 | 0.7000 | 0.8600 | 0.7376 |
| z-score on predicted intervals | 0.8800 | 0.8267 | 0.7800 | 0.8800 | 0.8157 |
| correlation prior on predicted intervals | 0.8800 | 0.8067 | 0.7400 | 0.8800 | 0.8099 |
| random | 0.8800 | 0.5158 | 0.2903 | 0.7328 | 0.5052 |

Interpretation:

The current LaGraph RCA ranking is not meaningfully better than a z-score
baseline when evaluated on the same predicted intervals. This is the strongest
warning signal from this check.

## Architecture Diagnosis

The current mechanism-prior graph is not useless, but its role is different from
what we originally hoped:

| Observation | Meaning |
|---|---|
| `base_only` equals `exported` on predicted-event RCA | Reconstruction evidence dominates the exported ranking |
| `mechanism_only` is weaker | Mechanism score is not yet a strong standalone RCA signal |
| z-score is competitive | HAI subsystem-level RCA can be solved well by simple deviation ranking |
| correlation prior is not stronger than z-score | Static normal correlation alone is not enough as a main contribution |
| faithfulness improved under mechanism-prior graph | The graph is more defensible as explanation evidence than as pure ranking booster |

## Decision

The current architecture is not ready to freeze for a strong CCF-A-style paper
claim if the main claim is "better RCA ranking." Full ablation should wait.

The defensible current claim is narrower:

```text
LaGraph provides competitive detection, useful predicted-event subsystem RCA,
and more faithful graph-based explanation than random/non-neighbor evidence.
```

The stronger paper claim still needs one of the following:

1. Variable/tag-level RCA labels where z-score is no longer trivially strong.
2. A mechanism-residual RCA score that improves over z-score and reconstruction.
3. A graph faithfulness/explanation-centered paper framing, where ranking SOTA is
   not the primary claim.

## Next Architecture Direction

The next architecture change should target mechanism evidence directly:

```text
normal mechanism model
-> expected channel behavior from graph neighbors
-> event-time mechanism residual
-> local pre-event contrast
-> RCA score = residual lift, not raw reconstruction magnitude
```

This is different from the current implementation, where reconstruction evidence
dominates and mechanism score is only a weak additive term.

Recommended next experiment:

```text
mechanism-residual RCA:
score_i = max(0, mechanism_error_i(event) - mechanism_error_i(pre_event))
```

Then compare against:

```text
z-score, correlation_prior, base_only, mechanism_only, base_plus_mechanism
```

The architecture should only be frozen if this residual mechanism score creates
a real gap over z-score/reconstruction, or if we explicitly shift the paper to a
faithful-explanation contribution.
