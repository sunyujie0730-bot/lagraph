# Markdown Staleness Audit - 2026-06-17

This audit lists Markdown files that may no longer match the current LaGraph
code and paper direction.

Execution status: the obvious obsolete files were deleted on 2026-06-17, and
historically useful architecture/provenance files were moved to
`D:\la_v12\docs\archive\2026-06-17\`. See
`D:\la_v12\docs\DOCS_CLEANUP_LOG_2026_06_17.md` for the executed cleanup log.

## Current Reference Point

The current codebase has moved beyond the old `full`, `mechanism-prior-graph`,
and HAI-first narratives. The active research line is now around
source-bottleneck RCA, evidence fusion, source/response separation, and
WADI/SWaT predicted-event variable-level RCA. Recent experimental profiles in
`ts_benchmark/run_single.py` include:

- `source-bottleneck-specificity-rca`
- `source-bottleneck-root-response-*`
- `source-bottleneck-evidence-fusion-*`
- `source-bottleneck-consistency-rca`

Therefore documents that present `full`, `mechanism-prior-graph`, v10
SparseLaGraph, HAI-only RCA, or MSL/SWaT detection as the current main story are
stale.

## Delete Or Archive Candidates

These files are likely misleading if kept as active documentation.

| File | Last updated | Why stale | Suggested action |
|---|---:|---|---|
| `D:\la_v12\PROJECT_UNDERSTANDING.md` | 2026-05-22 | Describes LaGraph v10 SparseLaGraph, old `/home/professor3/...` path, old model structure, and legacy scoring. | Archive or delete after confirming no unique setup notes are needed. |
| `D:\la_v12\.analysis_summary.md` | 2026-05-15 | Early raw-F diagnosis and bug list; useful historically but not current architecture. | Delete or move to `docs/archive/2025-05-debug/`. |
| `D:\la_v12\docs\REFACTOR_PLAN.md` | 2026-05-13 | v10 refactor plan; says several modules were removed and DDP retained. This conflicts with later single-GPU and RCA-focused development. | Archive. |
| `D:\la_v12\docs\HYPERPARAMETER_ANALYSIS.md` | 2026-05-14 | v10/v11 hyperparameter report; old dropout/VQ/cooldown/batch reasoning. | Archive. |
| `D:\la_v12\docs\update.md` | 2026-05-20 | A task prompt for v11.3 P1-FIXED, not maintained documentation. Contains instructions that may no longer match code. | Delete or archive as prompt history. |
| `D:\la_v12\docs\CHANGELOG.md` | 2026-05-14 | Only records early v11 changes; not maintained after major RCA architecture iterations. | Archive or replace with a concise current changelog. |
| `D:\la_v12\docs\VQ_TUNING_LOG.md` | 2026-05-23 | Focused on VQ tuning for MSL/SWaT detection, not the current RCA story. | Archive. |
| `D:\la_v12\docs\STATE_AWARE_RESEARCH_PLAN.md` | 2026-05-25 | State-aware branch was a candidate, not the current main model. | Archive unless revived. |
| `D:\la_v12\docs\RCA_EXPERIMENT_PLAN.md` | 2026-05-26 | HAI-centric RCA plan with old score formula and old recommended parameters. | Archive after extracting any generic metric definitions. |
| `D:\la_v12\docs\GROUP_MEETING_REPORT_2026_05_27.md` | 2026-05-27 | Presents `mechanism-prior-graph` as current main line; no longer current. | Archive as meeting-history material. |
| `D:\la_v12\docs\PAPER_VALIDITY_CHECK_2026_05_27.md` | 2026-05-27 | HAI RCA baseline check; useful historical evidence but not current WADI/SWaT variable-level RCA. | Archive. |
| `D:\la_v12\docs\SWAT_A6_SMOKE_TEST_2026_05_28.md` | 2026-05-28 | Dataset smoke test for SWaT A6; not part of current main benchmark. | Archive, not delete if raw-data provenance matters. |
| `D:\la_v12\docs\WADI_A2_SMOKE_TEST_2026_05_28.md` | 2026-05-28 | Dataset smoke test for WADI A2; current main WADI file is A1 ds10. | Archive, not delete if raw-data provenance matters. |
| `D:\la_v12\docs\WEEKLY_PROGRESS_2026-05-25.md` | 2026-05-25 | Weekly status before later RCA pivot. | Archive as meeting-history material. |

## Replace Rather Than Delete

These files are important, but their current wording can mislead future paper
writing.

| File | Problem | Suggested action |
|---|---|---|
| `D:\la_v12\docs\ARCHITECTURE.md` | Very large mixed-history document. It still says the default reporting profile is `full`, contains old MSL/HAI commands, and mixes rejected profiles with current ideas. | Rewrite into a short current architecture document, then archive this as `ARCHITECTURE_HISTORY.md`. |
| `D:\la_v12\docs\ARCHITECTURE_SIMPLIFICATION.md` | Useful history of removing dead modules, but the profile table is old and not aligned with source-bottleneck/evidence-fusion RCA. | Archive after moving the useful "why modules were removed" part into the new architecture doc. |
| `D:\la_v12\result\analysis\architecture_experiment_summary.md` | Useful chronological record but too large and mixed; early sections are stale and later sections are hard to find. | Split into `history` and `current_findings`, or keep as raw lab notebook only. |

## Keep But Update

These are still aligned with the paper direction, but likely need updated
numbers or terminology after the latest experiments.

| File | Why keep | Needed update |
|---|---|---|
| `D:\la_v12\docs\RCA_EVALUATION_PROTOCOL.md` | Defines the RCA protocol and the predicted-event evaluation flow. | Update dataset roles: HAI should be supplemental/coarse, WADI/SWaT should be main variable-level RCA; confirm current main profile. |
| `D:\la_v12\docs\MANUSCRIPT_DRAFT_ZH_2026_06_07.md` | Closest to the current paper narrative: predicted-event source-variable ranking. | Refresh method name, current profile, and final metrics. |
| `D:\la_v12\docs\NATURE_MANUSCRIPT_FRAMEWORK_2026_06_07.md` | Useful high-level paper framing. | Update from early `source-bottleneck-specificity` to latest validated profile if adopted. |
| `D:\la_v12\docs\VENUE_AND_RESULT_POSITIONING_2026_06_07.md` | Useful result-positioning document. | Refresh numbers and remove stale claims after latest WADI/SWaT runs. |
| `D:\la_v12\docs\VARIABLE_LEVEL_RCA_LITERATURE_POSITIONING_2026_06_07.md` | Useful literature positioning for variable-level RCA. | Update if new comparison papers or MSDS data are added. |
| `D:\la_v12\docs\SWAT_WADI_RCA_PAPERS_COMPARISON_2026_06_07.md` | Useful external comparison summary. | Keep; refresh only if reproducing external baselines. |
| `D:\la_v12\docs\PAPER_REVIEW_RISK_AND_SCOPE.md` | Still useful as a reviewer-risk filter. | Move into current paper-planning docs or condense. |

## Result-Analysis Markdown Policy

Most Markdown files under `D:\la_v12\result\analysis\` are experiment records,
not maintained architecture documents. They should not be used as current
system descriptions unless explicitly referenced by a current summary.

Recommended policy:

- Keep latest consolidated summaries:
  - `D:\la_v12\result\analysis\optimization_summary_20260617.md`
  - `D:\la_v12\result\analysis\journal_validation\journal_validation_summary.md`
  - `D:\la_v12\result\analysis\journal_validation\journal_validation_interpretation.md`
- Treat older failure analyses and queue reports as raw lab notebooks.
- Do not cite old `result/analysis/*.md` files in the paper unless the
  associated CSV metrics are still present and the profile is still relevant.

## Recommended Cleanup Order

1. Create `D:\la_v12\docs\archive\`.
2. Move the "Delete Or Archive Candidates" there first instead of deleting.
3. Rewrite `docs/ARCHITECTURE.md` as the single source of truth.
4. Rename the old large `ARCHITECTURE.md` to `ARCHITECTURE_HISTORY.md`.
5. Keep current paper/literature/protocol docs, then refresh their metrics.
