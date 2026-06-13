# SWaT/WADI 变量级 RCA 与 Attribution 论文对比

检索日期：2026-06-07  
关注：同时使用 SWaT/WADI，且和变量级 root cause localization、attack attribution、counterfactual explanation 相关的论文。  
排除主线：只做异常检测 F1 的论文不放入主对比，只作为背景。

## 1. 结论先行

不是只有 REASON 一篇。当前最有价值的同数据集对照至少包括：

1. **REASON, KDD 2023**：离线 root cause localization，SWaT/WADI 同时使用，报告 PR@K、MAP@K、MRR。
2. **CORAL, KDD 2023**：在线 root cause analysis，SWaT/WADI 同时使用，报告 PR@K、MAP@K、MRR。
3. **LEMMA-RCA, 2024/ICLR OpenReview**：把 SWaT/WADI 构造成 OT RCA benchmark，并系统评估 Dynotears、PC、C-LSTM、GOLEM、RCD、epsilon-Diagnosis、CIRCA、REASON、CORAL 等方法。
4. **NDSS 2024 Attribution**：不是 RCA ranking，但明确评估“被操纵 sensor/actuator 的 attribution”，同时使用 SWaT/WADI。
5. **NOMS 2025 FM-based Attack Attribution**：水系统 attack attribution，SWaT/WADI 同时使用，报告 Top-1/Top-5。
6. **AR-Pro, NeurIPS 2024**：反事实 anomaly repair/explanation，使用 SWaT/WADI/HAI，不是 RCA 数值对比。
7. **Causal Digital Twins, Machine Learning with Applications 2026**：SWaT/WADI/HAI，报告 root cause accuracy，但指标定义和我们的 RCA ranking 不同，属于弱可比。

最适合放进论文主数值比较的是前三个：**REASON、CORAL、LEMMA-RCA 中的 baseline 表**。

## 2. 严格 RCA 论文与 LaGraph 对比

注意：外部论文的 `PR@K` 不等于我们的 `Hit@K`。它们通常把 top-K 预测 root causes 中真实 root causes 的比例做归一化；我们的主表是 predicted-event variable RCA 的 Hit@K/MRR。因此最稳的是先比较 MRR，再把 top-K 当弱参考。

| 方法 | 来源 | 任务设置 | SWaT top-1/PR@1 | SWaT top-5/PR@5 | SWaT MRR | WADI top-1/PR@1 | WADI top-5/PR@5 | WADI MRR | 与 LaGraph 的关系 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| REASON | KDD 2023 | offline RCA | 0.250 | 0.667 | 0.410 | 0.286 | 0.650 | 0.534 | 最直接强对照 |
| CORAL | KDD 2023 | online RCA | 0.063 | 0.552 | 0.317 | 0.357 | 0.600 | 0.519 | 在线 RCA 强对照 |
| CIRCA | LEMMA-RCA 表 | offline RCA baseline | 0.188 | 0.250 | 0.287 | 0.143 | 0.550 | 0.350 | 可作为 causal inference baseline |
| RCD | LEMMA-RCA 表 | offline RCA baseline | 0.125 | 0.125 | 0.228 | 0.071 | 0.400 | 0.264 | 可作为 causal discovery baseline |
| C-LSTM | LEMMA-RCA 表 | Granger-style baseline | 0.125 | 0.281 | 0.294 | 0.000 | 0.350 | 0.244 | 可作为时序因果 baseline |
| PC | LEMMA-RCA 表 | causal graph baseline | 0.125 | 0.344 | 0.262 | 0.071 | 0.350 | 0.277 | 传统因果图 baseline |
| Dynotears | LEMMA-RCA 表 | dynamic DAG baseline | 0.125 | 0.323 | 0.279 | 0.071 | 0.300 | 0.222 | 动态因果 baseline |
| LaGraph 当前 | 本项目 | predicted-event variable RCA | 0.394 | 0.667 | 0.509 | 0.500 | 0.714 | 0.596 | 数值更高，但协议需统一后才能声称优于 |

初步判断：

- **MRR 维度**：LaGraph 当前 SWaT 0.5087、WADI 0.5960，高于 REASON 的 SWaT 0.4099、WADI 0.5335，也高于 CORAL 的 SWaT 0.317、WADI 0.519。
- **Top-1 维度**：LaGraph SWaT Hit@1 0.3939，高于 REASON PR@1 0.25；LaGraph WADI Hit@1 0.50，高于 REASON PR@1 0.286 和 CORAL PR@1 0.357。
- **Top-5 维度**：LaGraph SWaT Hit@5 0.6667 与 REASON PR@5 0.667 几乎相同；LaGraph WADI Hit@5 0.7143 高于 REASON/CORAL 的 0.65/0.60。
- **审稿风险**：PR@K 与 Hit@K 定义不同，predicted-event 协议也不同。正式论文中不能直接写“beats REASON”，除非复现实验或重新计算统一指标。

## 3. LEMMA-RCA 的意义

LEMMA-RCA 很重要，因为它把 SWaT/WADI 正式纳入 RCA benchmark，而不是只把它们当异常检测数据集。

它的 OT 子数据集统计：

| 子数据集 | 实体数量 | fault types | 平均时间戳 | 说明 |
|---|---:|---:|---:|---|
| SWaT | 51 | 16 | 56239.88 per fault | water treatment |
| WADI | 123 | 9 | 85248.47 per fault | water distribution |

它的核心实验价值：

- 证明 SWaT/WADI 可以被组织成 RCA benchmark。
- 提供多个 baseline：Dynotears、PC、C-LSTM、GOLEM、RCD、epsilon-Diagnosis、CIRCA、REASON、CORAL。
- 明确指出 SWaT/WADI 的 RCA 比 IT microservice 子数据集更难，因为 fault 短、间隔短，容易被 RCA 方法错过。

对我们最有用的论文写法：

> Following recent RCA benchmarks that construct OT RCA tasks from SWaT and WADI, we evaluate variable-level root-cause ranking using MRR and top-K localization metrics. Unlike benchmark-style offline RCA, our setting performs predicted-event RCA, requiring the model to localize root variables from detected events rather than oracle fault windows.

## 4. Attribution / Explanation 方向论文

这些论文和我们不是同一评价协议，但 related work 很重要。

| 论文 | Venue | 是否 SWaT/WADI | 指标 | 关键结果 | 与我们关系 |
|---|---|---:|---|---|---|
| Attributions for ML-based ICS Anomaly Detection | NDSS 2024 | 是 | Top-k attribution, AvgRank | SWaT/WADI 共 47 attacks、67 manipulations；raw-error Top-1 通常低于 40%；ensemble attribution 最好 | 证明 sensor/actuator attribution 很难，可作为 attribution baseline 思路 |
| FM-based Attack Attribution | NOMS 2025 | 是 | AvgRank, Top-1, Top-5 | SWaT all attacks Top-1 约 6-7%、Top-5 约 18-21%；SWaT multi-feature FM Top-1 25%、Top-5 35%；WADI single-feature Top-1 约 52-70%、Top-5 约 72-75% | 和我们都做 attack/root feature ranking，但它是 attribution 而非 event RCA |
| AR-Pro | NeurIPS 2024 | 是，另有 HAI | formal counterfactual explanation metrics | 使用 SWaT/WADI/HAI 做 anomaly repair，不报告 RCA MRR/PR@K | 可作为解释性异常检测相关工作，不适合主数值比较 |
| Causal Digital Twins | Machine Learning with Applications 2026 | 是，另有 HAI | root cause accuracy / attribution accuracy | 报告 78.4% root cause accuracy，跨数据集 attribution accuracy 约 66-75% | 有水系统因果诊断叙事，但指标和协议不透明，弱可比 |

## 5. 论文主表建议

如果我们要把论文写得硬，主表不应该只放 random/z-score/correlation。建议主表分三层：

### 表 A：同协议本地强基线

| Baseline 类型 | 方法 |
|---|---|
| Simple score | residual-only / z-score / forecast-error-only |
| Graph prior | correlation graph / random graph / graph-centrality + residual |
| Attribution | occlusion / integrated gradients / SHAP-style |
| RCA architecture | no source gate / no bottleneck / no event specificity / no propagation suppression |

### 表 B：外部 RCA 方法复现

优先复现：

1. REASON-style random-walk propagation over causal graph。
2. CORAL-style online graph update，如果工程量太大，可以先跑其 offline causal graph + propagation 近似。
3. CIRCA/RCD-style causal ranking。
4. PC/C-LSTM/Dynotears/GOLEM + propagation，这些在 REASON/LEMMA-RCA 中已经是标准 baseline。

### 表 C：跨论文背景对照

只用于 discussion，不作为 SOTA claim：

| 方法 | SWaT MRR | WADI MRR | 说明 |
|---|---:|---:|---|
| REASON | 0.410 | 0.534 | KDD 2023 offline RCA |
| CORAL | 0.317 | 0.519 | KDD 2023 online RCA |
| LEMMA-RCA best offline baseline | 0.410 | 0.534 | REASON |
| LaGraph | 0.509 | 0.596 | predicted-event RCA，协议不同 |

## 6. 当前 LaGraph 水平判断

只看 SWaT/WADI 变量级 RCA 相关文献，LaGraph 的位置比之前判断更清楚：

- **不是没有参照**：已经有 REASON、CORAL、LEMMA-RCA 这条 KDD/RCA benchmark 线。
- **MRR 数值很好**：LaGraph 在 SWaT/WADI 的 MRR 都高于 REASON 和 CORAL 报告值。
- **Top-5 不算压倒性**：SWaT Hit@5 与 REASON PR@5 基本持平，WADI 略高。
- **最大短板是协议未统一**：如果不复现外部方法，论文里只能说“compares favorably in magnitude”，不能说“outperforms state of the art”。
- **最值得补的实验**：把 REASON/LEMMA-RCA 的 baseline 协议迁移到我们的 predicted-event RCA 上，至少复现 PC、C-LSTM、Dynotears、CIRCA、REASON-style propagation 中的 2-3 个。

## 7. 推荐引用顺序

主结果/强相关：

1. REASON, KDD 2023: https://nijingchao.github.io/paper/kdd23_reason.pdf
2. CORAL, KDD 2023: https://doi.org/10.1145/3580305.3599392
3. LEMMA-RCA, 2024/ICLR OpenReview: https://openreview.net/pdf?id=0R8JUzjSdq

Attribution / explanation 相关：

4. NDSS 2024 Attribution: https://www.ndss-symposium.org/ndss-paper/attributions-for-ml-based-ics-anomaly-detection-from-theory-to-practice/
5. FM-based Attack Attribution: https://arxiv.org/abs/2503.01229
6. AR-Pro, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d3591b6746204b332acb464b775d38d-Abstract-Conference.html
7. Causal Digital Twins, MLWA 2026: https://www.sciencedirect.com/science/article/pii/S2666827025002075

