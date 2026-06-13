# 变量级 RCA 文献与 LaGraph 结果定位

检索日期：2026-06-07  
关注范围：变量级 root cause localization / anomaly diagnosis / dimension attribution  
不关注：异常检测 F1、AUROC、point adjustment 等检测指标

## 1. 当前结论

如果只看变量级 RCA，LaGraph 现在的结果是有竞争力的，尤其是在 SWaT/WADI 这种工业控制公共数据集上。但这里必须区分“严格可比”和“背景标尺”：不同数据集、不同 root-cause 标签粒度、不同 event 定义、不同 top-k 指标不能用于直接宣称超过某个方法，只能用于判断结果大概处在什么量级。

当前主线 `source-bottleneck-specificity-rca`：

| 数据集 | 协议 | MRR | Hit@1 | Hit@3 | Hit@5 | 初步定位 |
|---|---|---:|---:|---:|---:|---|
| WADI | predicted-event variable RCA | 0.5960 | 0.5000 | 0.7143 | 0.7143 | 同协议内强；跨论文只能作为量级参考 |
| SWaT | predicted-event variable RCA | 0.5087 | 0.3939 | 0.5758 | 0.6667 | 同协议内中强；与 AERCA 同数据集但协议不同，不能直接宣称超越 |

谨慎结论：现在可以说“变量级 RCA 达到可发表的中高水平”，不能说“全面超过 SOTA”。原因是不同论文的 RCA 协议差异很大：有的按 timestamp，有的按 event/case，有的 ground-truth root variables 很多，有的只在私有 AIOps 数据集或 SMD/MSDS 上评估。

## 1.1 比较证据等级

| 等级 | 条件 | 能支持什么结论 | 本项目当前例子 |
|---|---|---|---|
| A. 严格可比 | 同数据集、同 split、同 predicted/true-event 协议、同 root labels、同指标 | 可以说方法 A 优于方法 B | LaGraph vs z-score/random/correlation/soft-hierarchical 等本地同协议基线 |
| B. 弱可比 | 同数据集，但 event 定义、root labels、top-k 指标或采样方式不同 | 只能说数值处于相近/更高/更低区间，不能宣称 SOTA 超越 | LaGraph SWaT vs AERCA SWaT |
| C. 背景标尺 | 不同数据集或不同系统域 | 只能说明 RCA 文献中常见 top-k 水平和写法 | TranAD/MTAD-GAT/CIRCA/CausalRCA 等 |

后续论文里最稳的主张应该来自 A 级证据。B/C 级文献主要服务于 related work 和 discussion，而不是主结果表里的“beat SOTA”。

## 2. 最值得对照的变量级 RCA 论文

### 2.1 MTAD-GAT, ICDM 2020

论文：Multivariate Time-Series Anomaly Detection via Graph Attention Network  
链接：https://arxiv.org/abs/2009.02040

这篇是早期把“异常检测 + 变量级诊断”放在一起写得比较清楚的工作。它把每个 feature 的 inference score 作为 root cause score，并在私有 TSA/Flink 数据集上做 anomaly diagnosis。作者人工标注了 700+ clear-root-cause instances。

关键 RCA 结果：

| 数据集 | 指标 | 结果 |
|---|---|---:|
| TSA 私有数据 | NDCG@5 | 0.8556 |
| TSA 私有数据 | HitRate@100% | 0.7428 |
| TSA 私有数据 | HitRate@150% | 0.8561 |

对 LaGraph 的意义：

- MTAD-GAT 的 RCA 数值很高，但它用的是私有 TSA 数据，不是 SWaT/WADI。
- 它提供了一个很好的写法：先给 feature-wise score，再给 HitRate/NDCG，再配 case study。
- 不能直接说我们的 WADI Hit@5 0.7143 低于它，因为 HitRate@100% 的 top-k 取决于 ground-truth root variables 数量，不等价于 Hit@5。

### 2.2 TranAD, PVLDB 2022

论文：TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data  
链接：https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf

TranAD 的 RCA 部分叫 anomaly diagnosis，使用 HitRate@P% 和 NDCG@P%。它只在 SMD 和 MSDS 上展示诊断表，未给 SWaT/WADI 的变量级 RCA 表。

关键 RCA 结果：

| 数据集 | 方法 | H@100% | H@150% | N@100% | N@150% |
|---|---|---:|---:|---:|---:|
| SMD | TranAD | 0.4981 | 0.6401 | 0.4941 | 0.6178 |
| MSDS | TranAD | 0.4630 | 0.7533 | 0.5981 | 0.6963 |

同表强基线范围：

- SMD 上，DAGMM/USAD/CAE-M/TranAD 的 H@100 大致在 0.47-0.50，TiTAD 后续报告的 SMD H@100 为 0.5328。
- MSDS 上，MTAD-GAT H@100 为 0.5812，TranAD H@150 为 0.7533。

对 LaGraph 的意义：

- 我们 WADI Hit@1 0.5000 与 TranAD SMD H@100 0.4981 数值接近，但这只是背景标尺，因为数据集和指标定义不同。
- 我们 SWaT Hit@1 0.3939 低于 TranAD SMD/MSDS H@100，但不能解释为方法弱于 TranAD，因为 SWaT 的工业过程传播和 root label 机制不同。
- 这篇论文只能说明：在公开维度诊断表里，0.45-0.55 的 top-level hit/recall 是常见强模型区间；它不能作为直接 superiority baseline。

### 2.3 TiTAD, Electronics 2025

论文：TiTAD: Time-Invariant Transformer for Multivariate Time Series Anomaly Detection  
链接：https://www.mdpi.com/2079-9292/14/7/1401

TiTAD 不是最高级别 venue，但它的好处是复用了 TranAD 风格的 anomaly diagnosis 表，并把多个常见检测模型作为诊断基线。

关键 RCA 结果：

| 数据集 | 方法 | H@100% | H@150% | N@100% | N@150% |
|---|---|---:|---:|---:|---:|
| SMD | TiTAD | 0.5328 | 0.6460 | 0.5703 | 0.6379 |

对 LaGraph 的意义：

- 它给了一个较新的变量级诊断参照：SMD H@100 约 0.53。
- 我们 WADI Hit@1 0.50 在数值量级上接近这个级别；SWaT Hit@1 0.394 稍弱但 Hit@5 0.667 可用。由于数据集不同，这只能作为背景量级参考。

### 2.4 AERCA, ICLR 2025 Oral

论文：Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery  
链接：https://openreview.net/forum?id=JHC8jdmCrk  
PDF：https://openreview.net/pdf?id=k38Th3x4d9

这是最重要的直接对照之一。它不是把 RCA 当附属解释，而是明确做 multivariate time series RCA。AERCA 把异常定义为 exogenous variables 上的 intervention，通过 Granger causal discovery 和 exogenous residual score 做 root cause localization。

它使用 AC@K，也就是 top-k root cause recall。真实数据包含 SWaT 和 MSDS。

关键真实数据 RCA 结果：

| 数据集 | 方法 | AC@1 | AC@3 | AC@5 | AC@10 | Avg@10 |
|---|---|---:|---:|---:|---:|---:|
| SWaT | epsilon-Diagnosis | 0.075 | 0.125 | 0.125 | 0.375 | 0.180 |
| SWaT | RCD | 0.000 | 0.000 | 0.000 | 0.300 | 0.100 |
| SWaT | CIRCA | 0.000 | 0.000 | 0.000 | 0.300 | 0.100 |
| SWaT | AERCA | 0.220 | 0.290 | 0.330 | 0.455 | 0.342 |
| MSDS | CIRCA | 0.454 | 0.860 | 0.917 | 1.000 | 0.809 |
| MSDS | AERCA | 0.381 | 0.908 | 0.974 | 1.000 | 0.896 |

对 LaGraph 的意义：

- AERCA 是 ICLR 2025 Oral，且直接报告 SWaT 变量级 RCA。它在 SWaT 上 AC@1 只有 0.220、AC@5 只有 0.330，作者也说明 SWaT 上所有方法都会明显下降。
- 我们 SWaT Hit@1 0.3939、Hit@3 0.5758、Hit@5 0.6667，数值上高于 AERCA 的 SWaT AC@1/3/5，但这仍然不是严格可比结果。
- 但不能直接宣称超过 AERCA，因为 AERCA 的 SWaT 设置不同：它按 sequences 评估，平均 root variables 为 13.35，并且使用自己的 downsampling 和 AC@K 定义；我们的协议是 predicted-event RCA。
- 这篇最适合作为引言和讨论里的证据：SWaT 变量级 RCA 本身很难，强 causal 方法在 SWaT 上也只有中等 top-k recall。若要把它变成主结果对比，需要复现 AERCA 协议或把 AERCA/RCD/CIRCA 跑到我们的 predicted-event 标签协议下。

### 2.5 CIRCA, KDD 2022 Applied Data Science Track

论文：Causal Inference-Based Root Cause Analysis for Online Service Systems with Intervention Recognition  
链接：https://arxiv.org/abs/2206.05871  
PDF：https://netman.aiops.org/wp-content/uploads/2022/08/KDD22-CIRCA.pdf

CIRCA 是 online service systems 的 metric-level RCA，不是工业控制过程，但它的评价方式和我们的变量级 RCA 很接近：给定故障 case，返回 top-k root cause metrics。

真实 Oracle 数据集结果：

| 方法 | AC@1 | AC@5 | Avg@5 |
|---|---:|---:|---:|
| NSigma | 0.323 | 0.662 | 0.525 |
| DFS-MH | 0.268 | 0.439 | 0.372 |
| CIRCA | 0.404 | 0.763 | 0.603 |
| Ideal | 0.929 | 1.000 | 0.986 |

对 LaGraph 的意义：

- 我们 SWaT Hit@1 0.3939 和 CIRCA real-world AC@1 0.404 基本同一量级。
- 我们 WADI Hit@1 0.5000 数值高于 CIRCA real-world AC@1，但数据集和系统拓扑完全不同，不能写成直接超过。
- 我们 WADI Hit@5 0.7143、SWaT Hit@5 0.6667 略低于 CIRCA AC@5 0.763，但这同样只能作为量级参考。
- 这说明我们的 top-1 水平进入了 KDD applied RCA 论文常见的实用区间；top-5 仍有提升空间。

### 2.6 RCD, NeurIPS 2022

论文：Root Cause Analysis of Failures in Microservices through Causal Discovery  
链接：https://proceedings.neurips.cc/paper_files/paper/2022/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html  
PDF：https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf

RCD 是微服务场景的 causal-discovery RCA，核心思想是把 failure 当作 root cause 上的 intervention，只学习和 root cause 相关的局部 causal graph，以降低全图因果发现成本。

对 LaGraph 的意义：

- 它是方法论上很重要的引用：局部因果发现、top-k recall、intervention-root-cause 叙事。
- 但它的系统是 microservices，不是传感器级工业过程；可作为相关工作，不应作为唯一强数值对照。
- AERCA 把 RCD 作为 SWaT/MSDS 的 RCA baseline，因此在我们的论文里可通过 AERCA 的表来间接说明 RCD 在 SWaT 上并不强。

### 2.7 CausalRCA, Journal of Systems and Software 2023

论文：CausalRCA: Causal Inference based Precise Fine-grained Root Cause Localization for Microservice Applications  
链接：https://arxiv.org/abs/2209.02500

这篇强调 fine-grained root cause localization，即不仅定位故障服务，还定位故障服务中的具体指标。

关键结果：

| 任务 | 指标 | 结果 |
|---|---|---:|
| faulty service 内 fine-grained root cause metric localization | average AC@3 | 0.719 |
| overall performance | Avg@5 improvement | +9.43% |

对 LaGraph 的意义：

- 我们 WADI Hit@3 0.7143 基本达到 CausalRCA 报告的 fine-grained AC@3 级别。
- 我们 SWaT Hit@3 0.5758 低于这个水平，但仍明显高于 AERCA 在 SWaT 上的 AC@3。
- CausalRCA 是微服务，不是 ICS；适合作为“fine-grained metric RCA 的 top-k recall 水平”参照。

## 3. 和 LaGraph 的横向水平判断

本节不是严格排行榜，只是量级校准。严格主表仍应只放同数据集、同协议、同指标的 baseline。

### 3.1 只看 top-1 的量级校准

| 论文/方法 | 数据集/场景 | Top-1 类指标 | 与 LaGraph 的关系 |
|---|---|---:|---|
| TranAD | SMD | H@100 0.4981 | 接近 WADI Hit@1 0.5000，高于 SWaT 0.3939 |
| TranAD | MSDS | H@100 0.4630 | 低于 WADI，高于 SWaT |
| TiTAD | SMD | H@100 0.5328 | 略高于 WADI，高于 SWaT |
| AERCA | SWaT | AC@1 0.220 | 低于 LaGraph SWaT 0.3939 |
| AERCA | MSDS | AC@1 0.381 | 低于 WADI，接近 SWaT |
| CIRCA | Oracle real-world | AC@1 0.404 | 低于 WADI，接近 SWaT |
| LaGraph | WADI | Hit@1 0.5000 | 中高水平 |
| LaGraph | SWaT | Hit@1 0.3939 | 中等偏强 |

判断：LaGraph 的 WADI top-1 已进入公开 RCA 论文常见强结果区间；SWaT top-1 不是特别高，但数值上高于 AERCA 的 SWaT 报告值，且接近 CIRCA 的真实系统 AC@1。由于协议不同，这句话只能放在 discussion，不应写成“超过 AERCA/CIRCA”。

### 3.2 只看 top-3/top-5 的量级校准

| 论文/方法 | 数据集/场景 | Top-k 类指标 | 与 LaGraph 的关系 |
|---|---|---:|---|
| AERCA | SWaT | AC@3 0.290, AC@5 0.330 | 明显低于 LaGraph SWaT Hit@3 0.5758 / Hit@5 0.6667 |
| AERCA | MSDS | AC@3 0.908, AC@5 0.974 | 明显高于 LaGraph，但 MSDS 更容易/协议不同 |
| CIRCA | Oracle real-world | AC@5 0.763 | 高于 LaGraph WADI/SWaT Hit@5 |
| CausalRCA | microservice metric | AC@3 0.719 | 接近 LaGraph WADI Hit@3，高于 SWaT Hit@3 |
| LaGraph | WADI | Hit@3 0.7143, Hit@5 0.7143 | 强 |
| LaGraph | SWaT | Hit@3 0.5758, Hit@5 0.6667 | 中强 |

判断：WADI top-3 已经很强；SWaT top-3/top-5 可以支撑论文，但如果要冲更高，需要在同协议下加入更强 RCA baseline，或者提升到 Hit@3 0.65+ 并提供更强解释案例。

## 4. 我们应该如何在论文中表述

建议表述：

> Existing anomaly diagnosis work often reports variable-level HitRate or AC@K on SMD/MSDS or private AIOps datasets. In contrast, variable-level RCA on SWaT remains difficult: a recent ICLR 2025 causal RCA method reports AC@1/3/5 of 0.220/0.290/0.330 on SWaT. Under our predicted-event RCA protocol, LaGraph achieves Hit@1/3/5 of 0.394/0.576/0.667 on SWaT and 0.500/0.714/0.714 on WADI, suggesting that source-bottleneck specificity provides competitive root-variable ranking on industrial control systems.

中文版本：

> 既有 anomaly diagnosis 工作多在 SMD/MSDS 或私有 AIOps 数据集上报告变量级 HitRate/AC@K。相比之下，SWaT 上的变量级 RCA 明显更困难：ICLR 2025 的 AERCA 在 SWaT 上报告的 AC@1/3/5 为 0.220/0.290/0.330。本文在 predicted-event RCA 协议下，LaGraph 在 SWaT 上达到 Hit@1/3/5 = 0.394/0.576/0.667，在 WADI 上达到 0.500/0.714/0.714，说明 source-bottleneck specificity 对工业控制系统中的根因变量排序具有较强竞争力。

## 5. 还缺什么才能更稳

1. 统一指标：正式稿建议额外报告 AC@1/3/5、HitRate@100%/150%、NDCG@K，便于和 AERCA/TranAD/MTAD-GAT 连接。
2. 直接 RCA baseline：优先跑 AERCA-style exogenous residual / Granger baseline，至少在 SWaT 上跑一个可复现实验。
3. 诊断型 baseline：加入 residual-only、forecast-error-only、reconstruction-error-only、integrated gradients/occlusion attribution。
4. 事件协议说明：把 predicted-event RCA 和 true-event RCA 区分清楚；外部论文多数不是 predicted-event event-level RCA。
5. 统计稳定性：3-5 seed，给 Hit@1/MRR 的 bootstrap CI。
6. 案例图：至少展示一个 SWaT 和一个 WADI 的变量排名案例，标出 top-5、真实 root variable、propagation variables。

## 6. 推荐引用清单

核心必引：

1. MTAD-GAT, ICDM 2020：https://arxiv.org/abs/2009.02040
2. TranAD, PVLDB 2022：https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf
3. AERCA, ICLR 2025 Oral：https://openreview.net/forum?id=JHC8jdmCrk
4. CIRCA, KDD 2022：https://arxiv.org/abs/2206.05871
5. RCD, NeurIPS 2022：https://proceedings.neurips.cc/paper_files/paper/2022/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html
6. CausalRCA, JSS 2023：https://arxiv.org/abs/2209.02500

可补充：

1. TiTAD, Electronics 2025：https://www.mdpi.com/2079-9292/14/7/1401
2. PyRCA library：https://arxiv.org/abs/2306.11477
3. Causal structure-based root cause analysis of outliers, ICML 2022：https://proceedings.mlr.press/v162/budhathoki22a.html
4. Root Cause Identification for Collective Anomalies in Time Series, AISTATS 2023：https://proceedings.mlr.press/v206/assaad23a.html

## 7. 最终定位

变量级 RCA 维度上，我会把当前 LaGraph 定为：

- WADI：`B+`，已经有较强论文价值。
- SWaT：`B` 到 `B+`，比 AERCA 的 SWaT 表强，但需要统一协议或直接复现基线来让说法更稳。
- 跨数据集总体：`B+` 的应用型 RCA 结果，适合 EAAI/ESWA/KBS/Neurocomputing 这类期刊；如果补上 AERCA/CIRCA/RCD-style baseline 和稳定性，有机会冲更强的工业智能/数据挖掘 venue。

最重要的策略不是继续强调检测，而是把论文主表改成：

1. Variable-level predicted-event RCA main table。
2. Public RCA literature comparison table。
3. Direct baseline table。
4. Ablation table。
5. Case-study figure。
