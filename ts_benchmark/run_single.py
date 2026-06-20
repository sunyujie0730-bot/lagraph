# -*- coding: utf-8 -*-
"""
LaGraph 单模型简化运行入口
============================
只跑 LaGraph，不需要基线对比，不需要复杂的 CLI。

用法:
  python ts_benchmark/run_single.py                 # 默认 10 epochs，全量数据集
  python ts_benchmark/run_single.py --fast          # 快速测试：3 epochs
  python ts_benchmark/run_single.py --epochs 30 --n-gpus 8   # 自定义训练轮次
  python ts_benchmark/run_single.py --epochs 30 --num-workers 2 --prefetch-factor 2

说明:
  - 始终运行 DETECT_META.csv 中所有可用数据集
  - --fast / --epochs 只控制训练轮次，不影响数据集选择
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings

import pandas as pd
import torch

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ts_benchmark.common.constant import CONFIG_PATH
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend

warnings.filterwarnings("ignore")


def _diagnose_cuda():
    """诊断 CUDA 是否真的可用"""
    import torch
    available = torch.cuda.is_available()
    print(f"  CUDA available       : {available}")
    if available:
        print(f"  CUDA version         : {torch.version.cuda}")
        print(f"  GPU count            : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU[{i}]              : {torch.cuda.get_device_name(i)}")
        print(f"  CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    else:
        print("  WARNING: torch.cuda.is_available() == False")
        print(f"     torch version     : {torch.__version__}")
    print()
    return available


EVAL_CONFIG = "all_detect_label_config.json"
AVAILABLE_EVAL_CONFIGS = [
    "all_detect_label_config.json",
    "all_detect_score_config.json",
    "fixed_detect_label_config.json",
    "fixed_detect_score_config.json",
    "unfixed_detect_label_config.json",
    "unfixed_detect_score_config.json",
]
DEFAULT_SAVE_DIR = "label/LaGraph"
FAST_EPOCHS = 3
DEFAULT_EPOCHS = 10


def main():
    parser = argparse.ArgumentParser(description="LaGraph 单模型运行")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"训练轮次 (default: {DEFAULT_EPOCHS}, --fast: {FAST_EPOCHS})",
    )
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help=(
            "Use the paper main protocol when --epochs is not set: "
            "max epochs=15 and validation-best checkpoint selection."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"快速测试模式 ({FAST_EPOCHS} epochs)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"结果保存子目录 (default: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--eval-config",
        choices=AVAILABLE_EVAL_CONFIGS,
        default=EVAL_CONFIG,
        help=(
            "Evaluation config. Use unfixed_detect_label_config.json for datasets "
            "with metadata train_lens, such as TE_MM_* and HAI_*."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="Random seed for evaluation strategy and model initialization (default: 2021)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="指定要跑的数据集文件名 (如 swat.csv HAI_21_03_test1.csv)，不指定则跑全部",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="GPU 数量开关 (default: 1=单卡, 0=CPU; >1 会被降级为单卡，DDP 已停用)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count (default: 2; use 0 if Windows shared-memory errors occur)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="每个 DataLoader worker 预取 batch 数 (default: 2; num_workers=0 时自动忽略)",
    )
    parser.add_argument(
        "--inference-num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker count for post-training scoring/RCA export "
            "(default: 0 for Windows stability on large SWaT/WADI runs)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size for sensitivity tests (default: fixed profile value 256)",
    )
    parser.add_argument(
        "--robust-input-preprocess",
        action="store_true",
        help="Clip each channel by normal-training quantiles before StandardScaler.",
    )
    parser.add_argument(
        "--input-clip-lower-quantile",
        type=float,
        default=None,
        help="Lower quantile for --robust-input-preprocess, e.g. 0.001.",
    )
    parser.add_argument(
        "--input-clip-upper-quantile",
        type=float,
        default=None,
        help="Upper quantile for --robust-input-preprocess, e.g. 0.999.",
    )
    parser.add_argument(
        "--arch-profile",
        choices=[
            "full",
            "with-scorer",
            "core",
            "no-vq",
            "graph-only",
            "prior-guided-graph",
            "structure-consistent",
            "structure-biased",
            "mechanism-graph",
            "mechanism-prior-graph",
            "mechanism-coupled",
            "mechanism-coupled-lite",
            "mechanism-coupled-noscore",
            "mechanism-predictive",
            "mechanism-predictive-rca",
            "mechanism-predictive-robust-rca",
            "source-propagation-rca",
            "normal-mechanism-source",
            "graph-coupled-source",
            "gated-dual-source",
            "causal-gated-source",
            "causal-source-rca",
            "causal-source-robust-rca",
            "source-gated-causal-rca",
            "source-effect-rca",
            "mechanism-masked-rca",
            "mechanism-root-rca",
            "mechanism-root-sharp-rca",
            "onset-source-sharp-rca",
            "adaptive-source-sharp-rca",
            "hierarchical-source-rca",
            "soft-hierarchical-rca",
            "mechanism-guided-source-rca",
            "soft-hierarchical-no-normal-prior",
            "mechanism-feedback-rca",
            "masked-responsibility-rca",
            "interventional-graph-source-rca",
            "source-propagation-trained-rca",
            "source-propagation-prior-rca",
            "source-innovation-rca",
            "temporal-innovation-rca",
            "source-bottleneck-rca",
            "source-bottleneck-balanced-rca",
            "source-bottleneck-onset-balanced-rca",
            "source-bottleneck-onset-wide-rca",
            "source-bottleneck-aligned-onset-rca",
            "source-bottleneck-aligned-wide-rca",
            "source-bottleneck-contrast-balanced-rca",
            "source-bottleneck-specificity-rca",
            "source-bottleneck-topk-rerank-rca",
            "source-bottleneck-root-score-rca",
            "source-bottleneck-root-score-w03-rca",
            "source-bottleneck-root-response-rca",
            "source-bottleneck-root-response-hardpos-rca",
            "source-bottleneck-root-response-softpos-rca",
            "source-bottleneck-root-response-top20-calibrated-rca",
            "source-bottleneck-root-response-innovation-top20-rca",
            "source-bottleneck-pairwise-root-response-rca",
            "source-bottleneck-pairwise-aux-root-response-rca",
            "source-bottleneck-event-responsibility-top20-rca",
            "source-bottleneck-event-responsibility-head-top20-rca",
            "source-bottleneck-event-evidence-top20-rerank-rca",
            "source-bottleneck-event-balanced-evidence-top20-rerank-rca",
            "source-bottleneck-event-swat-priority-evidence-top20-rerank-rca",
            "source-bottleneck-event-top1-coverage-evidence-top20-rerank-rca",
            "source-bottleneck-evidence-fusion-head-top1-coverage-rca",
            "source-bottleneck-evidence-fusion-head-detached-rca",
            "source-bottleneck-evidence-fusion-secondary-fill-rca",
            "source-bottleneck-evidence-fusion-stable-fill-rca",
            "source-bottleneck-stable-score-responsibility-rca",
            "source-bottleneck-stable-score-responsibility-rerank-rca",
            "source-bottleneck-stable-graph-evidence-only-rca",
            "source-bottleneck-stable-graph-evidence-reliable-score-rca",
            "source-bottleneck-stable-graph-evidence-source-gate-guard-rca",
            "source-bottleneck-stable-graph-evidence-onset-source-gate-guard-rca",
            "source-bottleneck-stable-graph-evidence-onset-input-source-gate-guard-rca",
            "source-bottleneck-stable-graph-evidence-effect-margin-guard-rca",
            "source-bottleneck-stable-graph-evidence-event-route-guard-rca",
            "source-bottleneck-stable-graph-evidence-event-route-response-guard-rca",
            "source-bottleneck-stable-graph-evidence-event-route-response-root-rerank-rca",
            "source-bottleneck-causal-innovation-response-rca",
            "source-propagation-strict-cross-mechanism-rca",
            "source-propagation-residual-sps-rca",
            "source-propagation-bounded-sps-fusion-rca",
            "source-bottleneck-dual-expert-representation-fusion-rca",
            "source-bottleneck-source-expert-representation-rca",
            "source-bottleneck-stable-graph-evidence-learned-responsibility-rerank-rca",
            "source-bottleneck-stable-graph-evidence-adaptive-learned-decoder-rca",
            "source-bottleneck-directed-sparse-graph-rca",
            "source-bottleneck-directed-graph-evidence-only-rca",
            "source-bottleneck-directed-graph-supervised-rca",
            "source-bottleneck-directed-graph-calibrated-rca",
            "source-bottleneck-directed-graph-signed-calibrated-rca",
            "source-bottleneck-prior-supported-transfer-rca",
            "source-bottleneck-response-only-graph-rca",
            "source-bottleneck-response-only-multilag-graph-rca",
            "source-bottleneck-response-aligned-multilag-graph-rca",
            "source-bottleneck-response-graph-fusion-rca",
            "source-bottleneck-response-graph-fusion-detached-rca",
            "source-bottleneck-response-suppressor-rca",
            "source-bottleneck-response-suppressor-score-responsibility-rca",
            "source-bottleneck-response-suppressor-responsibility-rca",
            "source-bottleneck-stable-no-rca-graph-score-rca",
            "source-bottleneck-stable-no-encoder-channel-graph-rca",
            "source-bottleneck-stable-no-encoder-channel-graph-score-responsibility-rca",
            "source-bottleneck-stable-no-encoder-channel-graph-score-responsibility-rerank-rca",
            "source-bottleneck-stable-no-mechanism-training-rca",
            "source-bottleneck-evidence-fusion-stable-fill-aligned-rca",
            "source-bottleneck-evidence-fusion-stable-pairwise-rca",
            "source-bottleneck-evidence-fusion-strong-pairwise-rca",
            "source-bottleneck-evidence-fusion-end2end-pairwise-rca",
            "source-bottleneck-consistency-rca",
            "source-bottleneck-consistency-head-rca",
            "source-bottleneck-consistency-head-e2e-trainonly-rca",
            "source-bottleneck-source-interaction-head-rca",
            "source-bottleneck-source-interaction-head-direct-rca",
            "source-bottleneck-evidence-fusion-competitive-rca",
            "source-bottleneck-adaptive-evidence-rerank-rca",
            "source-bottleneck-adaptive-evidence-wadi-rerank-rca",
            "source-bottleneck-adaptive-evidence-margin-rerank-rca",
            "source-bottleneck-rca-aware-checkpoint",
            "source-bottleneck-normalized-rca-checkpoint",
            "source-bottleneck-adaptive-mechanism-rca",
            "source-bottleneck-event-adaptive-mechanism-rca",
            "source-bottleneck-conservative-mechanism-rca",
            "source-bottleneck-source-gated-mechanism-rca",
            "source-bottleneck-mechanism-train-only-rca",
            "source-bottleneck-no-mechanism-decoder-rca",
            "source-bottleneck-event-stable-rca",
            "source-bottleneck-no-channel-masked-rca",
            "source-bottleneck-no-mechanism-rca",
            "source-bottleneck-no-source-gate-rca",
            "source-bottleneck-no-source-bottleneck-rca",
            "source-bottleneck-no-source-propagation-rca",
            "source-bottleneck-corefine-rca",
            "source-aware-dual-corefine-rca",
            "source-preserving-mechanism-fusion-rca",
            "source-preserving-ultralight-rca",
            "source-preserving-light-rca",
            "lagged-directional-mechanism-rca",
            "source-bottleneck-trained-specificity-rca",
            "source-bottleneck-gate-specificity-rca",
            "source-bottleneck-effective-specificity-rca",
            "source-bottleneck-head-specificity-rca",
            "interventional-fused-source-rca",
            "interventional-consensus-source-rca",
            "interventional-context-source-rca",
            "mechanism-calibrated-rca",
            "source-preserving-rca",
            "synthetic-responsibility-rca",
            "counterfactual-source-rca",
            "onset-source-rca",
            "channel-only",
            "temporal-only",
            "state-aware",
            "state-aware-dynamic",
            "state-aware-causal",
            "parallel-dual",
            "parallel-dual-time",
            "parallel-dual-dynamic",
            "residual-dual",
            "residual-dual-time",
            "graph-shift",
            "graph-shift-lite",
            "graph-shift-strong",
            "causal-lag",
            "causal-lag-score",
            "causal-lag-score-strong",
            "causal-cf-gain",
            "causal-cf-hurt",
            "synthetic-aux",
            "synthetic-aux-q75",
            "full-event-affinity-lite",
            "full-event-affinity",
            "full-event-affinity-strong",
            "full-event-persistence",
            "loss-smoothl1",
            "loss-logcosh",
            "loss-mse-mae",
            "loss-mse-logcosh",
            "loss-mse-smoothl1",
            "loss-diff",
            "loss-smoothl1-diff",
            "dynamic-temporal",
            "dynamic-temporal-gated",
            "dynamic-temporal-gated-vqscore",
            "dynamic-temporal-regularized",
            "dynamic-temporal-robust",
            "dynamic-temporal-robust-lite",
            "dynamic-temporal-channelnorm",
            "dynamic-temporal-channelnorm-scale",
            "dynamic-temporal-aff",
            "dynamic-temporal-aff-lite",
            "dynamic-temporal-aff-peak",
            "dynamic-temporal-aff-wide",
            "no-scorer",
            "no-boundary",
            "reconstruction",
        ],
        default="full",
        help="架构配置: full=双图+VQ训练+重建分数; dynamic-temporal=通道图+动态时序图; channel-only/temporal-only=单图消融; with-scorer=旧多尺度/VQ分数路径; graph-only=双图+重建分数; no-vq/core 关闭 VQ; reconstruction 仅保留重建主干",
    )
    parser.add_argument(
        "--vq-cooldown-epochs",
        type=int,
        default=None,
        help="VQ codebook-only cooldown epochs; default uses model config",
    )
    parser.add_argument(
        "--lambda-vq",
        type=float,
        default=None,
        help="VQ loss weight; default uses model config",
    )
    parser.add_argument(
        "--vq-score-weight",
        type=float,
        default=None,
        help="VQ anomaly score fusion weight; default uses model config",
    )
    parser.add_argument(
        "--score-topk-k",
        type=int,
        default=None,
        help="Override channel top-k aggregation used by the reconstruction anomaly score",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=["mean", "max", "q75", "q90", "q95", "quantile", "center", "last"],
        default=None,
        help="How overlapping window scores are aggregated back to point scores",
    )
    parser.add_argument(
        "--score-aggregation-quantile",
        type=float,
        default=None,
        help="Quantile used when --score-aggregation=quantile (0~1)",
    )
    parser.add_argument(
        "--score-center-width",
        type=int,
        default=None,
        help="Number of central offsets used when --score-aggregation=center",
    )
    parser.add_argument(
        "--score-smoothing-window",
        type=int,
        default=None,
        help="Odd-size score smoothing window used before thresholding",
    )
    parser.add_argument(
        "--score-smoothing-method",
        choices=["mean", "max"],
        default=None,
        help="Score smoothing method used before thresholding",
    )
    parser.add_argument(
        "--prediction-fill-gap",
        type=int,
        default=None,
        help="Fill predicted normal gaps up to this length between anomaly segments",
    )
    parser.add_argument(
        "--prediction-min-len",
        type=int,
        default=None,
        help="Remove predicted anomaly segments shorter than this length",
    )
    parser.add_argument(
        "--prediction-dilate",
        type=int,
        default=None,
        help="Dilate predicted anomaly segments by this many points on both sides",
    )
    parser.add_argument(
        "--event-persistence-window",
        type=int,
        default=None,
        help="Window for event-persistence score amplification",
    )
    parser.add_argument(
        "--event-persistence-weight",
        type=float,
        default=None,
        help="Weight for event-persistence score amplification",
    )
    parser.add_argument(
        "--dynamic-temporal-residual-init",
        type=float,
        default=None,
        help="Override the residual gate initialization for dynamic temporal graph profiles",
    )
    parser.add_argument(
        "--dynamic-temporal-topk",
        type=int,
        default=None,
        help="Override row-wise top-k sparsity for dynamic temporal graph profiles",
    )
    parser.add_argument(
        "--temporal-graph-lr-scale",
        type=float,
        default=None,
        help="Override the temporal graph learning-rate scale",
    )
    parser.add_argument(
        "--channel-prior-align",
        type=float,
        default=None,
        help="Override normal-structure alignment loss weight for channel-prior profiles",
    )
    parser.add_argument(
        "--channel-prior-weight",
        type=float,
        default=None,
        help="Override hard normal-prior mixing weight for channel-prior profiles",
    )
    parser.add_argument(
        "--channel-prior-bias",
        type=float,
        default=None,
        help="Override normal-prior logit bias for channel-prior profiles",
    )
    parser.add_argument(
        "--channel-prior-topk",
        type=int,
        default=None,
        help="Override top-k edges retained in the normal channel correlation prior",
    )
    parser.add_argument(
        "--state-aware-num-states",
        type=int,
        default=None,
        help="Override number of latent operating states for state-aware graph fusion",
    )
    parser.add_argument(
        "--state-aware-graph-gate-init",
        type=float,
        default=None,
        help="Initial channel-graph weight for state-aware graph fusion",
    )
    parser.add_argument(
        "--state-aware-residual-init",
        type=float,
        default=None,
        help="Initial residual strength for state-aware correction over the serial graph path",
    )
    parser.add_argument(
        "--lambda-state-balance",
        type=float,
        default=None,
        help="Weight for using latent operating states across a batch",
    )
    parser.add_argument(
        "--lambda-state-confidence",
        type=float,
        default=None,
        help="Weight for low-entropy state assignments in state-aware profiles",
    )
    parser.add_argument(
        "--causal-score-weight",
        type=float,
        default=None,
        help="Override causal score fusion weight for causal profiles",
    )
    parser.add_argument(
        "--lambda-causal-mechanism",
        type=float,
        default=None,
        help="Override lagged causal mechanism loss weight",
    )
    parser.add_argument(
        "--lambda-channel-mechanism",
        type=float,
        default=None,
        help="Override channel-neighbor mechanism reconstruction loss weight",
    )
    parser.add_argument(
        "--channel-mechanism-score-weight",
        type=float,
        default=None,
        help="Override channel mechanism-violation score fusion weight",
    )
    parser.add_argument(
        "--mechanism-coupling-init",
        type=float,
        default=None,
        help="Override initial mechanism coupling gate for mechanism-coupled profiles",
    )
    parser.add_argument(
        "--mechanism-predictive-blend-init",
        type=float,
        default=None,
        help="Override initial mechanism-predictive fusion gate",
    )
    parser.add_argument(
        "--causal-topk",
        type=int,
        default=None,
        help="Override lagged causal parent top-k",
    )
    parser.add_argument(
        "--lambda-synthetic-anomaly",
        type=float,
        default=None,
        help="Override synthetic anomaly auxiliary loss weight",
    )
    parser.add_argument(
        "--lambda-synthetic-rca",
        type=float,
        default=None,
        help="Override synthetic variable-level RCA ranking loss weight",
    )
    parser.add_argument(
        "--synthetic-aux-interval",
        type=int,
        default=None,
        help="Run synthetic anomaly/RCA auxiliary loss every N training batches; 1 keeps the original behavior",
    )
    parser.add_argument(
        "--synthetic-rca-margin",
        type=float,
        default=None,
        help="Margin for synthetic variable-level RCA ranking loss",
    )
    parser.add_argument(
        "--synthetic-rca-topk",
        type=int,
        default=None,
        help="Number of hardest non-root variables used by synthetic RCA ranking loss",
    )
    parser.add_argument(
        "--synthetic-rca-positive-aggregation",
        choices=[
            "mean",
            "min",
            "weakest",
            "bottomk",
            "bottom",
            "weak_mean",
            "blend",
            "mean_bottomk",
            "mean_weak",
        ],
        default=None,
        help="How positive root variables are aggregated inside synthetic RCA ranking loss.",
    )
    parser.add_argument(
        "--synthetic-rca-positive-topk",
        type=int,
        default=None,
        help="Number of weakest positive root variables used when positive aggregation is bottomk.",
    )
    parser.add_argument(
        "--synthetic-rca-positive-min-weight",
        type=float,
        default=None,
        help="Blend weight for weakest positive roots when positive aggregation is blend.",
    )
    parser.add_argument(
        "--synthetic-rca-bce-weight",
        type=float,
        default=None,
        help="Weight for BCE part of synthetic variable-responsibility training.",
    )
    parser.add_argument(
        "--synthetic-rca-rank-weight",
        type=float,
        default=None,
        help="Weight for ranking part of synthetic variable-responsibility training.",
    )
    parser.add_argument(
        "--synthetic-score-weight",
        type=float,
        default=None,
        help="Override synthetic anomaly head score fusion weight",
    )
    parser.add_argument(
        "--lambda-source-bottleneck",
        type=float,
        default=None,
        help="Weight for source-gate temporal bottleneck supervision on synthetic source-effect events",
    )
    parser.add_argument(
        "--source-bottleneck-bce-weight",
        type=float,
        default=None,
        help="BCE term weight inside source-gate temporal bottleneck supervision",
    )
    parser.add_argument(
        "--source-bottleneck-rank-weight",
        type=float,
        default=None,
        help="Onset ranking term weight inside source-gate temporal bottleneck supervision",
    )
    parser.add_argument(
        "--source-bottleneck-effect-suppress-weight",
        type=float,
        default=None,
        help="Propagation-channel suppression weight inside source-gate temporal bottleneck supervision",
    )
    parser.add_argument(
        "--root-response-penalty-init",
        type=float,
        default=None,
        help="Initial penalty applied to graph-supported response evidence in the root-response RCA head.",
    )
    parser.add_argument(
        "--root-response-confidence-discount",
        type=float,
        default=None,
        help="Maximum source-confidence discount applied to response evidence in the root-response RCA head.",
    )
    parser.add_argument(
        "--root-response-use-innovation-split",
        action="store_true",
        help="Move unexplained propagation residual into the source side of the root-response RCA head.",
    )
    parser.add_argument(
        "--pairwise-root-response",
        action="store_true",
        help="Use a low-rank pairwise source-to-response relation head for root-response RCA.",
    )
    parser.add_argument(
        "--lambda-pairwise-root-response",
        type=float,
        default=None,
        help="Auxiliary weight for supervising pairwise source-to-response relations on synthetic events.",
    )
    parser.add_argument(
        "--pairwise-root-response-logit-weight",
        type=float,
        default=None,
        help="Weight of pairwise source-response support inside the root-response logit; 0 keeps it as training-only supervision.",
    )
    parser.add_argument(
        "--root-response-source-branch-weight",
        type=float,
        default=None,
        help="Auxiliary weight for supervising root-response source evidence on synthetic source onsets.",
    )
    parser.add_argument(
        "--root-response-response-branch-weight",
        type=float,
        default=None,
        help="Auxiliary weight for ranking graph-supported response evidence on synthetic effect variables.",
    )
    parser.add_argument(
        "--root-response-response-suppress-weight",
        type=float,
        default=None,
        help="Penalty weight that keeps response evidence low on synthetic source variables.",
    )
    parser.add_argument(
        "--lambda-event-responsibility",
        type=float,
        default=None,
        help="Auxiliary loss weight for the event-level channel responsibility head.",
    )
    parser.add_argument(
        "--event-responsibility-rank-weight",
        type=float,
        default=None,
        help="Ranking-loss weight inside the event-level channel responsibility head.",
    )
    parser.add_argument(
        "--event-responsibility-effect-suppress-weight",
        type=float,
        default=None,
        help="Penalty that keeps responsibility low on synthetic propagated-effect channels.",
    )
    parser.add_argument(
        "--lambda-channel-masked",
        type=float,
        default=None,
        help="Override masked-channel mechanism modeling loss weight",
    )
    parser.add_argument(
        "--channel-mask-interval",
        type=int,
        default=None,
        help="Run masked-channel mechanism modeling every N training batches",
    )
    parser.add_argument(
        "--channel-mask-ratio",
        type=float,
        default=None,
        help="Fraction of channels masked per training sample for mechanism modeling",
    )
    parser.add_argument(
        "--channel-mask-value",
        choices=["zero", "mean"],
        default=None,
        help="Fill value for masked channels during mechanism modeling",
    )
    parser.add_argument(
        "--reconstruction-loss",
        choices=[
            "mse",
            "mae",
            "smooth_l1",
            "log_cosh",
            "charbonnier",
            "mse_mae",
            "mse_log_cosh",
            "mse_smooth_l1",
        ],
        default=None,
        help="Override the reconstruction training loss type",
    )
    parser.add_argument(
        "--smooth-l1-beta",
        type=float,
        default=None,
        help="SmoothL1 beta when --reconstruction-loss=smooth_l1",
    )
    parser.add_argument(
        "--mse-l1-alpha",
        type=float,
        default=None,
        help="MSE weight for reconstruction-loss=mse_mae",
    )
    parser.add_argument(
        "--mse-robust-alpha",
        type=float,
        default=None,
        help="MSE weight for reconstruction-loss=mse_log_cosh or mse_smooth_l1",
    )
    parser.add_argument(
        "--charbonnier-eps",
        type=float,
        default=None,
        help="Epsilon for reconstruction-loss=charbonnier",
    )
    parser.add_argument(
        "--lambda-temporal-diff-loss",
        type=float,
        default=None,
        help="Weight for temporal first-difference reconstruction loss",
    )
    parser.add_argument(
        "--export-rca",
        action="store_true",
        help="Export event-level root-cause rankings based on channel reconstruction contributions.",
    )
    parser.add_argument(
        "--rca-export-lite",
        action="store_true",
        help="Export only selected prediction-key RCA events and truncated rankings.",
    )
    parser.add_argument(
        "--rca-export-top-k",
        type=int,
        default=None,
        help="Top-K channel ranking length kept when --rca-export-lite is enabled.",
    )
    parser.add_argument(
        "--rca-event-local-export",
        action="store_true",
        help="Compute RCA channel scores only around true/predicted events to speed up large datasets.",
    )
    parser.add_argument(
        "--rca-event-local-margin",
        type=int,
        default=None,
        help="Point margin around true/predicted events used by --rca-event-local-export.",
    )
    parser.add_argument(
        "--rca-split-predicted-events",
        action="store_true",
        help="Split long predicted anomaly segments into local RCA windows without changing detection labels.",
    )
    parser.add_argument(
        "--rca-split-max-event-len",
        type=int,
        default=None,
        help="Maximum length of a local predicted RCA window when splitting long predicted events.",
    )
    parser.add_argument(
        "--rca-split-stride",
        type=int,
        default=None,
        help="Stride for local predicted RCA windows when splitting long predicted events.",
    )
    parser.add_argument(
        "--use-latest-checkpoint",
        action="store_true",
        help="Use the latest trained checkpoint for evaluation instead of the best validation-loss checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=["profile", "best-val", "latest", "rca-aware", "normalized-rca-aware"],
        default="profile",
        help=(
            "Checkpoint selection policy. 'profile' keeps arch-profile defaults; "
            "'best-val' forces validation-loss checkpoint selection; 'latest' evaluates "
            "the last epoch. RCA-aware policies are experimental."
        ),
    )
    parser.add_argument(
        "--rca-aware-checkpoint",
        action="store_true",
        help="Select checkpoints using validation loss plus a synthetic source-RCA proxy.",
    )
    parser.add_argument(
        "--rca-checkpoint-normalize",
        action="store_true",
        help="Use normalized validation loss for RCA-aware checkpoint selection.",
    )
    parser.add_argument(
        "--rca-checkpoint-proxy-weight",
        type=float,
        default=None,
        help="Weight of the synthetic source-RCA proxy used for RCA-aware checkpoint selection.",
    )
    parser.add_argument(
        "--rca-checkpoint-proxy-batches",
        type=int,
        default=None,
        help="Number of validation batches used by RCA-aware checkpoint proxy.",
    )
    parser.add_argument(
        "--rca-checkpoint-min-epoch",
        type=int,
        default=None,
        help="First epoch allowed to use RCA-aware checkpoint selection.",
    )
    parser.add_argument(
        "--enable-visualization-hooks",
        action="store_true",
        help="Enable diagnostic visualization hooks and vis_data.json export. Disabled by default for fast experiments.",
    )
    parser.add_argument(
        "--debug-loss-breakdown",
        action="store_true",
        help="Record per-epoch training loss component summaries for diagnosis.",
    )
    parser.add_argument(
        "--debug-loss-log-path",
        type=str,
        default=None,
        help="JSONL path for --debug-loss-breakdown summaries.",
    )
    parser.add_argument(
        "--debug-loss-max-batches",
        type=int,
        default=None,
        help="Maximum batches per epoch included in loss breakdown; 0 means all batches.",
    )
    parser.add_argument(
        "--debug-train-max-batches",
        type=int,
        default=None,
        help="Diagnostic-only cap on training batches per epoch; 0 means full training.",
    )
    parser.add_argument(
        "--rca-graph-weight",
        type=float,
        default=None,
        help="Weight for graph-propagated attribution in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-mechanism-weight",
        type=float,
        default=None,
        help="Weight for channel mechanism-violation attribution in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-source-weight",
        type=float,
        default=None,
        help="Weight for event-level source attribution when source/propagation RCA is enabled.",
    )
    parser.add_argument(
        "--rca-source-base-weight",
        type=float,
        default=None,
        help="Weight for base reconstruction residual inside the RCA source score.",
    )
    parser.add_argument(
        "--rca-source-score-weight",
        type=float,
        default=None,
        help="Weight for the fused source score inside normalized source/propagation RCA.",
    )
    parser.add_argument(
        "--rca-source-gate-weight",
        type=float,
        default=None,
        help="Weight for learned source-gate evidence inside RCA rankings.",
    )
    parser.add_argument(
        "--rca-root-score-weight",
        type=float,
        default=None,
        help="Weight for learned RootScore evidence inside RCA rankings.",
    )
    parser.add_argument(
        "--rca-event-responsibility-weight",
        type=float,
        default=None,
        help="Weight for learned event-responsibility evidence inside RCA rankings.",
    )
    parser.add_argument(
        "--rca-root-score-signal",
        choices=["prob", "source_confidence", "source_evidence", "logit"],
        default=None,
        help="Neural signal exported as learned RootScore evidence in RCA rankings.",
    )
    parser.add_argument(
        "--rca-root-score-pooling",
        choices=["mean", "early_mean", "top_quantile", "max"],
        default=None,
        help="Event aggregation used for learned RootScore evidence in RCA rankings.",
    )
    parser.add_argument(
        "--rca-root-score-head-ratio",
        type=float,
        default=None,
        help="Early-event fraction used when --rca-root-score-pooling=early_mean.",
    )
    parser.add_argument(
        "--rca-root-score-head-points",
        type=int,
        default=None,
        help="Maximum leading points used when --rca-root-score-pooling=early_mean.",
    )
    parser.add_argument(
        "--rca-root-score-top-quantile",
        type=float,
        default=None,
        help="Per-channel quantile threshold used when --rca-root-score-pooling=top_quantile.",
    )
    parser.add_argument(
        "--rca-source-consensus-weight",
        type=float,
        default=None,
        help="Weight for source evidence gated by base/graph consensus in RCA rankings.",
    )
    parser.add_argument(
        "--rca-onset-consensus-weight",
        type=float,
        default=None,
        help="Weight for onset evidence gated by base/graph consensus in RCA rankings.",
    )
    parser.add_argument(
        "--rca-source-consensus-mode",
        choices=["base", "graph", "max_bg", "min_bg", "sqrt_bg", "gate", "none"],
        default=None,
        help="Consensus gate used by source/onset RCA evidence.",
    )
    parser.add_argument(
        "--rca-propagation-weight",
        type=float,
        default=None,
        help="Weight for graph-propagated attribution when source/propagation RCA is enabled.",
    )
    parser.add_argument(
        "--rca-source-mechanism-weight",
        type=float,
        default=None,
        help="Weight for mechanism-prior deviation inside the RCA source score.",
    )
    parser.add_argument(
        "--rca-source-innovation-weight",
        type=float,
        default=None,
        help="Weight for normal-prior-unexplained residual inside the RCA source score.",
    )
    parser.add_argument(
        "--rca-source-innovation-mode",
        choices=["series", "onset_directional", "directional_onset", "temporal"],
        default=None,
        help="How to compute source innovation in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-source-innovation-neighbor-weight",
        type=float,
        default=None,
        help="Neighbor support strength subtracted when computing source innovation.",
    )
    parser.add_argument(
        "--rca-source-innovation-lead-points",
        type=int,
        default=None,
        help="Minimum event-local lead for neighbors to explain a variable in directional source innovation.",
    )
    parser.add_argument(
        "--rca-causal-weight",
        type=float,
        default=None,
        help="Weight for lagged causal deviation inside the RCA source score.",
    )
    parser.add_argument(
        "--rca-synthetic-weight",
        type=float,
        default=None,
        help="Weight for learned synthetic-responsibility score inside RCA rankings.",
    )
    parser.add_argument(
        "--rca-counterfactual-weight",
        type=float,
        default=None,
        help="Weight for event-local counterfactual responsibility inside RCA rankings.",
    )
    parser.add_argument(
        "--rca-counterfactual-candidates",
        type=int,
        default=None,
        help="Top-K channels tested by event-local counterfactual RCA.",
    )
    parser.add_argument(
        "--rca-counterfactual-max-windows",
        type=int,
        default=None,
        help="Maximum sampled windows per event for counterfactual RCA.",
    )
    parser.add_argument(
        "--rca-onset-weight",
        type=float,
        default=None,
        help="Weight for early local-baseline crossing in event-level RCA rankings.",
    )
    parser.add_argument(
        "--rca-onset-baseline-window",
        type=int,
        default=None,
        help="Number of points before each event used to estimate onset baselines.",
    )
    parser.add_argument(
        "--rca-onset-z",
        type=float,
        default=None,
        help="Robust z threshold used to decide whether a variable has crossed its onset baseline.",
    )
    parser.add_argument(
        "--rca-graph-direction",
        choices=["outgoing", "incoming", "both"],
        default=None,
        help="Direction used to propagate channel errors over the learned channel graph for RCA.",
    )
    parser.add_argument(
        "--rca-contrast-window",
        type=int,
        default=None,
        help="Number of points before each anomaly event used as the local RCA baseline.",
    )
    parser.add_argument(
        "--rca-contrast-weight",
        type=float,
        default=None,
        help="Weight for positive event-vs-local-baseline lift in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-mechanism-residual-window",
        type=int,
        default=None,
        help="Number of points before each event used as the mechanism-residual RCA baseline.",
    )
    parser.add_argument(
        "--rca-mechanism-residual-weight",
        type=float,
        default=None,
        help="Weight for positive mechanism-error lift in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-adaptive-mechanism-gate-weight",
        type=float,
        default=None,
        help="Weight for event-adaptive mechanism evidence gated by source/onset support.",
    )
    parser.add_argument(
        "--rca-adaptive-mechanism-gate-floor",
        type=float,
        default=None,
        help="Minimum adaptive mechanism gate value in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-adaptive-mechanism-gate-mode",
        choices=["source_onset", "source_gate", "onset", "base", "base_source_onset"],
        default=None,
        help="Support signal used to decide when mechanism evidence should affect RCA.",
    )
    parser.add_argument(
        "--rca-adaptive-mechanism-selection-weight",
        type=float,
        default=None,
        help="Blend weight for event-level selection between direct source evidence and mechanism evidence.",
    )
    parser.add_argument(
        "--rca-adaptive-mechanism-selection-floor",
        type=float,
        default=None,
        help="Minimum event-level mechanism selection gate value in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-adaptive-mechanism-selection-mode",
        choices=["source_onset", "source_gate", "onset", "base", "base_source_onset"],
        default=None,
        help="Support signal used to select mechanism evidence at event level.",
    )
    parser.add_argument(
        "--rca-conservative-mechanism-weight",
        type=float,
        default=None,
        help="Weight for mechanism evidence that is gated by source/onset support.",
    )
    parser.add_argument(
        "--rca-conservative-mechanism-support-floor",
        type=float,
        default=None,
        help="Minimum normalized source/onset support required before mechanism evidence can boost RCA.",
    )
    parser.add_argument(
        "--rca-conservative-mechanism-candidate-topk",
        type=int,
        default=None,
        help="Restrict conservative mechanism boosting to the top-K direct source candidates; 0 disables the mask.",
    )
    parser.add_argument(
        "--rca-conservative-mechanism-support-mode",
        choices=["source_onset", "source_gate", "onset", "source_score", "base", "base_source_onset"],
        default=None,
        help="Support signal used by conservative mechanism RCA.",
    )
    parser.add_argument(
        "--rca-source-gated-mechanism-weight",
        type=float,
        default=None,
        help="Weight for strict source-gated mechanism evidence in exported RCA rankings.",
    )
    parser.add_argument(
        "--rca-source-gated-mechanism-support-floor",
        type=float,
        default=None,
        help="Minimum normalized source support required before strict mechanism evidence can boost RCA.",
    )
    parser.add_argument(
        "--rca-source-gated-mechanism-candidate-topk",
        type=int,
        default=None,
        help="Restrict strict source-gated mechanism boosting to the top-K direct source candidates; 0 disables the mask.",
    )
    parser.add_argument(
        "--rca-source-gated-mechanism-support-mode",
        choices=["source_onset", "source_gate", "onset", "source_score", "base", "base_source_onset"],
        default=None,
        help="Support signal used by strict source-gated mechanism RCA.",
    )
    parser.add_argument(
        "--rca-event-head-ratio",
        type=float,
        default=None,
        help="Use only the first fraction of each event for RCA ranking; 1.0 keeps full-event aggregation.",
    )
    parser.add_argument(
        "--rca-event-head-points",
        type=int,
        default=None,
        help="Use at most this many leading points of each event for RCA ranking.",
    )
    parser.add_argument(
        "--rca-prediction-key",
        type=str,
        default=None,
        help="Prediction key used for predicted-event RCA export, e.g. pot, 0.5, 1.0, 5.",
    )
    args = parser.parse_args()

    # 确定训练轮次
    if args.epochs is not None:
        train_epochs = args.epochs
    elif args.fast:
        train_epochs = FAST_EPOCHS
    elif args.paper_protocol:
        train_epochs = 15
    else:
        train_epochs = DEFAULT_EPOCHS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 始终使用 large_detect（覆盖所有数据集）
    data_config = {
        "data_set_name": ["large_detect"],
    }
    
    # 确定数据集列表：用户指定则用它，否则自动排除不兼容的数据集
    EXCLUDED_FILES = {
        "CMAPSS_FD001.csv", "CMAPSS_FD002.csv",
        "CMAPSS_FD003.csv", "CMAPSS_FD004.csv",
        "SKAB_all.csv",   # 标签格式不兼容
    }
    EXCLUDED_PREFIXES = (
        "TE_MM_",          # 工业迁移数据集默认不混入常规实验；需要时用 --datasets 显式运行
        "HAI_",
    )

    def _is_default_excluded(file_name: str) -> bool:
        return file_name in EXCLUDED_FILES or file_name.startswith(EXCLUDED_PREFIXES)

    if args.datasets:
        data_config["data_name_list"] = args.datasets
    else:
        # 默认排除不适用的数据集
        data_src = LocalAnomalyDetectDataSource()
        all_files = data_src.dataset.metadata["file_name"].tolist()
        filtered = [f for f in all_files if not _is_default_excluded(f)]
        data_config["data_name_list"] = filtered
        excluded = [f for f in all_files if _is_default_excluded(f)]
        if excluded:
            print(f"  Note: Excluded datasets: {excluded}")

    # ============================================================
    # v10 SparseLaGraph 固定配置（不再随 GPU 数缩放容量）
    # ============================================================
    # 根据 v10 设计文档（REFACTOR_PLAN.md）：
    #   - d_model=128, e_layers=2, n_heads=4 → 参数量 ~0.4M
    #   - 已移除 contrastive/freq/pred/prototype 等零贡献模块
    #   - 仅保留 MSE重建 + L1稀疏正则
    #   - 差分学习率 + Warmup + CosineAnnealing + EarlyStopping
    # ============================================================
    if args.n_gpus > 1:
        print(f"  [INFO] --n-gpus {args.n_gpus} requested; DDP is disabled for RTX 5070 migration, using 1 GPU.")
    n_gpus = 0 if args.n_gpus == 0 else 1
    dataloader_num_workers = max(0, args.num_workers)
    dataloader_prefetch_factor = max(1, args.prefetch_factor)
    inference_dataloader_num_workers = max(0, args.inference_num_workers)
    vq_cooldown_epochs = args.vq_cooldown_epochs
    lambda_vq = args.lambda_vq
    vq_score_weight = args.vq_score_weight

    # ---- v10 固定模型容量（无论 GPU 数多少都用相同容量） ----
    d_model_scale = 128
    e_layers_scale = 2
    n_heads_scale = 4

    # ★ HYPERPARAMETER ANALYSIS: batch_size
    # 
    # 问题：原 per_gpu_batch=2048 导致有效 batch=16384 (8卡)，
    # 梯度方差极大，训练 loss 曲线剧烈震荡。
    # 对于时间序列异常检测，每个样本是一个 (100, C) 的滑动窗口，
    # 每个 batch 包含大量"太相似"的窗口，产生冗余梯度。
    #
    # v10 修复分析：
    #   模型仅 0.4M 参数，MSE 重建任务简单（正常模式高度重复），
    #   小 batch 已足够估计梯度方向。
    #   256 vs 2048：2048 的 8× 冗余梯度反而引入更多噪声。
    #
    # 经验值（SWaT 52,800 训练样本, win_size=100, step=1）:
    #   batch=2048: 26 steps/epoch, 训练 loss 震荡 ±30%
    #   batch=256:  206 steps/epoch, 训练 loss 收敛平滑
    #   batch=128:  412 steps/epoch, 训练更稳定但速度慢
    #
    # 结论：batch=256 在收敛质量和训练速度之间取得最佳平衡
    # 
    # [数据科学依据]
    #   时间序列异常检测的"信号噪声比"远低于分类任务：
    #     分类：每个样本有独立标签，大 batch 降低梯度方差
    #     重建：每个窗口包含几乎相同的正常模式，大 batch 冗余
    #   因此 batch=256 比 2048 更合理。
    per_gpu_batch = 256
    if args.batch_size is not None:
        per_gpu_batch = max(1, int(args.batch_size))

    # ---- v10 固定 LR（不含 sqrt 缩放，因为模型容量固定） ----
    scaled_lr = 1e-4

    # ★ Hyperparameter analysis: warmup_epochs
    #   v11 新增 VQ Bottleneck:
    #     前 vq_cooldown_epochs=15 轮仅更新 codebook，需要足够预热。
    #     warmup_epochs=5 太短，codebook 在冷却期结束时还处于随机初始化状态。
    #   修复: warmup_epochs=10, vq_cooldown_epochs=15
    #     warmup 期间 LR 从 1e-6 线性升到 1e-4，codebook 有 10 轮稳定训练。
    warmup_epochs = 10

    arch_profiles = {
        "full": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "with-scorer": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": True,
        },
        "core": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": False,
            "use_multi_scale_scorer": True,
        },
        "no-vq": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": False,
            "use_multi_scale_scorer": True,
        },
        "graph-only": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": False,
            "use_multi_scale_scorer": False,
        },
        "prior-guided-graph": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.30,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
        },
        "structure-consistent": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.0,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.001,
        },
        "structure-biased": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.0,
            "channel_corr_prior_bias": 0.5,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.0,
        },
        "mechanism-graph": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "lambda_channel_mechanism": 0.02,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.05,
            "rca_mechanism_weight": 1.0,
        },
        "mechanism-prior-graph": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_weight": 0.0,
            "channel_corr_prior_bias": 0.5,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.001,
            "lambda_channel_mechanism": 0.02,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.05,
            "rca_mechanism_weight": 1.0,
        },
        "mechanism-coupled": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.02,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.05,
            "rca_mechanism_weight": 1.0,
        },
        "mechanism-coupled-lite": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.05,
            "lambda_channel_mechanism": 0.005,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.02,
            "rca_mechanism_weight": 0.5,
        },
        "mechanism-coupled-noscore": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.05,
            "lambda_channel_mechanism": 0.005,
            "use_channel_mechanism_score": False,
            "channel_mechanism_score_weight": 0.0,
            "rca_mechanism_weight": 1.0,
        },
        "mechanism-predictive": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "rca_mechanism_weight": 1.0,
            "rca_graph_weight": 0.2,
        },
        "mechanism-predictive-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "rca_mechanism_weight": 1.0,
            "rca_graph_weight": 0.2,
            "use_synthetic_rca_loss": True,
            "lambda_synthetic_rca": 0.05,
            "synthetic_rca_margin": 0.2,
            "synthetic_rca_topk": 5,
            "synthetic_rca_min_roots": 1,
            "synthetic_rca_max_roots": 3,
        },
        "mechanism-predictive-robust-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
        },
        "source-propagation-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 0.75,
            "rca_propagation_weight": 0.25,
            "rca_source_mechanism_weight": 0.10,
            "rca_causal_weight": 0.0,
            "rca_event_head_ratio": 0.30,
            "rca_event_head_points": 30,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "normal-mechanism-source": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 0.75,
            "rca_propagation_weight": 0.25,
            "rca_source_mechanism_weight": 0.15,
            "rca_causal_weight": 0.0,
            "rca_event_head_ratio": 0.30,
            "rca_event_head_points": 30,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "graph-coupled-source": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.03,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 0.75,
            "rca_propagation_weight": 0.25,
            "rca_source_mechanism_weight": 0.10,
            "rca_causal_weight": 0.0,
            "rca_event_head_ratio": 0.30,
            "rca_event_head_points": 30,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "gated-dual-source": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "use_state_aware_fusion": True,
            "state_aware_num_states": 4,
            "state_aware_graph_gate_init": 0.6,
            "state_aware_residual_init": 0.15,
            "lambda_state_balance": 0.001,
            "lambda_state_confidence": 0.001,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 0.75,
            "rca_propagation_weight": 0.25,
            "rca_source_mechanism_weight": 0.10,
            "rca_causal_weight": 0.0,
            "rca_event_head_ratio": 0.30,
            "rca_event_head_points": 30,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "causal-gated-source": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "use_state_aware_fusion": True,
            "state_aware_num_states": 4,
            "state_aware_graph_gate_init": 0.6,
            "state_aware_residual_init": 0.15,
            "lambda_state_balance": 0.001,
            "lambda_state_confidence": 0.001,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 0.75,
            "rca_propagation_weight": 0.25,
            "rca_source_mechanism_weight": 0.10,
            "rca_causal_weight": 0.05,
            "rca_event_head_ratio": 0.30,
            "rca_event_head_points": 30,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "causal-source-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 1.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
        },
        "causal-source-robust-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 1.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
        },
        "source-gated-causal-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_source_gate": True,
            "source_gate_init": 0.20,
            "lambda_source_gate_sparse": 0.001,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 0.5,
            "rca_source_gate_weight": 1.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "source-effect-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_source_gate": True,
            "source_gate_init": 0.20,
            "lambda_source_gate_sparse": 0.0005,
            "use_source_effect_synthetic": True,
            "lambda_source_effect": 0.06,
            "source_effect_interval": 16,
            "source_effect_min_len": 8,
            "source_effect_max_len": 30,
            "source_effect_min_roots": 1,
            "source_effect_max_roots": 2,
            "source_effect_neighbor_topk": 3,
            "source_effect_strength": 0.35,
            "source_effect_delay_max": 6,
            "source_effect_bce_weight": 1.0,
            "source_effect_rank_weight": 1.0,
            "source_effect_effect_rank_weight": 0.5,
            "source_effect_margin": 0.15,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 0.0,
            "rca_source_gate_weight": 0.4,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "mechanism-masked-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.05,
            "use_channel_masked_modeling": True,
            "lambda_channel_masked": 0.05,
            "channel_mask_interval": 8,
            "channel_mask_ratio": 0.15,
            "channel_mask_min_channels": 1,
            "channel_mask_value": "zero",
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 0.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "mechanism-root-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.05,
            "use_channel_masked_modeling": True,
            "lambda_channel_masked": 0.05,
            "channel_mask_interval": 8,
            "channel_mask_ratio": 0.15,
            "channel_mask_min_channels": 1,
            "channel_mask_value": "zero",
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_event_component_normalize": True,
            "rca_graph_weight": 1.0,
            "rca_graph_penalty_weight": 0.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 0.20,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_mechanism_residual_window": 300,
            "rca_mechanism_residual_weight": 0.25,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "mechanism-root-sharp-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.05,
            "use_channel_masked_modeling": True,
            "lambda_channel_masked": 0.05,
            "channel_mask_interval": 8,
            "channel_mask_ratio": 0.15,
            "channel_mask_min_channels": 1,
            "channel_mask_value": "zero",
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_event_component_normalize": True,
            "rca_graph_weight": 1.0,
            "rca_graph_penalty_weight": 0.10,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.50,
            "rca_source_interaction_weight": 0.50,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 0.80,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_mechanism_residual_window": 300,
            "rca_mechanism_residual_weight": 0.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "onset-source-sharp-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.05,
            "use_channel_masked_modeling": True,
            "lambda_channel_masked": 0.05,
            "channel_mask_interval": 8,
            "channel_mask_ratio": 0.15,
            "channel_mask_min_channels": 1,
            "channel_mask_value": "zero",
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_event_component_normalize": True,
            "rca_graph_weight": 1.0,
            "rca_graph_penalty_weight": 0.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_source_interaction_weight": 0.0,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 1.50,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_mechanism_residual_window": 300,
            "rca_mechanism_residual_weight": 0.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "adaptive-source-sharp-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.05,
            "use_channel_masked_modeling": True,
            "lambda_channel_masked": 0.05,
            "channel_mask_interval": 8,
            "channel_mask_ratio": 0.15,
            "channel_mask_min_channels": 1,
            "channel_mask_value": "zero",
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_event_component_normalize": True,
            "rca_graph_weight": 1.0,
            "rca_graph_penalty_weight": 0.10,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.50,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_source_interaction_weight": 2.0,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 0.75,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_mechanism_residual_window": 300,
            "rca_mechanism_residual_weight": 0.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "hierarchical-source-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": False,
            "use_vq_bypass": True,
            "vq_cooldown_epochs": 2,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "use_mechanism_coupled_decoder": True,
            "mechanism_coupling_init": 0.15,
            "lambda_channel_mechanism": 0.05,
            "use_channel_masked_modeling": True,
            "lambda_channel_masked": 0.05,
            "channel_mask_interval": 8,
            "channel_mask_ratio": 0.15,
            "channel_mask_min_channels": 1,
            "channel_mask_value": "zero",
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_event_component_normalize": True,
            "rca_graph_weight": 1.0,
            "rca_graph_penalty_weight": 0.10,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.50,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_source_interaction_weight": 2.0,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 0.75,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_mechanism_residual_window": 300,
            "rca_mechanism_residual_weight": 0.0,
            "rca_hierarchical_mode": "annotate",
            "rca_hierarchical_group_aggregation": "max",
            "rca_hierarchical_group_topk": 0,
            "rca_hierarchical_group_boost": 0.0,
            "rca_hierarchical_outside_penalty": 0.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
            "rca_split_predicted_events": True,
            "rca_split_max_event_len": 120,
            "rca_split_stride": 60,
        },
        "synthetic-responsibility-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "use_synthetic_rca_head": True,
            "use_synthetic_rca_loss": True,
            "use_latest_checkpoint": True,
            "eval_batch_size": 256,
            "lambda_synthetic_rca": 0.08,
            "synthetic_aux_interval": 2,
            "synthetic_rca_margin": 0.15,
            "synthetic_rca_topk": 8,
            "synthetic_rca_min_roots": 1,
            "synthetic_rca_max_roots": 3,
            "synthetic_rca_bce_weight": 1.0,
            "synthetic_rca_rank_weight": 0.5,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.4,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 0.2,
            "rca_synthetic_weight": 1.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "counterfactual-source-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "eval_batch_size": 256,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 0.2,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.0,
            "rca_causal_weight": 0.2,
            "rca_synthetic_weight": 0.0,
            "rca_counterfactual_weight": 1.0,
            "rca_counterfactual_candidates": 12,
            "rca_counterfactual_max_windows": 32,
            "rca_counterfactual_batch_candidates": 4,
            "rca_counterfactual_baseline_window": 300,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
        },
        "onset-source-rca": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 3, 6, 12],
            "causal_topk": 5,
            "causal_detach_backbone": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_channel_corr_prior": True,
            "channel_corr_prior_topk": 5,
            "lambda_channel_prior_align": 0.01,
            "use_mechanism_predictive_head": True,
            "mechanism_predictive_blend_init": 0.30,
            "lambda_channel_mechanism": 0.05,
            "lambda_causal_mechanism": 0.01,
            "lambda_causal_sparse": 0.001,
            "use_channel_mechanism_score": True,
            "channel_mechanism_score_weight": 0.30,
            "rca_use_source_propagation": True,
            "rca_graph_weight": 1.0,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.15,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 0.5,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
        },
        "channel-only": {
            "use_channel_graph": True,
            "use_temporal_graph": False,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "temporal-only": {
            "use_channel_graph": False,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "state-aware": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_state_aware_fusion": True,
            "state_aware_num_states": 4,
            "state_aware_graph_gate_init": 0.6,
            "state_aware_residual_init": 0.15,
            "lambda_state_balance": 0.001,
            "lambda_state_confidence": 0.001,
        },
        "state-aware-dynamic": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_state_aware_fusion": True,
            "state_aware_num_states": 4,
            "state_aware_graph_gate_init": 0.6,
            "state_aware_residual_init": 0.15,
            "lambda_state_balance": 0.001,
            "lambda_state_confidence": 0.001,
        },
        "state-aware-causal": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_state_aware_fusion": True,
            "state_aware_num_states": 4,
            "state_aware_graph_gate_init": 0.6,
            "state_aware_residual_init": 0.15,
            "lambda_state_balance": 0.001,
            "lambda_state_confidence": 0.001,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 2, 4],
            "causal_topk": 5,
            "lambda_causal_mechanism": 0.05,
            "lambda_causal_sparse": 0.001,
            "use_causal_score": True,
            "causal_score_mode": "cf_parent_hurt",
            "causal_score_tail": "upper",
            "causal_score_weight": 0.05,
        },
        "parallel-dual": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_parallel_graph_fusion": True,
            "graph_fusion_gate_mode": "sample",
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "parallel-dual-time": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_parallel_graph_fusion": True,
            "graph_fusion_gate_mode": "time",
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "parallel-dual-dynamic": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_parallel_graph_fusion": True,
            "graph_fusion_gate_mode": "sample",
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "residual-dual": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_parallel_graph_fusion": True,
            "graph_fusion_gate_mode": "sample",
            "graph_fusion_strategy": "residual_serial",
            "graph_fusion_residual_init": 0.1,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "residual-dual-time": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_parallel_graph_fusion": True,
            "graph_fusion_gate_mode": "time",
            "graph_fusion_strategy": "residual_serial",
            "graph_fusion_residual_init": 0.1,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "graph-shift-lite": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_graph_shift_score": True,
            "graph_shift_score_weight": 0.03,
        },
        "graph-shift": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_graph_shift_score": True,
            "graph_shift_score_weight": 0.1,
        },
        "graph-shift-strong": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_graph_shift_score": True,
            "graph_shift_score_weight": 0.3,
        },
        "causal-lag": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 2, 4],
            "causal_topk": 5,
            "lambda_causal_mechanism": 0.05,
            "lambda_causal_sparse": 0.001,
            "use_causal_score": False,
        },
        "causal-lag-score": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 2, 4],
            "causal_topk": 5,
            "lambda_causal_mechanism": 0.05,
            "lambda_causal_sparse": 0.001,
            "use_causal_score": True,
            "causal_score_weight": 0.05,
        },
        "causal-lag-score-strong": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 2, 4],
            "causal_topk": 5,
            "lambda_causal_mechanism": 0.05,
            "lambda_causal_sparse": 0.001,
            "use_causal_score": True,
            "causal_score_weight": 0.15,
        },
        "causal-cf-gain": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 2, 4],
            "causal_topk": 5,
            "lambda_causal_mechanism": 0.05,
            "lambda_causal_sparse": 0.001,
            "use_causal_score": True,
            "causal_score_mode": "cf_parent_gain",
            "causal_score_tail": "lower",
            "causal_score_weight": 0.05,
        },
        "causal-cf-hurt": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_lagged_causal_graph": True,
            "causal_lags": [1, 2, 4],
            "causal_topk": 5,
            "lambda_causal_mechanism": 0.05,
            "lambda_causal_sparse": 0.001,
            "use_causal_score": True,
            "causal_score_mode": "cf_parent_hurt",
            "causal_score_tail": "upper",
            "causal_score_weight": 0.10,
        },
        "synthetic-aux": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_synthetic_anomaly_aux": True,
            "use_synthetic_anomaly_head": True,
            "lambda_synthetic_anomaly": 0.05,
            "use_synthetic_score": True,
            "synthetic_score_weight": 0.10,
        },
        "synthetic-aux-q75": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_synthetic_anomaly_aux": True,
            "use_synthetic_anomaly_head": True,
            "lambda_synthetic_anomaly": 0.05,
            "use_synthetic_score": True,
            "synthetic_score_weight": 0.10,
            "score_aggregation": "q75",
        },
        "full-event-affinity-lite": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_smoothing_window": 3,
            "score_smoothing_method": "mean",
            "prediction_fill_gap": 2,
            "prediction_min_len": 1,
            "prediction_dilate": 1,
        },
        "full-event-affinity": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_smoothing_window": 5,
            "score_smoothing_method": "mean",
            "prediction_fill_gap": 6,
            "prediction_min_len": 2,
            "prediction_dilate": 2,
        },
        "full-event-affinity-strong": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_smoothing_window": 7,
            "score_smoothing_method": "max",
            "prediction_fill_gap": 10,
            "prediction_min_len": 2,
            "prediction_dilate": 4,
        },
        "full-event-persistence": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_event_persistence_score": True,
            "event_persistence_window": 9,
            "event_persistence_weight": 0.5,
            "score_smoothing_window": 1,
            "prediction_fill_gap": 0,
            "prediction_min_len": 1,
            "prediction_dilate": 0,
        },
        "loss-smoothl1": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "smooth_l1",
            "smooth_l1_beta": 1.0,
        },
        "loss-logcosh": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "log_cosh",
        },
        "loss-mse-mae": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "mse_mae",
            "mse_l1_alpha": 0.7,
        },
        "loss-mse-logcosh": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "mse_log_cosh",
            "mse_robust_alpha": 0.9,
        },
        "loss-mse-smoothl1": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "mse_smooth_l1",
            "mse_robust_alpha": 0.9,
            "smooth_l1_beta": 1.0,
        },
        "loss-diff": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "mse",
            "lambda_temporal_diff_loss": 0.1,
        },
        "loss-smoothl1-diff": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "reconstruction_loss_type": "smooth_l1",
            "smooth_l1_beta": 1.0,
            "lambda_temporal_diff_loss": 0.1,
        },
        "dynamic-temporal": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 1.0,
            "temporal_graph_lr_scale": 0.1,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-gated": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-gated-vqscore": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "use_direct_vq_score": True,
            "vq_score_weight": 0.1,
        },
        "dynamic-temporal-regularized": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_temporal_graph_regularization": True,
            "lambda_temporal_graph_smooth": 0.01,
            "lambda_temporal_graph_locality": 0.001,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-robust": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_robust_reconstruction_loss": True,
            "robust_loss_trim_ratio": 0.05,
            "robust_loss_min_weight": 0.2,
            "robust_loss_warmup_epochs": 10,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-robust-lite": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_robust_reconstruction_loss": True,
            "robust_loss_trim_ratio": 0.02,
            "robust_loss_min_weight": 0.5,
            "robust_loss_warmup_epochs": 10,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-channelnorm": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_score_channel_normalization": True,
            "score_channel_norm_mode": "robust_z",
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-channelnorm-scale": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_score_channel_normalization": True,
            "score_channel_norm_mode": "scale",
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "dynamic-temporal-aff": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "dynamic_temporal_gate_mode": "time_channel",
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_smoothing_window": 5,
            "prediction_fill_gap": 6,
            "prediction_min_len": 3,
            "prediction_dilate": 1,
        },
        "dynamic-temporal-aff-lite": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "dynamic_temporal_gate_mode": "time_channel",
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_smoothing_window": 3,
            "prediction_fill_gap": 2,
            "prediction_min_len": 1,
            "prediction_dilate": 0,
        },
        "dynamic-temporal-aff-peak": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_smoothing_window": 5,
            "score_smoothing_method": "max",
            "prediction_fill_gap": 0,
            "prediction_min_len": 1,
            "prediction_dilate": 0,
        },
        "dynamic-temporal-aff-wide": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_dynamic_temporal_graph": True,
            "dynamic_temporal_residual_init": 0.1,
            "temporal_graph_lr_scale": 1.0,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
            "score_topk_k": 8,
        },
        "no-scorer": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": False,
        },
        "no-boundary": {
            "use_channel_graph": True,
            "use_temporal_graph": True,
            "use_vq_bypass": True,
            "use_multi_scale_scorer": True,
        },
        "reconstruction": {
            "use_channel_graph": False,
            "use_temporal_graph": False,
            "use_vq_bypass": False,
            "use_multi_scale_scorer": False,
        },
    }
    arch_profiles["soft-hierarchical-rca"] = dict(
        arch_profiles["hierarchical-source-rca"],
        rca_hierarchical_mode="soft",
        rca_hierarchical_group_aggregation="max",
        rca_hierarchical_group_topk=0,
        rca_hierarchical_group_boost=0.2,
        rca_hierarchical_outside_penalty=0.0,
    )
    arch_profiles["mechanism-guided-source-rca"] = dict(
        arch_profiles["soft-hierarchical-rca"],
        use_source_gate=True,
        source_gate_init=0.20,
        lambda_source_gate_sparse=0.0002,
        use_mechanism_coupled_decoder=True,
        mechanism_coupling_init=0.18,
        use_channel_masked_modeling=True,
        lambda_channel_masked=0.06,
        channel_mask_interval=4,
        channel_mask_ratio=0.20,
        channel_mask_min_channels=1,
        channel_mask_value="mean",
        lambda_channel_mechanism=0.06,
        use_channel_mechanism_score=True,
        channel_mechanism_score_weight=0.35,
        rca_event_component_normalize=True,
        rca_source_weight=1.0,
        rca_source_base_weight=0.45,
        rca_source_mechanism_weight=0.0,
        rca_source_gate_weight=0.20,
        rca_mechanism_guided_source_weight=0.10,
        rca_propagation_weight=0.10,
        rca_graph_penalty_weight=0.10,
        rca_onset_weight=0.75,
        rca_onset_baseline_window=300,
        rca_onset_z=2.0,
        rca_mechanism_residual_window=300,
        rca_mechanism_residual_weight=0.0,
        rca_hierarchical_group_boost=0.2,
        rca_event_head_ratio=1.0,
        rca_event_head_points=0,
        rca_prediction_key="15",
        rca_event_local_export=True,
        rca_event_local_margin=100,
        rca_split_predicted_events=True,
        rca_split_max_event_len=120,
        rca_split_stride=60,
    )
    arch_profiles["soft-hierarchical-no-normal-prior"] = dict(
        arch_profiles["soft-hierarchical-rca"],
        use_channel_corr_prior=False,
        channel_corr_prior_weight=0.0,
        channel_corr_prior_bias=0.0,
        lambda_channel_prior_align=0.0,
    )
    arch_profiles["mechanism-feedback-rca"] = dict(
        arch_profiles["soft-hierarchical-rca"],
        use_mechanism_residual_feedback=True,
        mechanism_feedback_init=0.03,
        mechanism_feedback_detach=True,
        mechanism_feedback_norm="sample_l1",
        mechanism_feedback_clip=3.0,
        lambda_channel_mechanism=0.08,
        channel_mechanism_score_weight=0.30,
        use_channel_masked_modeling=True,
        lambda_channel_masked=0.05,
        channel_mask_interval=8,
        channel_mask_ratio=0.15,
        channel_mask_min_channels=1,
        channel_mask_value="zero",
        rca_source_base_weight=0.50,
        rca_source_mechanism_weight=0.10,
        rca_source_interaction_weight=2.0,
        rca_graph_penalty_weight=0.10,
        rca_onset_weight=0.75,
    )
    arch_profiles["masked-responsibility-rca"] = dict(
        arch_profiles["mechanism-masked-rca"],
        use_synthetic_rca_head=True,
        use_synthetic_rca_loss=False,
        lambda_synthetic_rca=0.0,
        lambda_masked_rca_head=0.05,
        masked_rca_bce_weight=1.0,
        masked_rca_rank_weight=1.0,
        synthetic_rca_margin=0.15,
        synthetic_rca_topk=8,
        rca_synthetic_weight=0.35,
        rca_source_base_weight=0.45,
        rca_source_mechanism_weight=0.0,
        rca_source_interaction_weight=1.5,
        rca_graph_penalty_weight=0.10,
        rca_onset_weight=0.75,
    )
    arch_profiles["interventional-graph-source-rca"] = dict(
        arch_profiles["soft-hierarchical-rca"],
        use_channel_corr_prior=False,
        channel_corr_prior_weight=0.0,
        channel_corr_prior_bias=0.0,
        lambda_channel_prior_align=0.0,
        use_source_gate=True,
        source_gate_init=0.20,
        lambda_source_gate_sparse=0.0002,
        use_channel_masked_modeling=True,
        lambda_channel_masked=0.08,
        channel_mask_interval=4,
        channel_mask_ratio=0.20,
        channel_mask_min_channels=1,
        channel_mask_value="mean",
        use_interventional_channel_masking=True,
        lambda_interventional_source_bce=0.02,
        lambda_interventional_source_rank=0.03,
        lambda_interventional_graph_support=0.01,
        use_mechanism_coupled_decoder=True,
        mechanism_coupling_init=0.20,
        lambda_channel_mechanism=0.05,
        use_channel_mechanism_score=True,
        channel_mechanism_score_weight=0.35,
        rca_source_base_weight=0.45,
        rca_source_gate_weight=0.20,
        rca_source_interaction_weight=2.0,
        rca_graph_penalty_weight=0.10,
        rca_onset_weight=0.75,
        rca_hierarchical_group_boost=0.2,
    )
    arch_profiles["source-propagation-trained-rca"] = dict(
        arch_profiles["interventional-graph-source-rca"],
        use_source_effect_synthetic=True,
        lambda_source_effect=0.03,
        source_effect_interval=12,
        source_effect_min_len=10,
        source_effect_max_len=40,
        source_effect_min_roots=1,
        source_effect_max_roots=2,
        source_effect_neighbor_topk=3,
        source_effect_strength=0.35,
        source_effect_delay_max=8,
        source_effect_bce_weight=0.50,
        source_effect_rank_weight=0.75,
        source_effect_effect_rank_weight=0.75,
        source_effect_onset_rank_weight=1.00,
        source_effect_margin=0.15,
        source_effect_onset_margin=0.20,
        rca_source_base_weight=0.45,
        rca_source_gate_weight=0.25,
        rca_source_mechanism_weight=0.0,
        rca_mechanism_guided_source_weight=0.0,
        rca_propagation_weight=0.0,
        rca_graph_penalty_weight=0.10,
        rca_onset_weight=0.75,
    )
    arch_profiles["source-propagation-prior-rca"] = dict(
        arch_profiles["source-propagation-trained-rca"],
        source_effect_use_channel_prior=True,
        source_effect_prior_topk=5,
        use_channel_corr_prior=False,
        channel_corr_prior_weight=0.0,
        channel_corr_prior_bias=0.0,
        lambda_channel_prior_align=0.0,
    )
    arch_profiles["source-innovation-rca"] = dict(
        arch_profiles["source-propagation-prior-rca"],
        rca_source_innovation_weight=0.25,
        rca_source_innovation_neighbor_weight=1.0,
    )
    arch_profiles["temporal-innovation-rca"] = dict(
        arch_profiles["source-propagation-prior-rca"],
        rca_source_innovation_weight=0.15,
        rca_source_innovation_mode="onset_directional",
        rca_source_innovation_neighbor_weight=0.50,
        rca_source_innovation_lead_points=1,
    )
    arch_profiles["source-bottleneck-rca"] = dict(
        arch_profiles["source-propagation-prior-rca"],
        lambda_source_effect=0.04,
        source_effect_interval=8,
        source_effect_bce_weight=0.60,
        source_effect_rank_weight=0.85,
        source_effect_effect_rank_weight=0.90,
        source_effect_onset_rank_weight=1.25,
        lambda_source_bottleneck=0.025,
        source_bottleneck_bce_weight=1.0,
        source_bottleneck_rank_weight=0.75,
        source_bottleneck_effect_suppress_weight=0.50,
        rca_source_base_weight=0.40,
        rca_source_gate_weight=0.35,
        rca_source_interaction_weight=2.0,
        rca_onset_weight=0.85,
        rca_source_innovation_weight=0.0,
    )
    arch_profiles["source-bottleneck-balanced-rca"] = dict(
        arch_profiles["source-bottleneck-rca"],
        rca_source_base_weight=1.0,
        rca_source_score_weight=0.0,
        rca_source_gate_weight=0.50,
        rca_onset_weight=0.75,
        rca_source_interaction_weight=1.0,
        rca_graph_penalty_weight=0.05,
        rca_source_innovation_weight=0.0,
        rca_hierarchical_mode="annotate",
        rca_hierarchical_group_boost=0.0,
        rca_hierarchical_outside_penalty=0.0,
    )
    arch_profiles["source-bottleneck-onset-balanced-rca"] = dict(
        arch_profiles["source-bottleneck-balanced-rca"],
        rca_event_head_ratio=0.30,
        rca_event_head_points=30,
    )
    arch_profiles["source-bottleneck-onset-wide-rca"] = dict(
        arch_profiles["source-bottleneck-balanced-rca"],
        rca_event_head_ratio=0.50,
        rca_event_head_points=60,
    )
    arch_profiles["source-bottleneck-aligned-onset-rca"] = dict(
        arch_profiles["source-bottleneck-balanced-rca"],
        rca_align_event_onset=True,
        rca_align_event_onset_baseline_window=300,
        rca_align_event_onset_z=2.0,
        rca_align_event_onset_quantile=0.90,
        rca_event_head_ratio=0.30,
        rca_event_head_points=30,
    )
    arch_profiles["source-bottleneck-aligned-wide-rca"] = dict(
        arch_profiles["source-bottleneck-balanced-rca"],
        rca_align_event_onset=True,
        rca_align_event_onset_baseline_window=300,
        rca_align_event_onset_z=2.0,
        rca_align_event_onset_quantile=0.90,
        rca_event_head_ratio=0.50,
        rca_event_head_points=60,
    )
    arch_profiles["source-bottleneck-contrast-balanced-rca"] = dict(
        arch_profiles["source-bottleneck-balanced-rca"],
        rca_contrast_window=300,
        rca_contrast_weight=0.50,
    )
    arch_profiles["source-bottleneck-specificity-rca"] = dict(
        arch_profiles["source-bottleneck-balanced-rca"],
        rca_event_specificity_weight=3.0,
        rca_event_specificity_top_k=1,
        rca_event_specificity_threshold=0.35,
        rca_event_specificity_min_events=20,
    )
    arch_profiles["source-bottleneck-topk-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_topk_rerank=True,
        rca_topk_rerank_k=5,
        rca_topk_rerank_original_weight=1.0,
        rca_topk_rerank_group_weight=0.2,
        rca_topk_rerank_onset_weight=0.2,
        rca_topk_rerank_source_gate_weight=0.0,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_topk_rerank_graph_penalty_weight=0.0,
    )
    arch_profiles["source-bottleneck-root-score-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_root_score_head=True,
        lambda_source_effect_root_score=2.0,
        root_score_bce_weight=1.0,
        root_score_rank_weight=1.0,
        root_score_effect_suppress_weight=0.5,
        rca_root_score_weight=0.80,
        rca_source_base_weight=0.70,
        rca_source_gate_weight=0.35,
        rca_onset_weight=0.60,
        rca_source_interaction_weight=0.75,
        rca_topk_rerank=False,
    )
    arch_profiles["source-bottleneck-root-score-w03-rca"] = dict(
        arch_profiles["source-bottleneck-root-score-rca"],
        rca_root_score_weight=0.30,
    )
    arch_profiles["source-bottleneck-root-response-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_root_score_head=True,
        root_score_head_mode="root_response",
        root_score_detach_features=False,
        root_response_penalty_init=1.0,
        root_response_confidence_discount=0.75,
        lambda_source_effect_root_score=2.0,
        root_score_bce_weight=1.0,
        root_score_rank_weight=1.0,
        root_score_effect_suppress_weight=0.75,
        root_response_source_branch_weight=0.50,
        root_response_response_branch_weight=0.35,
        root_response_response_suppress_weight=0.50,
        rca_root_score_weight=0.60,
        rca_root_score_signal="prob",
        rca_root_score_pooling="mean",
        rca_root_score_head_ratio=0.30,
        rca_root_score_head_points=30,
        rca_root_score_top_quantile=0.80,
        rca_source_base_weight=0.70,
        rca_source_gate_weight=0.35,
        rca_onset_weight=0.60,
        rca_source_interaction_weight=0.75,
        rca_topk_rerank=False,
    )
    arch_profiles["source-bottleneck-root-response-hardpos-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-rca"],
        synthetic_rca_positive_aggregation="bottomk",
        synthetic_rca_positive_topk=1,
    )
    arch_profiles["source-bottleneck-root-response-softpos-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-rca"],
        synthetic_rca_positive_aggregation="blend",
        synthetic_rca_positive_topk=1,
        synthetic_rca_positive_min_weight=0.35,
    )
    arch_profiles["source-bottleneck-root-response-top20-calibrated-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-rca"],
        rca_topk_rerank=True,
        rca_topk_rerank_k=20,
        rca_topk_rerank_original_weight=1.0,
        rca_topk_rerank_base_weight=0.45,
        rca_topk_rerank_group_weight=0.0,
        rca_topk_rerank_onset_weight=0.20,
        rca_topk_rerank_source_gate_weight=0.0,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_topk_rerank_graph_penalty_weight=0.05,
    )
    arch_profiles["source-bottleneck-root-response-innovation-top20-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-top20-calibrated-rca"],
        root_response_use_innovation_split=True,
    )
    arch_profiles["source-bottleneck-pairwise-root-response-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-top20-calibrated-rca"],
        use_pairwise_root_response_head=True,
        pairwise_root_response_rank=8,
        pairwise_root_response_graph_weight=0.70,
        pairwise_root_response_reward_init=0.25,
        pairwise_root_response_penalty_init=0.35,
        pairwise_root_response_logit_weight=1.0,
        lambda_pairwise_root_response=0.15,
        root_response_use_innovation_split=True,
        rca_root_score_weight=0.70,
        rca_source_interaction_weight=0.85,
    )
    arch_profiles["source-bottleneck-pairwise-aux-root-response-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-top20-calibrated-rca"],
        use_pairwise_root_response_head=True,
        pairwise_root_response_rank=8,
        pairwise_root_response_graph_weight=0.70,
        pairwise_root_response_reward_init=0.05,
        pairwise_root_response_penalty_init=0.10,
        pairwise_root_response_logit_weight=0.0,
        lambda_pairwise_root_response=0.10,
        root_response_use_innovation_split=True,
        rca_root_score_weight=0.60,
        rca_source_interaction_weight=0.75,
    )
    arch_profiles["source-bottleneck-event-responsibility-top20-rca"] = dict(
        arch_profiles["source-bottleneck-root-response-top20-calibrated-rca"],
        use_event_responsibility_head=True,
        event_responsibility_hidden=16,
        event_responsibility_detach_features=True,
        lambda_event_responsibility=0.035,
        event_responsibility_ce_weight=1.0,
        event_responsibility_rank_weight=0.75,
        event_responsibility_effect_suppress_weight=0.25,
        event_responsibility_entropy_weight=0.0,
        rca_event_responsibility_weight=0.30,
    )
    arch_profiles["source-bottleneck-event-responsibility-head-top20-rca"] = dict(
        arch_profiles["source-bottleneck-event-responsibility-top20-rca"],
        rca_event_responsibility_pooling="head_mean",
        rca_event_responsibility_head_ratio=0.30,
        rca_event_responsibility_head_points=30,
        rca_event_responsibility_top_quantile=0.80,
    )
    arch_profiles["source-bottleneck-event-evidence-top20-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-event-responsibility-head-top20-rca"],
        rca_topk_rerank=True,
        rca_topk_rerank_k=20,
        rca_topk_rerank_original_weight=0.0,
        rca_topk_rerank_base_weight=0.45,
        rca_topk_rerank_group_weight=0.0,
        rca_topk_rerank_onset_weight=0.25,
        rca_topk_rerank_source_gate_weight=1.0,
        rca_topk_rerank_mechanism_residual_weight=0.25,
        rca_topk_rerank_source_interaction_weight=0.25,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_component_scope="event",
        rca_event_specificity_source_guard_weight=0.50,
        rca_event_specificity_source_guard_floor=0.0,
    )
    arch_profiles["source-bottleneck-event-balanced-evidence-top20-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-event-responsibility-head-top20-rca"],
        rca_topk_rerank=True,
        rca_topk_rerank_k=20,
        rca_topk_rerank_original_weight=0.10,
        rca_topk_rerank_base_weight=0.45,
        rca_topk_rerank_group_weight=0.0,
        rca_topk_rerank_onset_weight=0.25,
        rca_topk_rerank_source_gate_weight=0.10,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_topk_rerank_source_interaction_weight=0.25,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_component_scope="event",
        rca_event_specificity_source_guard_weight=0.50,
        rca_event_specificity_source_guard_floor=0.0,
    )
    arch_profiles["source-bottleneck-event-swat-priority-evidence-top20-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-event-responsibility-head-top20-rca"],
        rca_topk_rerank=True,
        rca_topk_rerank_k=20,
        rca_topk_rerank_original_weight=0.0,
        rca_topk_rerank_base_weight=0.70,
        rca_topk_rerank_group_weight=0.0,
        rca_topk_rerank_onset_weight=0.0,
        rca_topk_rerank_source_gate_weight=0.35,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_topk_rerank_source_interaction_weight=0.50,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_component_scope="event",
        rca_event_specificity_source_guard_weight=0.50,
        rca_event_specificity_source_guard_floor=0.0,
    )
    arch_profiles["source-bottleneck-event-top1-coverage-evidence-top20-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-event-responsibility-head-top20-rca"],
        rca_topk_rerank=True,
        rca_topk_rerank_k=20,
        rca_topk_rerank_original_weight=0.0,
        rca_topk_rerank_base_weight=0.45,
        rca_topk_rerank_group_weight=0.0,
        rca_topk_rerank_onset_weight=0.25,
        rca_topk_rerank_source_gate_weight=1.0,
        rca_topk_rerank_mechanism_residual_weight=0.25,
        rca_topk_rerank_source_interaction_weight=0.25,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_component_scope="event",
        rca_topk_rerank_keep_primary_top_k=0,
        rca_topk_rerank_fill_secondary_top_k=0,
        rca_topk_rerank_secondary_original_weight=0.0,
        rca_topk_rerank_secondary_base_weight=0.0,
        rca_topk_rerank_secondary_group_weight=0.0,
        rca_topk_rerank_secondary_onset_weight=0.0,
        rca_topk_rerank_secondary_source_gate_weight=0.0,
        rca_topk_rerank_secondary_mechanism_residual_weight=0.0,
        rca_topk_rerank_secondary_source_interaction_weight=0.0,
        rca_topk_rerank_secondary_graph_penalty_weight=0.0,
        rca_event_specificity_source_guard_weight=0.50,
        rca_event_specificity_source_guard_floor=0.0,
        rca_event_specificity_keep_top_k=1,
        rca_event_specificity_fill_secondary_top_k=3,
        rca_event_specificity_secondary_base_weight=0.70,
        rca_event_specificity_secondary_onset_weight=0.0,
        rca_event_specificity_secondary_source_gate_weight=0.0,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_source_interaction_weight=0.50,
    )
    arch_profiles["source-bottleneck-evidence-fusion-head-top1-coverage-rca"] = dict(
        arch_profiles["source-bottleneck-event-top1-coverage-evidence-top20-rerank-rca"],
        use_evidence_fusion_head=True,
        evidence_fusion_hidden=16,
        evidence_fusion_detach_inputs=False,
        lambda_evidence_fusion=0.20,
        evidence_fusion_bce_weight=1.0,
        evidence_fusion_rank_weight=1.0,
        evidence_fusion_effect_suppress_weight=0.25,
        evidence_fusion_entropy_weight=0.0,
        rca_evidence_fusion_weight=0.25,
        rca_evidence_fusion_pooling="head_mean",
        rca_evidence_fusion_head_ratio=0.30,
        rca_evidence_fusion_head_points=30,
        rca_evidence_fusion_top_quantile=0.80,
    )
    arch_profiles["source-bottleneck-evidence-fusion-head-detached-rca"] = dict(
        arch_profiles["source-bottleneck-event-top1-coverage-evidence-top20-rerank-rca"],
        use_evidence_fusion_head=True,
        evidence_fusion_hidden=16,
        evidence_fusion_detach_inputs=True,
        lambda_evidence_fusion=0.05,
        evidence_fusion_bce_weight=1.0,
        evidence_fusion_rank_weight=1.0,
        evidence_fusion_effect_suppress_weight=0.50,
        evidence_fusion_entropy_weight=0.0,
        rca_evidence_fusion_weight=0.08,
        rca_evidence_fusion_pooling="head_mean",
        rca_evidence_fusion_head_ratio=0.30,
        rca_evidence_fusion_head_points=30,
        rca_evidence_fusion_top_quantile=0.80,
    )
    arch_profiles["source-bottleneck-evidence-fusion-secondary-fill-rca"] = dict(
        arch_profiles["source-bottleneck-event-top1-coverage-evidence-top20-rerank-rca"],
        use_evidence_fusion_head=True,
        evidence_fusion_hidden=16,
        evidence_fusion_detach_inputs=True,
        lambda_evidence_fusion=0.05,
        evidence_fusion_bce_weight=1.0,
        evidence_fusion_rank_weight=1.0,
        evidence_fusion_effect_suppress_weight=0.50,
        evidence_fusion_entropy_weight=0.0,
        rca_evidence_fusion_weight=0.0,
        rca_event_specificity_secondary_evidence_fusion_weight=0.25,
    )
    arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-head-detached-rca"],
        rca_event_specificity_keep_top_k=1,
        rca_event_specificity_fill_secondary_top_k=5,
        rca_event_specificity_secondary_base_weight=0.45,
        rca_event_specificity_secondary_onset_weight=0.25,
        rca_event_specificity_secondary_source_gate_weight=1.0,
        rca_event_specificity_secondary_mechanism_residual_weight=0.25,
        rca_event_specificity_secondary_source_interaction_weight=0.25,
        rca_event_specificity_secondary_evidence_fusion_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-score-responsibility-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        rca_export_score_distribution="positive_l1",
        rca_export_score_distribution_top_m=20,
        rca_export_score_distribution_power=2.0,
        rca_export_score_distribution_tau=1.0,
        rca_export_responsibility_score=True,
        rca_export_responsibility_base_weight=0.35,
        rca_export_responsibility_source_gate_weight=1.0,
        rca_export_responsibility_onset_weight=0.35,
        rca_export_responsibility_mechanism_residual_weight=0.25,
        rca_export_responsibility_response_suppressor_low_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-score-responsibility-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-stable-score-responsibility-rca"],
        rca_responsibility_rerank=True,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-only-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        channel_graph_evidence_only=True,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-reliable-score-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-only-rca"],
        use_channel_graph_reliability=True,
        rca_export_score_distribution="positive_l1",
        rca_export_score_distribution_top_m=20,
        rca_export_score_distribution_power=2.0,
        rca_export_score_distribution_tau=1.0,
        rca_export_responsibility_score=True,
        rca_export_responsibility_base_weight=0.35,
        rca_export_responsibility_source_gate_weight=1.0,
        rca_export_responsibility_onset_weight=0.35,
        rca_export_responsibility_mechanism_residual_weight=0.25,
        rca_export_responsibility_response_suppressor_low_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-source-gate-guard-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-reliable-score-rca"],
        rca_source_gate_evidence_guard_weight=0.65,
        rca_source_gate_evidence_guard_floor=0.15,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-onset-source-gate-guard-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-source-gate-guard-rca"],
        use_onset_aware_source_gate=True,
        source_gate_onset_window=8,
        source_gate_onset_weight=0.35,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-onset-input-source-gate-guard-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-source-gate-guard-rca"],
        use_onset_aware_source_gate=True,
        source_gate_onset_window=8,
        source_gate_onset_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-effect-margin-guard-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-source-gate-guard-rca"],
        source_bottleneck_effect_suppress_weight=0.75,
        source_bottleneck_effect_margin_weight=0.75,
        source_bottleneck_effect_margin=0.10,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-event-route-guard-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-source-gate-guard-rca"],
        use_event_route_head=True,
        event_route_hidden=16,
        event_route_init=0.60,
        event_route_detach_inputs=True,
        event_route_response_penalty=0.0,
        lambda_event_route=0.06,
        event_route_rank_weight=1.0,
        event_route_effect_suppress_weight=0.20,
        event_route_pairwise_effect_weight=0.35,
        event_route_pairwise_effect_margin=0.15,
        event_route_alpha_weight=0.15,
        rca_event_route_weight=0.18,
        rca_export_responsibility_ranking_weight=0.60,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-event-route-response-guard-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-event-route-guard-rca"],
        event_route_response_penalty=0.65,
        lambda_event_route=0.08,
        event_route_effect_suppress_weight=0.35,
        event_route_pairwise_effect_weight=0.50,
        event_route_alpha_weight=0.10,
        rca_event_route_weight=0.16,
        rca_export_responsibility_ranking_weight=0.45,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-event-route-response-root-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-event-route-response-guard-rca"],
        rca_topk_rerank_base_weight=0.55,
        rca_topk_rerank_onset_weight=0.30,
        rca_topk_rerank_source_gate_weight=0.75,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_base_weight=0.55,
        rca_event_specificity_secondary_onset_weight=0.30,
        rca_event_specificity_secondary_source_gate_weight=0.75,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
        rca_export_responsibility_mechanism_residual_weight=0.0,
    )
    arch_profiles["source-bottleneck-causal-innovation-response-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-event-route-response-root-rerank-rca"],
        # Keep the detector unchanged. The causal branch is an RCA-only
        # predictor: delayed history forms source innovation, while the gain
        # from cross-channel parents is response evidence.
        use_lagged_causal_graph=True,
        causal_lags=[1, 3, 6, 12],
        causal_topk=5,
        causal_detach_backbone=True,
        lambda_causal_mechanism=0.05,
        lambda_causal_sparse=0.001,
        use_causal_response_evidence=True,
        lambda_causal_response=0.50,
        causal_response_margin=0.10,
        debug_loss_breakdown=True,
        debug_loss_max_batches=1,
    )
    arch_profiles["source-propagation-strict-cross-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-only-rca"],
        # Keep detection on the stable evidence-only path. RCA is produced by
        # the strict delayed mechanism and SPS-trained responsibility head.
        channel_graph_evidence_only=True,
        use_channel_graph_reliability=False,
        use_lagged_causal_graph=False,
        use_causal_response_evidence=False,
        use_mechanism_predictive_head=False,
        use_mechanism_coupled_decoder=False,
        lambda_channel_mechanism=0.0,
        use_channel_mechanism_score=False,
        use_channel_masked_modeling=False,
        lambda_channel_masked=0.0,
        use_interventional_channel_masking=False,
        lambda_interventional_source_bce=0.0,
        lambda_interventional_source_rank=0.0,
        lambda_interventional_graph_support=0.0,
        lambda_source_bottleneck=0.0,
        lambda_source_gate_sparse=0.0,
        use_source_gate=False,
        use_onset_aware_source_gate=False,
        use_root_score_head=False,
        use_pairwise_root_response_head=False,
        use_event_responsibility_head=False,
        use_event_route_head=False,
        use_response_suppressor_head=False,
        use_evidence_fusion_head=False,
        use_source_interaction_head=False,
        use_source_consistency_head=False,
        use_strict_cross_mechanism=True,
        strict_cross_lags=[1, 3, 6, 12],
        strict_cross_topk=5,
        strict_cross_detach_backbone=True,
        strict_cross_use_channel_prior=True,
        use_sps_role_head=True,
        sps_role_hidden=16,
        sps_role_detach_features=False,
        sps_lr_scale=10.0,
        lambda_strict_cross_mechanism=0.05,
        lambda_strict_cross_sparse=0.001,
        use_source_effect_synthetic=True,
        source_effect_interval=4,
        lambda_source_effect=0.50,
        source_effect_bce_weight=0.0,
        source_effect_rank_weight=0.0,
        source_effect_effect_rank_weight=0.0,
        source_effect_onset_rank_weight=0.0,
        source_effect_specificity_weight=0.0,
        source_effect_graph_alignment_weight=0.0,
        source_effect_use_channel_prior=False,
        lambda_source_effect_rca_head=0.0,
        lambda_source_effect_root_score=0.0,
        lambda_pairwise_root_response=0.0,
        lambda_event_responsibility=0.0,
        lambda_response_suppressor=0.0,
        lambda_evidence_fusion=0.0,
        lambda_source_interaction_head=0.0,
        lambda_source_consistency_head=0.0,
        lambda_event_route=0.0,
        lambda_causal_response=0.0,
        use_sps_teacher=True,
        sps_teacher_lags=[1, 3, 6, 12],
        sps_teacher_topk=3,
        sps_teacher_ridge=0.10,
        sps_teacher_max_samples=30000,
        sps_teacher_max_gain=0.65,
        sps_teacher_response_ratio=0.15,
        sps_teacher_onset_len=2,
        lambda_sps_source=1.0,
        lambda_sps_response=0.50,
        lambda_sps_separation=0.50,
        sps_separation_margin=0.10,
        # Root responsibility is the only learned RCA contribution at export.
        rca_use_source_propagation=False,
        rca_graph_weight=0.0,
        rca_graph_penalty_weight=0.0,
        rca_source_weight=0.0,
        rca_source_base_weight=0.0,
        rca_source_interaction_weight=0.0,
        rca_onset_weight=0.0,
        rca_event_specificity_weight=0.0,
        rca_event_specificity_source_guard_weight=0.0,
        rca_event_specificity_keep_top_k=0,
        rca_event_specificity_fill_secondary_top_k=0,
        rca_topk_rerank=False,
        rca_source_gate_weight=0.0,
        rca_root_score_weight=0.0,
        rca_event_route_weight=0.0,
        rca_response_suppressor_weight=0.0,
        rca_evidence_fusion_weight=0.0,
        rca_source_interaction_head_weight=0.0,
        rca_source_consistency_head_weight=0.0,
        rca_event_responsibility_weight=1.0,
        rca_topk_rerank_base_weight=0.0,
        rca_topk_rerank_onset_weight=0.0,
        rca_topk_rerank_source_gate_weight=0.0,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_base_weight=0.0,
        rca_event_specificity_secondary_onset_weight=0.0,
        rca_event_specificity_secondary_source_gate_weight=0.0,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
        debug_loss_breakdown=True,
        debug_loss_max_batches=1,
    )
    arch_profiles["source-propagation-residual-sps-rca"] = dict(
        arch_profiles["source-propagation-strict-cross-mechanism-rca"],
        # Generate source-response supervision in the residual mechanism
        # domain and export the learned role head directly.
        use_sps_residual_synthetic=True,
        rca_event_base_weight=0.0,
    )
    arch_profiles["source-propagation-bounded-sps-fusion-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-reliable-score-rca"],
        # Preserve the validated source-propagation path.  SPS supplies only a
        # zero-initialized bounded correction to its learned responsibility
        # logits and cannot replace the established ranking.
        use_strict_cross_mechanism=True,
        strict_cross_lags=[1, 3, 6, 12],
        strict_cross_topk=5,
        strict_cross_detach_backbone=True,
        strict_cross_use_channel_prior=True,
        lambda_strict_cross_mechanism=0.05,
        lambda_strict_cross_sparse=0.001,
        use_sps_role_head=True,
        sps_role_hidden=16,
        sps_role_detach_features=False,
        sps_lr_scale=10.0,
        use_bounded_sps_fusion=True,
        bounded_sps_max_correction=0.25,
        use_sps_teacher=True,
        use_sps_residual_synthetic=True,
        sps_teacher_lags=[1, 3, 6, 12],
        sps_teacher_topk=3,
        sps_teacher_ridge=0.10,
        sps_teacher_max_samples=30000,
        sps_teacher_max_gain=0.65,
        sps_teacher_response_ratio=0.15,
        sps_teacher_onset_len=2,
        lambda_sps_source=0.0,
        lambda_sps_response=0.0,
        lambda_sps_separation=0.0,
        lambda_residual_sps=0.05,
        source_effect_aux_batch_size=64,
        channel_mask_aux_batch_size=64,
        residual_sps_aux_batch_size=64,
        residual_sps_source_weight=1.0,
        residual_sps_response_weight=0.50,
        residual_sps_separation_weight=0.50,
        sps_separation_margin=0.10,
        debug_loss_breakdown=True,
        debug_loss_max_batches=1,
    )
    arch_profiles["source-bottleneck-dual-expert-representation-fusion-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-event-route-response-root-rerank-rca"],
        # Retain an independent temporal representation and learn when the
        # graph-conditioned representation should be trusted before decoding.
        channel_graph_evidence_only=False,
        use_dual_expert_fusion=True,
        dual_expert_hidden=16,
        dual_expert_graph_init=0.50,
        dual_expert_detach_inputs=True,
        lambda_dual_expert_selection=0.50,
        dual_expert_selection_temperature=0.25,
        dual_expert_balance_weight=0.01,
        dual_expert_min_usage=0.10,
        dual_expert_propagated_event_prob=0.50,
        dual_expert_mode_target_weight=0.70,
        debug_loss_breakdown=True,
        debug_loss_max_batches=1,
    )
    arch_profiles["source-bottleneck-source-expert-representation-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-event-route-response-root-rerank-rca"],
        # Detection keeps the evidence-only backbone; graph propagation is used
        # exclusively by the learned source-attribution representation branch.
        use_dual_expert_fusion=False,
        channel_graph_evidence_only=True,
        use_source_expert_branch=True,
        source_expert_hidden=16,
        source_expert_graph_init=0.50,
        source_expert_detach_gate_inputs=True,
        source_expert_gradient_checkpoint=True,
        source_expert_gate_scope="event",
        lambda_source_expert=0.60,
        lambda_source_expert_gate=0.80,
        source_expert_effect_suppress_weight=0.25,
        source_expert_pairwise_effect_weight=0.50,
        source_expert_pairwise_effect_margin=0.15,
        dual_expert_propagated_event_prob=0.50,
        debug_loss_breakdown=True,
        debug_loss_max_batches=1,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-learned-responsibility-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-reliable-score-rca"],
        rca_responsibility_rerank=True,
        rca_export_responsibility_base_weight=0.35,
        rca_export_responsibility_source_gate_weight=0.0,
        rca_export_responsibility_onset_weight=0.25,
        rca_export_responsibility_mechanism_residual_weight=0.0,
        rca_export_responsibility_root_weight=0.25,
        rca_export_responsibility_event_weight=0.20,
        rca_export_responsibility_evidence_fusion_weight=0.20,
        rca_export_responsibility_source_interaction_weight=0.0,
        rca_export_responsibility_graph_low_weight=0.0,
        rca_export_responsibility_response_suppressor_low_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-graph-evidence-adaptive-learned-decoder-rca"] = dict(
        arch_profiles["source-bottleneck-stable-graph-evidence-source-gate-guard-rca"],
        rca_adaptive_evidence_rerank=True,
        rca_adaptive_evidence_top_k=20,
        rca_adaptive_evidence_keep_top_k=1,
        rca_adaptive_evidence_fill_top_k=5,
        rca_adaptive_evidence_gate="source_top_fallback_rank",
        rca_adaptive_evidence_source_old_rank_threshold=3,
        rca_adaptive_evidence_source_base_weight=0.45,
        rca_adaptive_evidence_source_onset_weight=0.25,
        rca_adaptive_evidence_source_gate_weight=1.0,
        rca_adaptive_evidence_source_mechanism_residual_weight=0.25,
        rca_adaptive_evidence_source_interaction_weight=0.25,
        rca_adaptive_evidence_fallback_original_weight=0.0,
        rca_adaptive_evidence_fallback_base_weight=0.35,
        rca_adaptive_evidence_fallback_onset_weight=0.25,
        rca_adaptive_evidence_fallback_mechanism_residual_weight=0.0,
        rca_adaptive_evidence_fallback_root_weight=0.25,
        rca_adaptive_evidence_fallback_event_weight=0.20,
        rca_adaptive_evidence_fallback_evidence_fusion_weight=0.20,
        rca_adaptive_evidence_fallback_source_interaction_weight=0.0,
        rca_adaptive_evidence_fallback_source_consistency_weight=0.0,
        rca_adaptive_evidence_fallback_graph_low_weight=0.0,
        rca_adaptive_evidence_fallback_response_suppressor_low_weight=0.0,
    )
    arch_profiles["source-bottleneck-directed-sparse-graph-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        channel_graph_type="directed_sparse",
        directed_graph_parent_topk=5,
        directed_graph_embedding_dim=8,
        channel_graph_evidence_only=False,
        use_mechanism_predictive_head=False,
        use_mechanism_coupled_decoder=False,
        use_channel_masked_modeling=False,
        lambda_channel_masked=0.0,
        lambda_channel_mechanism=0.05,
    )
    arch_profiles["source-bottleneck-directed-graph-evidence-only-rca"] = dict(
        arch_profiles["source-bottleneck-directed-sparse-graph-rca"],
        channel_graph_evidence_only=True,
    )
    arch_profiles["source-bottleneck-directed-graph-supervised-rca"] = dict(
        arch_profiles["source-bottleneck-directed-graph-evidence-only-rca"],
        lambda_channel_mechanism=0.01,
        use_channel_mechanism_score=False,
        channel_mechanism_score_weight=0.0,
        source_effect_graph_alignment_weight=1.0,
        source_effect_graph_reverse_weight=0.5,
    )
    arch_profiles["source-bottleneck-directed-graph-calibrated-rca"] = dict(
        arch_profiles["source-bottleneck-directed-graph-supervised-rca"],
        use_channel_graph_reliability=True,
    )
    arch_profiles["source-bottleneck-directed-graph-signed-calibrated-rca"] = dict(
        arch_profiles["source-bottleneck-directed-graph-supervised-rca"],
        directed_graph_use_signed_transfer=True,
        directed_graph_transfer_rank=8,
        directed_graph_transfer_scale=2.0,
        use_channel_graph_reliability=True,
        channel_graph_lr_scale=1.0,
        lambda_channel_mechanism=0.05,
    )
    arch_profiles["source-bottleneck-prior-supported-transfer-rca"] = dict(
        arch_profiles["source-bottleneck-directed-graph-supervised-rca"],
        use_channel_corr_prior=True,
        channel_corr_prior_selection_axis="target",
        channel_corr_prior_topk=5,
        channel_corr_prior_weight=1.0,
        channel_corr_prior_bias=0.0,
        lambda_channel_prior_align=0.0,
        directed_graph_use_signed_transfer=True,
        directed_graph_transfer_mode="full",
        directed_graph_transfer_scale=4.0,
        use_channel_graph_reliability=True,
        channel_graph_lr_scale=1.0,
        lambda_locality_l1=0.0,
        lambda_channel_mechanism=0.05,
        source_effect_graph_alignment_weight=0.0,
        source_effect_graph_reverse_weight=0.0,
    )
    arch_profiles["source-bottleneck-response-only-graph-rca"] = dict(
        arch_profiles["source-bottleneck-prior-supported-transfer-rca"],
        channel_graph_role="response_only",
        rca_mechanism_residual_weight=0.0,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_topk_rerank_secondary_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
    )
    arch_profiles["source-bottleneck-response-only-multilag-graph-rca"] = dict(
        arch_profiles["source-bottleneck-response-only-graph-rca"],
        use_multilag_graph_propagation=True,
        graph_propagation_lags=[0, 1, 2, 4, 8],
    )
    arch_profiles["source-bottleneck-response-aligned-multilag-graph-rca"] = dict(
        arch_profiles["source-bottleneck-response-only-multilag-graph-rca"],
        graph_propagation_lag_mode="alignment",
        graph_propagation_alignment_temperature=0.2,
    )
    arch_profiles["source-bottleneck-response-graph-fusion-rca"] = dict(
        arch_profiles["source-bottleneck-response-aligned-multilag-graph-rca"],
        evidence_fusion_use_graph_response=True,
        evidence_fusion_detach_inputs=False,
        evidence_fusion_pairwise_effect_weight=0.15,
        evidence_fusion_pairwise_effect_margin=0.15,
        rca_evidence_fusion_weight=0.10,
        rca_event_specificity_secondary_evidence_fusion_weight=0.10,
    )
    arch_profiles["source-bottleneck-response-graph-fusion-detached-rca"] = dict(
        arch_profiles["source-bottleneck-response-aligned-multilag-graph-rca"],
        evidence_fusion_use_graph_response=True,
        evidence_fusion_detach_inputs=True,
        evidence_fusion_pairwise_effect_weight=0.15,
        evidence_fusion_pairwise_effect_margin=0.15,
        rca_evidence_fusion_weight=0.08,
        rca_event_specificity_secondary_evidence_fusion_weight=0.0,
    )
    arch_profiles["source-bottleneck-response-suppressor-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        use_response_suppressor_head=True,
        response_suppressor_hidden=16,
        response_suppressor_detach_inputs=True,
        lambda_response_suppressor=0.05,
        response_suppressor_bce_weight=1.0,
        response_suppressor_rank_weight=0.75,
        response_suppressor_source_leak_weight=0.75,
        rca_graph_penalty_weight=0.0,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_secondary_graph_penalty_weight=0.0,
        rca_response_suppressor_weight=0.20,
        rca_response_suppressor_source_guard_mode="source_onset",
        rca_response_suppressor_source_guard_floor=0.10,
    )
    arch_profiles["source-bottleneck-response-suppressor-score-responsibility-rca"] = dict(
        arch_profiles["source-bottleneck-response-suppressor-rca"],
        rca_export_score_distribution="positive_l1",
        rca_export_score_distribution_top_m=20,
        rca_export_score_distribution_power=2.0,
        rca_export_score_distribution_tau=1.0,
        rca_export_responsibility_score=True,
        rca_export_responsibility_base_weight=0.45,
        rca_export_responsibility_source_gate_weight=0.70,
        rca_export_responsibility_onset_weight=0.35,
        rca_export_responsibility_mechanism_residual_weight=0.0,
        rca_export_responsibility_response_suppressor_low_weight=0.0,
    )
    arch_profiles["source-bottleneck-response-suppressor-responsibility-rca"] = dict(
        arch_profiles["source-bottleneck-response-suppressor-rca"],
        rca_event_component_normalize=True,
        rca_event_base_weight=0.45,
        rca_source_gate_weight=0.70,
        rca_onset_weight=0.35,
        rca_mechanism_residual_weight=0.0,
        rca_response_suppressor_weight=0.0,
        rca_export_score_distribution="positive_l1",
        rca_export_score_distribution_top_m=20,
        rca_export_score_distribution_power=2.0,
        rca_export_score_distribution_tau=1.0,
    )
    arch_profiles["source-bottleneck-stable-no-rca-graph-score-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        rca_graph_weight=0.0,
        rca_graph_penalty_weight=0.0,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_secondary_graph_penalty_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-no-encoder-channel-graph-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        use_channel_graph=False,
        use_mechanism_predictive_head=False,
        use_mechanism_coupled_decoder=False,
        use_channel_masked_modeling=False,
        lambda_channel_masked=0.0,
        lambda_channel_mechanism=0.0,
        use_channel_mechanism_score=False,
        channel_mechanism_score_weight=0.0,
        rca_graph_weight=0.0,
        rca_graph_penalty_weight=0.0,
        rca_source_gate_weight=0.0,
        rca_topk_rerank_source_gate_weight=0.0,
        rca_topk_rerank_secondary_source_gate_weight=0.0,
        rca_event_specificity_secondary_source_gate_weight=0.0,
        rca_topk_rerank_graph_penalty_weight=0.0,
        rca_topk_rerank_secondary_graph_penalty_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-no-encoder-channel-graph-score-responsibility-rca"] = dict(
        arch_profiles["source-bottleneck-stable-no-encoder-channel-graph-rca"],
        rca_export_score_distribution="positive_l1",
        rca_export_score_distribution_top_m=20,
        rca_export_score_distribution_power=2.0,
        rca_export_score_distribution_tau=1.0,
        rca_export_responsibility_score=True,
        rca_export_responsibility_base_weight=0.35,
        rca_export_responsibility_source_gate_weight=1.0,
        rca_export_responsibility_onset_weight=0.35,
        rca_export_responsibility_mechanism_residual_weight=0.25,
        rca_export_responsibility_response_suppressor_low_weight=0.0,
    )
    arch_profiles["source-bottleneck-stable-no-encoder-channel-graph-score-responsibility-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-stable-no-encoder-channel-graph-score-responsibility-rca"],
        rca_responsibility_rerank=True,
    )
    arch_profiles["source-bottleneck-stable-no-mechanism-training-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        use_mechanism_predictive_head=False,
        use_mechanism_coupled_decoder=False,
        use_channel_masked_modeling=False,
        lambda_channel_masked=0.0,
        lambda_channel_mechanism=0.0,
        use_channel_mechanism_score=False,
        channel_mechanism_score_weight=0.0,
        rca_source_mechanism_weight=0.0,
        rca_mechanism_guided_source_weight=0.0,
        rca_mechanism_residual_weight=0.0,
        rca_topk_rerank_mechanism_residual_weight=0.0,
        rca_topk_rerank_secondary_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
    )
    arch_profiles["source-bottleneck-evidence-fusion-stable-fill-aligned-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        rca_align_event_onset=True,
        rca_align_event_onset_baseline_window=300,
        rca_align_event_onset_z=2.0,
        rca_align_event_onset_quantile=0.90,
        rca_event_head_ratio=0.30,
        rca_event_head_points=30,
    )
    arch_profiles["source-bottleneck-evidence-fusion-stable-pairwise-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        evidence_fusion_pairwise_effect_weight=0.75,
        evidence_fusion_pairwise_effect_margin=0.12,
    )
    arch_profiles["source-bottleneck-evidence-fusion-strong-pairwise-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-pairwise-rca"],
        evidence_fusion_detach_inputs=True,
        lambda_evidence_fusion=1.0,
        evidence_fusion_pairwise_effect_weight=1.0,
        evidence_fusion_pairwise_effect_margin=0.15,
        rca_evidence_fusion_weight=0.20,
        rca_event_specificity_secondary_evidence_fusion_weight=0.20,
    )
    arch_profiles["source-bottleneck-evidence-fusion-end2end-pairwise-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-pairwise-rca"],
        evidence_fusion_detach_inputs=False,
        lambda_evidence_fusion=0.035,
    )
    arch_profiles["source-bottleneck-consistency-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        source_effect_consistency_weight=0.50,
        source_effect_consistency_rank_weight=0.75,
        source_effect_consistency_effect_rank_weight=0.75,
        source_effect_consistency_effect_suppress_weight=0.25,
        source_effect_consistency_gate_align_weight=0.20,
        source_effect_consistency_margin=0.15,
    )
    arch_profiles["source-bottleneck-consistency-head-rca"] = dict(
        arch_profiles["source-bottleneck-consistency-rca"],
        use_source_consistency_head=True,
        source_consistency_detach_inputs=True,
        lambda_source_consistency_head=0.60,
        source_consistency_head_bce_weight=0.75,
        source_consistency_head_rank_weight=1.0,
        source_consistency_head_effect_suppress_weight=0.25,
        source_consistency_head_pairwise_effect_weight=0.50,
        source_consistency_head_pairwise_effect_margin=0.15,
        rca_source_consistency_head_weight=0.10,
        rca_source_consistency_head_pooling="head_mean",
        rca_source_consistency_head_ratio=0.30,
        rca_source_consistency_head_points=30,
        rca_source_consistency_head_top_quantile=0.80,
        rca_topk_rerank_source_consistency_head_weight=0.15,
        rca_topk_rerank_secondary_source_consistency_head_weight=0.20,
        rca_event_specificity_secondary_source_consistency_head_weight=0.20,
    )
    arch_profiles["source-bottleneck-consistency-head-e2e-trainonly-rca"] = dict(
        arch_profiles["source-bottleneck-consistency-head-rca"],
        source_consistency_detach_inputs=False,
        rca_source_consistency_head_weight=0.0,
        rca_topk_rerank_source_consistency_head_weight=0.0,
        rca_topk_rerank_secondary_source_consistency_head_weight=0.0,
        rca_event_specificity_secondary_source_consistency_head_weight=0.0,
    )
    arch_profiles["source-bottleneck-source-interaction-head-rca"] = dict(
        arch_profiles["source-bottleneck-evidence-fusion-stable-fill-rca"],
        use_source_interaction_head=True,
        source_interaction_detach_inputs=True,
        lambda_source_interaction_head=0.08,
        source_interaction_head_bce_weight=0.75,
        source_interaction_head_rank_weight=1.0,
        source_interaction_head_effect_suppress_weight=0.25,
        source_interaction_head_pairwise_effect_weight=0.50,
        source_interaction_head_pairwise_effect_margin=0.15,
        rca_source_interaction_head_weight=0.0,
        rca_source_interaction_head_pooling="head_mean",
        rca_source_interaction_head_ratio=0.30,
        rca_source_interaction_head_points=30,
        rca_source_interaction_head_top_quantile=0.80,
    )
    arch_profiles["source-bottleneck-source-interaction-head-direct-rca"] = dict(
        arch_profiles["source-bottleneck-source-interaction-head-rca"],
        rca_source_interaction_head_weight=0.08,
    )
    arch_profiles["source-bottleneck-evidence-fusion-competitive-rca"] = dict(
        arch_profiles["source-bottleneck-event-top1-coverage-evidence-top20-rerank-rca"],
        use_evidence_fusion_head=True,
        evidence_fusion_hidden=16,
        evidence_fusion_detach_inputs=False,
        evidence_fusion_loss_mode="softmax",
        evidence_fusion_export_mode="softmax",
        lambda_evidence_fusion=0.10,
        evidence_fusion_bce_weight=1.0,
        evidence_fusion_rank_weight=1.0,
        evidence_fusion_effect_suppress_weight=0.50,
        evidence_fusion_entropy_weight=0.0,
        rca_evidence_fusion_weight=0.12,
        rca_evidence_fusion_pooling="head_mean",
        rca_evidence_fusion_head_ratio=0.30,
        rca_evidence_fusion_head_points=30,
        rca_evidence_fusion_top_quantile=0.80,
        rca_event_specificity_keep_top_k=1,
        rca_event_specificity_fill_secondary_top_k=5,
        rca_event_specificity_secondary_base_weight=0.35,
        rca_event_specificity_secondary_onset_weight=0.20,
        rca_event_specificity_secondary_source_gate_weight=0.70,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_source_interaction_weight=0.25,
        rca_event_specificity_secondary_evidence_fusion_weight=0.35,
    )
    arch_profiles["source-bottleneck-adaptive-evidence-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-event-swat-priority-evidence-top20-rerank-rca"],
        rca_event_specificity_keep_top_k=0,
        rca_event_specificity_fill_secondary_top_k=0,
        rca_event_specificity_secondary_base_weight=0.0,
        rca_event_specificity_secondary_onset_weight=0.0,
        rca_event_specificity_secondary_source_gate_weight=0.0,
        rca_event_specificity_secondary_mechanism_residual_weight=0.0,
        rca_event_specificity_secondary_source_interaction_weight=0.0,
        rca_event_specificity_secondary_evidence_fusion_weight=0.0,
        rca_adaptive_evidence_rerank=True,
        rca_adaptive_evidence_top_k=20,
        rca_adaptive_evidence_keep_top_k=1,
        rca_adaptive_evidence_fill_top_k=5,
        rca_adaptive_evidence_gate="source_top_old_rank",
        rca_adaptive_evidence_source_old_rank_threshold=1,
        rca_adaptive_evidence_source_base_weight=0.45,
        rca_adaptive_evidence_source_onset_weight=0.25,
        rca_adaptive_evidence_source_gate_weight=1.0,
        rca_adaptive_evidence_source_mechanism_residual_weight=0.25,
        rca_adaptive_evidence_source_interaction_weight=0.25,
        rca_adaptive_evidence_fallback_original_weight=1.0,
        rca_adaptive_evidence_fallback_base_weight=0.45,
        rca_adaptive_evidence_fallback_onset_weight=0.20,
    )
    arch_profiles["source-bottleneck-adaptive-evidence-wadi-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-adaptive-evidence-rerank-rca"],
        rca_adaptive_evidence_gate="exported_top_source_rank",
        rca_adaptive_evidence_source_old_rank_threshold=2,
    )
    arch_profiles["source-bottleneck-adaptive-evidence-margin-rerank-rca"] = dict(
        arch_profiles["source-bottleneck-adaptive-evidence-rerank-rca"],
        rca_adaptive_evidence_gate="source_margin",
        rca_adaptive_evidence_source_margin_threshold=0.35,
    )
    arch_profiles["source-bottleneck-rca-aware-checkpoint"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_rca_aware_checkpoint=True,
        rca_checkpoint_proxy_weight=0.02,
        rca_checkpoint_proxy_batches=2,
        rca_checkpoint_min_epoch=4,
    )
    arch_profiles["source-bottleneck-normalized-rca-checkpoint"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_rca_aware_checkpoint=True,
        rca_checkpoint_normalize=True,
        rca_checkpoint_proxy_weight=0.5,
        rca_checkpoint_proxy_batches=2,
        rca_checkpoint_min_epoch=4,
    )
    arch_profiles["source-bottleneck-adaptive-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_adaptive_mechanism_gate_weight=0.20,
        rca_adaptive_mechanism_gate_floor=0.05,
        rca_adaptive_mechanism_gate_mode="source_onset",
    )
    arch_profiles["source-bottleneck-event-adaptive-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_adaptive_mechanism_gate_weight=0.0,
        rca_adaptive_mechanism_selection_weight=0.50,
        rca_adaptive_mechanism_selection_floor=0.05,
        rca_adaptive_mechanism_selection_mode="source_onset",
    )
    arch_profiles["source-bottleneck-conservative-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_adaptive_mechanism_gate_weight=0.0,
        rca_adaptive_mechanism_selection_weight=0.0,
        rca_conservative_mechanism_weight=0.25,
        rca_conservative_mechanism_support_floor=0.25,
        rca_conservative_mechanism_candidate_topk=8,
        rca_conservative_mechanism_support_mode="source_onset",
    )
    arch_profiles["source-bottleneck-source-gated-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_adaptive_mechanism_gate_weight=0.0,
        rca_adaptive_mechanism_selection_weight=0.0,
        rca_conservative_mechanism_weight=0.0,
        rca_source_gated_mechanism_weight=0.25,
        rca_source_gated_mechanism_support_floor=0.20,
        rca_source_gated_mechanism_candidate_topk=8,
        rca_source_gated_mechanism_support_mode="source_onset",
    )
    arch_profiles["source-bottleneck-mechanism-train-only-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_source_mechanism_weight=0.0,
        rca_mechanism_guided_source_weight=0.0,
        rca_mechanism_residual_weight=0.0,
        rca_adaptive_mechanism_gate_weight=0.0,
        rca_adaptive_mechanism_selection_weight=0.0,
        rca_conservative_mechanism_weight=0.0,
        rca_source_gated_mechanism_weight=0.0,
    )
    arch_profiles["source-bottleneck-no-mechanism-decoder-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_mechanism_coupled_decoder=False,
        lambda_channel_mechanism=0.0,
    )
    arch_profiles["source-bottleneck-event-stable-rca"] = dict(
        arch_profiles["source-bottleneck-no-mechanism-decoder-rca"],
        score_smoothing_window=3,
        score_smoothing_method="mean",
        prediction_fill_gap=2,
        prediction_min_len=1,
        prediction_dilate=1,
        rca_hierarchical_group_aggregation="mean",
    )
    arch_profiles["source-bottleneck-no-channel-masked-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_channel_masked_modeling=False,
        lambda_channel_masked=0.0,
    )
    arch_profiles["source-bottleneck-no-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_mechanism_predictive_head=False,
        use_mechanism_coupled_decoder=False,
        use_channel_masked_modeling=False,
        lambda_channel_masked=0.0,
        lambda_channel_mechanism=0.0,
        use_channel_mechanism_score=False,
        channel_mechanism_score_weight=0.0,
        rca_source_mechanism_weight=0.0,
        rca_mechanism_guided_source_weight=0.0,
        rca_mechanism_residual_weight=0.0,
    )
    arch_profiles["source-bottleneck-no-source-gate-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_source_gate=False,
        lambda_source_gate_sparse=0.0,
        lambda_interventional_source_bce=0.0,
        interventional_rank_signal="mechanism",
        lambda_source_bottleneck=0.0,
        source_bottleneck_bce_weight=0.0,
        source_bottleneck_rank_weight=0.0,
        source_bottleneck_effect_suppress_weight=0.0,
        source_bottleneck_specificity_weight=0.0,
        rca_source_gate_weight=0.0,
    )
    arch_profiles["source-bottleneck-no-source-bottleneck-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_source_gate=False,
        lambda_source_gate_sparse=0.0,
        lambda_source_bottleneck=0.0,
        source_bottleneck_bce_weight=0.0,
        source_bottleneck_rank_weight=0.0,
        source_bottleneck_effect_suppress_weight=0.0,
        source_bottleneck_specificity_weight=0.0,
        rca_source_gate_weight=0.0,
        rca_source_interaction_weight=0.0,
    )
    arch_profiles["source-bottleneck-no-source-propagation-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        rca_use_source_propagation=False,
        rca_source_weight=0.0,
        rca_source_base_weight=0.0,
        rca_source_score_weight=0.0,
        rca_source_gate_weight=0.0,
        rca_source_interaction_weight=0.0,
        rca_propagation_weight=0.0,
        rca_source_mechanism_weight=0.0,
        rca_source_innovation_weight=0.0,
        rca_graph_penalty_weight=0.0,
        rca_onset_weight=0.0,
        rca_event_specificity_weight=0.0,
    )
    arch_profiles["source-bottleneck-corefine-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_channel_temporal_corefinement=True,
        corefinement_init=0.10,
        corefinement_detach_first_pass=True,
    )
    arch_profiles["source-aware-dual-corefine-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_channel_temporal_corefinement=True,
        corefinement_init=0.10,
        corefinement_detach_first_pass=True,
        use_source_aware_corefinement=True,
        source_aware_corefinement_init=0.15,
        source_aware_corefinement_detach_gate=True,
    )
    arch_profiles["source-preserving-mechanism-fusion-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_source_preserving_decoder=True,
        source_preserving_init=0.65,
        source_preserving_detach_gate=True,
    )
    arch_profiles["source-preserving-ultralight-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_source_preserving_decoder=True,
        source_preserving_init=0.20,
        source_preserving_detach_gate=True,
    )
    arch_profiles["source-preserving-light-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_source_preserving_decoder=True,
        source_preserving_init=0.35,
        source_preserving_detach_gate=True,
    )
    arch_profiles["lagged-directional-mechanism-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_lagged_causal_graph=True,
        causal_lags=[1, 3, 6, 12],
        causal_topk=5,
        causal_detach_backbone=False,
        lambda_causal_mechanism=0.01,
        lambda_causal_sparse=0.001,
        use_causal_score=True,
        causal_score_weight=0.05,
        rca_causal_weight=0.15,
    )
    arch_profiles["source-bottleneck-trained-specificity-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        source_bottleneck_specificity_weight=0.25,
    )
    arch_profiles["source-bottleneck-gate-specificity-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        source_effect_specificity_weight=2.0,
        source_bottleneck_specificity_weight=0.0,
    )
    arch_profiles["source-bottleneck-effective-specificity-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        source_effect_specificity_weight=100.0,
        source_bottleneck_specificity_weight=0.0,
        source_effect_interval=4,
    )
    arch_profiles["source-bottleneck-head-specificity-rca"] = dict(
        arch_profiles["source-bottleneck-specificity-rca"],
        use_synthetic_rca_head=True,
        use_synthetic_rca_loss=False,
        lambda_synthetic_rca=0.0,
        lambda_masked_rca_head=0.05,
        masked_rca_bce_weight=1.0,
        masked_rca_rank_weight=1.0,
        synthetic_rca_margin=0.15,
        synthetic_rca_topk=8,
        lambda_source_effect_rca_head=2.0,
        source_effect_rca_head_bce_weight=1.0,
        source_effect_rca_head_rank_weight=1.0,
        rca_synthetic_weight=0.60,
    )
    arch_profiles["interventional-fused-source-rca"] = dict(
        arch_profiles["interventional-graph-source-rca"],
        rca_source_weight=1.0,
        rca_source_base_weight=1.0,
        rca_source_score_weight=0.75,
        rca_propagation_weight=1.0,
        rca_source_gate_weight=0.10,
        rca_source_mechanism_weight=0.0,
        rca_graph_penalty_weight=0.0,
        rca_onset_weight=0.25,
        rca_source_interaction_weight=0.0,
        channel_mechanism_score_weight=0.20,
    )
    arch_profiles["interventional-consensus-source-rca"] = dict(
        arch_profiles["interventional-graph-source-rca"],
        rca_source_score_weight=0.0,
        rca_source_consensus_weight=0.50,
        rca_onset_consensus_weight=0.10,
        rca_source_consensus_mode="sqrt_bg",
    )
    arch_profiles["interventional-context-source-rca"] = dict(
        arch_profiles["interventional-graph-source-rca"],
        lambda_interventional_source_bce=0.0,
        lambda_interventional_source_rank=0.02,
        lambda_interventional_graph_support=0.02,
        interventional_rank_signal="mechanism",
    )
    arch_profiles["mechanism-calibrated-rca"] = dict(
        arch_profiles["soft-hierarchical-rca"],
        rca_source_base_weight=0.25,
        rca_source_mechanism_weight=0.10,
        rca_onset_weight=0.50,
        rca_source_interaction_weight=2.0,
        rca_graph_penalty_weight=0.10,
        rca_hierarchical_group_boost=0.2,
    )
    arch_profiles["source-preserving-rca"] = dict(
        arch_profiles["soft-hierarchical-rca"],
        use_source_gate=True,
        source_gate_init=0.20,
        lambda_source_gate_sparse=0.0005,
        use_source_effect_synthetic=True,
        lambda_source_effect=0.02,
        source_effect_interval=32,
        source_effect_min_len=8,
        source_effect_max_len=30,
        source_effect_min_roots=1,
        source_effect_max_roots=2,
        source_effect_neighbor_topk=3,
        source_effect_strength=0.35,
        source_effect_delay_max=6,
        source_effect_bce_weight=1.0,
        source_effect_rank_weight=1.0,
        source_effect_effect_rank_weight=0.5,
        source_effect_margin=0.15,
        rca_source_gate_weight=0.0,
        rca_synthetic_weight=0.0,
    )

    arch_switches = arch_profiles[args.arch_profile]
    vq_hyper_params = {}
    if vq_cooldown_epochs is not None:
        vq_hyper_params["vq_cooldown_epochs"] = max(0, vq_cooldown_epochs)
    if lambda_vq is not None:
        vq_hyper_params["lambda_vq"] = max(0.0, lambda_vq)
    if vq_score_weight is not None:
        vq_hyper_params["vq_score_weight"] = max(0.0, vq_score_weight)
    score_hyper_params = {}
    if args.score_topk_k is not None:
        score_hyper_params["score_topk_k"] = max(1, args.score_topk_k)
    if args.score_aggregation is not None:
        score_hyper_params["score_aggregation"] = args.score_aggregation
    if args.score_aggregation_quantile is not None:
        score_hyper_params["score_aggregation_quantile"] = min(
            max(float(args.score_aggregation_quantile), 0.0), 1.0
        )
    if args.score_center_width is not None:
        score_hyper_params["score_center_width"] = max(1, args.score_center_width)
    if args.score_smoothing_window is not None:
        score_hyper_params["score_smoothing_window"] = max(1, args.score_smoothing_window)
    if args.score_smoothing_method is not None:
        score_hyper_params["score_smoothing_method"] = args.score_smoothing_method
    if args.prediction_fill_gap is not None:
        score_hyper_params["prediction_fill_gap"] = max(0, args.prediction_fill_gap)
    if args.prediction_min_len is not None:
        score_hyper_params["prediction_min_len"] = max(1, args.prediction_min_len)
    if args.prediction_dilate is not None:
        score_hyper_params["prediction_dilate"] = max(0, args.prediction_dilate)
    if args.event_persistence_window is not None:
        score_hyper_params["use_event_persistence_score"] = True
        score_hyper_params["event_persistence_window"] = max(1, args.event_persistence_window)
    if args.event_persistence_weight is not None:
        score_hyper_params["use_event_persistence_score"] = True
        score_hyper_params["event_persistence_weight"] = max(0.0, args.event_persistence_weight)
    if args.dynamic_temporal_residual_init is not None:
        score_hyper_params["dynamic_temporal_residual_init"] = max(0.0, args.dynamic_temporal_residual_init)
    if args.dynamic_temporal_topk is not None:
        score_hyper_params["dynamic_temporal_topk"] = max(1, args.dynamic_temporal_topk)
    if args.temporal_graph_lr_scale is not None:
        score_hyper_params["temporal_graph_lr_scale"] = max(0.0, args.temporal_graph_lr_scale)
    if args.channel_prior_align is not None:
        score_hyper_params["lambda_channel_prior_align"] = max(0.0, args.channel_prior_align)
        if args.channel_prior_align > 0:
            score_hyper_params["use_channel_corr_prior"] = True
    if args.channel_prior_weight is not None:
        score_hyper_params["channel_corr_prior_weight"] = min(
            max(float(args.channel_prior_weight), 0.0), 1.0
        )
        score_hyper_params["use_channel_corr_prior"] = True
    if args.channel_prior_bias is not None:
        score_hyper_params["channel_corr_prior_bias"] = max(0.0, float(args.channel_prior_bias))
        score_hyper_params["use_channel_corr_prior"] = True
    if args.channel_prior_topk is not None:
        score_hyper_params["channel_corr_prior_topk"] = max(1, args.channel_prior_topk)
        score_hyper_params["use_channel_corr_prior"] = True
    if args.state_aware_num_states is not None:
        score_hyper_params["state_aware_num_states"] = max(2, args.state_aware_num_states)
    if args.state_aware_graph_gate_init is not None:
        score_hyper_params["state_aware_graph_gate_init"] = min(
            max(float(args.state_aware_graph_gate_init), 1e-3), 1.0 - 1e-3
        )
    if args.state_aware_residual_init is not None:
        score_hyper_params["state_aware_residual_init"] = min(
            max(float(args.state_aware_residual_init), 1e-3), 1.0 - 1e-3
        )
    if args.lambda_state_balance is not None:
        score_hyper_params["lambda_state_balance"] = max(0.0, args.lambda_state_balance)
    if args.lambda_state_confidence is not None:
        score_hyper_params["lambda_state_confidence"] = max(0.0, args.lambda_state_confidence)
    if args.causal_score_weight is not None:
        score_hyper_params["causal_score_weight"] = max(0.0, args.causal_score_weight)
    if args.lambda_causal_mechanism is not None:
        score_hyper_params["lambda_causal_mechanism"] = max(0.0, args.lambda_causal_mechanism)
    if args.lambda_channel_mechanism is not None:
        score_hyper_params["lambda_channel_mechanism"] = max(0.0, args.lambda_channel_mechanism)
    if args.channel_mechanism_score_weight is not None:
        score_hyper_params["channel_mechanism_score_weight"] = max(
            0.0, args.channel_mechanism_score_weight,
        )
        score_hyper_params["use_channel_mechanism_score"] = True
    if args.mechanism_coupling_init is not None:
        score_hyper_params["mechanism_coupling_init"] = min(
            max(float(args.mechanism_coupling_init), 0.0),
            1.0,
        )
    if args.mechanism_predictive_blend_init is not None:
        score_hyper_params["mechanism_predictive_blend_init"] = min(
            max(float(args.mechanism_predictive_blend_init), 0.0),
            1.0,
        )
    if args.causal_topk is not None:
        score_hyper_params["causal_topk"] = max(1, args.causal_topk)
    if args.lambda_synthetic_anomaly is not None:
        score_hyper_params["lambda_synthetic_anomaly"] = max(0.0, args.lambda_synthetic_anomaly)
    if args.lambda_synthetic_rca is not None:
        score_hyper_params["lambda_synthetic_rca"] = max(0.0, args.lambda_synthetic_rca)
        score_hyper_params["use_synthetic_rca_loss"] = score_hyper_params["lambda_synthetic_rca"] > 0.0
    if args.synthetic_aux_interval is not None:
        score_hyper_params["synthetic_aux_interval"] = max(1, int(args.synthetic_aux_interval))
    if args.synthetic_rca_margin is not None:
        score_hyper_params["synthetic_rca_margin"] = max(float(args.synthetic_rca_margin), 0.0)
    if args.synthetic_rca_topk is not None:
        score_hyper_params["synthetic_rca_topk"] = max(1, int(args.synthetic_rca_topk))
    if args.synthetic_rca_positive_aggregation is not None:
        score_hyper_params["synthetic_rca_positive_aggregation"] = args.synthetic_rca_positive_aggregation
    if args.synthetic_rca_positive_topk is not None:
        score_hyper_params["synthetic_rca_positive_topk"] = max(
            1, int(args.synthetic_rca_positive_topk)
        )
    if args.synthetic_rca_positive_min_weight is not None:
        score_hyper_params["synthetic_rca_positive_min_weight"] = min(
            max(0.0, float(args.synthetic_rca_positive_min_weight)),
            1.0,
        )
    if args.synthetic_rca_bce_weight is not None:
        score_hyper_params["synthetic_rca_bce_weight"] = max(0.0, float(args.synthetic_rca_bce_weight))
    if args.synthetic_rca_rank_weight is not None:
        score_hyper_params["synthetic_rca_rank_weight"] = max(0.0, float(args.synthetic_rca_rank_weight))
    if args.synthetic_score_weight is not None:
        score_hyper_params["synthetic_score_weight"] = max(0.0, args.synthetic_score_weight)
    if args.lambda_source_bottleneck is not None:
        score_hyper_params["lambda_source_bottleneck"] = max(0.0, float(args.lambda_source_bottleneck))
    if args.source_bottleneck_bce_weight is not None:
        score_hyper_params["source_bottleneck_bce_weight"] = max(
            0.0, float(args.source_bottleneck_bce_weight)
        )
    if args.source_bottleneck_rank_weight is not None:
        score_hyper_params["source_bottleneck_rank_weight"] = max(
            0.0, float(args.source_bottleneck_rank_weight)
        )
    if args.source_bottleneck_effect_suppress_weight is not None:
        score_hyper_params["source_bottleneck_effect_suppress_weight"] = max(
            0.0, float(args.source_bottleneck_effect_suppress_weight)
        )
    if args.root_response_penalty_init is not None:
        score_hyper_params["root_response_penalty_init"] = max(
            1e-4, float(args.root_response_penalty_init)
        )
    if args.root_response_confidence_discount is not None:
        score_hyper_params["root_response_confidence_discount"] = min(
            max(0.0, float(args.root_response_confidence_discount)),
            0.95,
        )
    if args.root_response_use_innovation_split:
        score_hyper_params["root_response_use_innovation_split"] = True
    if args.pairwise_root_response:
        score_hyper_params["use_pairwise_root_response_head"] = True
        score_hyper_params["use_root_score_head"] = True
        score_hyper_params["root_score_head_mode"] = "root_response"
    if args.lambda_pairwise_root_response is not None:
        score_hyper_params["lambda_pairwise_root_response"] = max(
            0.0, float(args.lambda_pairwise_root_response)
        )
        if score_hyper_params["lambda_pairwise_root_response"] > 0.0:
            score_hyper_params["use_pairwise_root_response_head"] = True
            score_hyper_params["use_root_score_head"] = True
            score_hyper_params["root_score_head_mode"] = "root_response"
    if args.pairwise_root_response_logit_weight is not None:
        score_hyper_params["pairwise_root_response_logit_weight"] = max(
            0.0, float(args.pairwise_root_response_logit_weight)
        )
    if args.root_response_source_branch_weight is not None:
        score_hyper_params["root_response_source_branch_weight"] = max(
            0.0, float(args.root_response_source_branch_weight)
        )
    if args.root_response_response_branch_weight is not None:
        score_hyper_params["root_response_response_branch_weight"] = max(
            0.0, float(args.root_response_response_branch_weight)
        )
    if args.root_response_response_suppress_weight is not None:
        score_hyper_params["root_response_response_suppress_weight"] = max(
            0.0, float(args.root_response_response_suppress_weight)
        )
    if args.lambda_event_responsibility is not None:
        score_hyper_params["lambda_event_responsibility"] = max(
            0.0, float(args.lambda_event_responsibility)
        )
        score_hyper_params["use_event_responsibility_head"] = (
            score_hyper_params["lambda_event_responsibility"] > 0.0
        )
    if args.event_responsibility_rank_weight is not None:
        score_hyper_params["event_responsibility_rank_weight"] = max(
            0.0, float(args.event_responsibility_rank_weight)
        )
    if args.event_responsibility_effect_suppress_weight is not None:
        score_hyper_params["event_responsibility_effect_suppress_weight"] = max(
            0.0, float(args.event_responsibility_effect_suppress_weight)
        )
    if args.lambda_channel_masked is not None:
        score_hyper_params["lambda_channel_masked"] = max(0.0, float(args.lambda_channel_masked))
        score_hyper_params["use_channel_masked_modeling"] = score_hyper_params["lambda_channel_masked"] > 0.0
    if args.channel_mask_interval is not None:
        score_hyper_params["channel_mask_interval"] = max(1, int(args.channel_mask_interval))
    if args.channel_mask_ratio is not None:
        score_hyper_params["channel_mask_ratio"] = min(max(float(args.channel_mask_ratio), 1e-6), 1.0)
    if args.channel_mask_value is not None:
        score_hyper_params["channel_mask_value"] = args.channel_mask_value
    if args.reconstruction_loss is not None:
        score_hyper_params["reconstruction_loss_type"] = args.reconstruction_loss
    if args.smooth_l1_beta is not None:
        score_hyper_params["smooth_l1_beta"] = max(float(args.smooth_l1_beta), 1e-6)
    if args.mse_l1_alpha is not None:
        score_hyper_params["mse_l1_alpha"] = min(max(float(args.mse_l1_alpha), 0.0), 1.0)
    if args.mse_robust_alpha is not None:
        score_hyper_params["mse_robust_alpha"] = min(max(float(args.mse_robust_alpha), 0.0), 1.0)
    if args.charbonnier_eps is not None:
        score_hyper_params["charbonnier_eps"] = max(float(args.charbonnier_eps), 1e-12)
    if args.lambda_temporal_diff_loss is not None:
        score_hyper_params["lambda_temporal_diff_loss"] = max(0.0, args.lambda_temporal_diff_loss)
    if args.rca_graph_weight is not None:
        score_hyper_params["rca_graph_weight"] = max(0.0, args.rca_graph_weight)
    if args.rca_mechanism_weight is not None:
        score_hyper_params["rca_mechanism_weight"] = max(0.0, args.rca_mechanism_weight)
    if args.rca_source_weight is not None:
        score_hyper_params["rca_source_weight"] = max(0.0, float(args.rca_source_weight))
    if args.rca_source_base_weight is not None:
        score_hyper_params["rca_source_base_weight"] = max(0.0, float(args.rca_source_base_weight))
    if args.rca_source_score_weight is not None:
        score_hyper_params["rca_source_score_weight"] = max(0.0, float(args.rca_source_score_weight))
    if args.rca_source_gate_weight is not None:
        score_hyper_params["rca_source_gate_weight"] = max(0.0, float(args.rca_source_gate_weight))
    if args.rca_root_score_weight is not None:
        score_hyper_params["rca_root_score_weight"] = max(0.0, float(args.rca_root_score_weight))
    if args.rca_event_responsibility_weight is not None:
        score_hyper_params["rca_event_responsibility_weight"] = max(
            0.0, float(args.rca_event_responsibility_weight)
        )
        if score_hyper_params["rca_event_responsibility_weight"] > 0.0:
            score_hyper_params["use_event_responsibility_head"] = True
    if args.rca_root_score_signal is not None:
        score_hyper_params["rca_root_score_signal"] = args.rca_root_score_signal
    if args.rca_root_score_pooling is not None:
        score_hyper_params["rca_root_score_pooling"] = args.rca_root_score_pooling
    if args.rca_root_score_head_ratio is not None:
        score_hyper_params["rca_root_score_head_ratio"] = min(
            max(float(args.rca_root_score_head_ratio), 1e-6),
            1.0,
        )
    if args.rca_root_score_head_points is not None:
        score_hyper_params["rca_root_score_head_points"] = max(
            0, int(args.rca_root_score_head_points)
        )
    if args.rca_root_score_top_quantile is not None:
        score_hyper_params["rca_root_score_top_quantile"] = min(
            max(float(args.rca_root_score_top_quantile), 0.0),
            0.999,
        )
    if args.rca_source_consensus_weight is not None:
        score_hyper_params["rca_source_consensus_weight"] = max(
            0.0, float(args.rca_source_consensus_weight)
        )
    if args.rca_onset_consensus_weight is not None:
        score_hyper_params["rca_onset_consensus_weight"] = max(
            0.0, float(args.rca_onset_consensus_weight)
        )
    if args.rca_source_consensus_mode is not None:
        score_hyper_params["rca_source_consensus_mode"] = args.rca_source_consensus_mode
    if args.rca_propagation_weight is not None:
        score_hyper_params["rca_propagation_weight"] = max(0.0, float(args.rca_propagation_weight))
    if args.rca_source_mechanism_weight is not None:
        score_hyper_params["rca_source_mechanism_weight"] = max(0.0, float(args.rca_source_mechanism_weight))
    if args.rca_source_innovation_weight is not None:
        score_hyper_params["rca_source_innovation_weight"] = max(
            0.0, float(args.rca_source_innovation_weight)
        )
    if args.rca_source_innovation_mode is not None:
        score_hyper_params["rca_source_innovation_mode"] = args.rca_source_innovation_mode
    if args.rca_source_innovation_neighbor_weight is not None:
        score_hyper_params["rca_source_innovation_neighbor_weight"] = max(
            0.0, float(args.rca_source_innovation_neighbor_weight)
        )
    if args.rca_source_innovation_lead_points is not None:
        score_hyper_params["rca_source_innovation_lead_points"] = max(
            0, int(args.rca_source_innovation_lead_points)
        )
    if args.rca_causal_weight is not None:
        score_hyper_params["rca_causal_weight"] = max(0.0, float(args.rca_causal_weight))
    if args.rca_synthetic_weight is not None:
        score_hyper_params["rca_synthetic_weight"] = max(0.0, float(args.rca_synthetic_weight))
    if args.rca_counterfactual_weight is not None:
        score_hyper_params["rca_counterfactual_weight"] = max(0.0, float(args.rca_counterfactual_weight))
    if args.rca_counterfactual_candidates is not None:
        score_hyper_params["rca_counterfactual_candidates"] = max(1, int(args.rca_counterfactual_candidates))
    if args.rca_counterfactual_max_windows is not None:
        score_hyper_params["rca_counterfactual_max_windows"] = max(1, int(args.rca_counterfactual_max_windows))
    if args.rca_onset_weight is not None:
        score_hyper_params["rca_onset_weight"] = max(0.0, float(args.rca_onset_weight))
    if args.rca_onset_baseline_window is not None:
        score_hyper_params["rca_onset_baseline_window"] = max(1, int(args.rca_onset_baseline_window))
    if args.rca_onset_z is not None:
        score_hyper_params["rca_onset_z"] = max(0.0, float(args.rca_onset_z))
    if args.rca_graph_direction is not None:
        score_hyper_params["rca_graph_direction"] = args.rca_graph_direction
    if args.rca_contrast_window is not None:
        score_hyper_params["rca_contrast_window"] = max(0, args.rca_contrast_window)
    if args.rca_contrast_weight is not None:
        score_hyper_params["rca_contrast_weight"] = max(0.0, args.rca_contrast_weight)
    if args.rca_mechanism_residual_window is not None:
        score_hyper_params["rca_mechanism_residual_window"] = max(0, args.rca_mechanism_residual_window)
    if args.rca_mechanism_residual_weight is not None:
        score_hyper_params["rca_mechanism_residual_weight"] = max(0.0, args.rca_mechanism_residual_weight)
    if args.rca_adaptive_mechanism_gate_weight is not None:
        score_hyper_params["rca_adaptive_mechanism_gate_weight"] = max(
            0.0, float(args.rca_adaptive_mechanism_gate_weight)
        )
    if args.rca_adaptive_mechanism_gate_floor is not None:
        score_hyper_params["rca_adaptive_mechanism_gate_floor"] = min(
            max(float(args.rca_adaptive_mechanism_gate_floor), 0.0),
            1.0,
        )
    if args.rca_adaptive_mechanism_gate_mode is not None:
        score_hyper_params["rca_adaptive_mechanism_gate_mode"] = args.rca_adaptive_mechanism_gate_mode
    if args.rca_adaptive_mechanism_selection_weight is not None:
        score_hyper_params["rca_adaptive_mechanism_selection_weight"] = min(
            max(float(args.rca_adaptive_mechanism_selection_weight), 0.0),
            1.0,
        )
    if args.rca_adaptive_mechanism_selection_floor is not None:
        score_hyper_params["rca_adaptive_mechanism_selection_floor"] = min(
            max(float(args.rca_adaptive_mechanism_selection_floor), 0.0),
            1.0,
        )
    if args.rca_adaptive_mechanism_selection_mode is not None:
        score_hyper_params["rca_adaptive_mechanism_selection_mode"] = (
            args.rca_adaptive_mechanism_selection_mode
        )
    if args.rca_conservative_mechanism_weight is not None:
        score_hyper_params["rca_conservative_mechanism_weight"] = max(
            0.0, float(args.rca_conservative_mechanism_weight)
        )
    if args.rca_conservative_mechanism_support_floor is not None:
        score_hyper_params["rca_conservative_mechanism_support_floor"] = min(
            max(float(args.rca_conservative_mechanism_support_floor), 0.0),
            1.0,
        )
    if args.rca_conservative_mechanism_candidate_topk is not None:
        score_hyper_params["rca_conservative_mechanism_candidate_topk"] = max(
            0, int(args.rca_conservative_mechanism_candidate_topk)
        )
    if args.rca_conservative_mechanism_support_mode is not None:
        score_hyper_params["rca_conservative_mechanism_support_mode"] = (
            args.rca_conservative_mechanism_support_mode
        )
    if args.rca_source_gated_mechanism_weight is not None:
        score_hyper_params["rca_source_gated_mechanism_weight"] = max(
            0.0, float(args.rca_source_gated_mechanism_weight)
        )
    if args.rca_source_gated_mechanism_support_floor is not None:
        score_hyper_params["rca_source_gated_mechanism_support_floor"] = min(
            max(float(args.rca_source_gated_mechanism_support_floor), 0.0),
            1.0,
        )
    if args.rca_source_gated_mechanism_candidate_topk is not None:
        score_hyper_params["rca_source_gated_mechanism_candidate_topk"] = max(
            0, int(args.rca_source_gated_mechanism_candidate_topk)
        )
    if args.rca_source_gated_mechanism_support_mode is not None:
        score_hyper_params["rca_source_gated_mechanism_support_mode"] = (
            args.rca_source_gated_mechanism_support_mode
        )
    if args.rca_event_head_ratio is not None:
        score_hyper_params["rca_event_head_ratio"] = min(max(float(args.rca_event_head_ratio), 1e-6), 1.0)
    if args.rca_event_head_points is not None:
        score_hyper_params["rca_event_head_points"] = max(0, int(args.rca_event_head_points))
    if args.rca_prediction_key is not None:
        score_hyper_params["rca_prediction_key"] = args.rca_prediction_key
    if args.rca_export_lite:
        score_hyper_params["rca_export_lite"] = True
    if args.rca_export_top_k is not None:
        score_hyper_params["rca_export_top_k"] = max(1, int(args.rca_export_top_k))
    if args.rca_event_local_export:
        score_hyper_params["rca_event_local_export"] = True
    if args.rca_event_local_margin is not None:
        score_hyper_params["rca_event_local_margin"] = max(0, int(args.rca_event_local_margin))
    if args.rca_split_predicted_events:
        score_hyper_params["rca_split_predicted_events"] = True
    if args.rca_split_max_event_len is not None:
        score_hyper_params["rca_split_max_event_len"] = max(1, int(args.rca_split_max_event_len))
    if args.rca_split_stride is not None:
        score_hyper_params["rca_split_stride"] = max(1, int(args.rca_split_stride))
    checkpoint_policy = args.checkpoint_policy
    if args.paper_protocol and checkpoint_policy == "profile":
        checkpoint_policy = "best-val"

    if checkpoint_policy == "best-val":
        score_hyper_params["use_latest_checkpoint"] = False
        score_hyper_params["use_rca_aware_checkpoint"] = False
        score_hyper_params["rca_checkpoint_normalize"] = False
    elif checkpoint_policy == "latest":
        score_hyper_params["use_latest_checkpoint"] = True
        score_hyper_params["use_rca_aware_checkpoint"] = False
        score_hyper_params["rca_checkpoint_normalize"] = False
    elif checkpoint_policy == "rca-aware":
        score_hyper_params["use_latest_checkpoint"] = False
        score_hyper_params["use_rca_aware_checkpoint"] = True
        score_hyper_params["rca_checkpoint_normalize"] = False
    elif checkpoint_policy == "normalized-rca-aware":
        score_hyper_params["use_latest_checkpoint"] = False
        score_hyper_params["use_rca_aware_checkpoint"] = True
        score_hyper_params["rca_checkpoint_normalize"] = True

    if args.use_latest_checkpoint:
        score_hyper_params["use_latest_checkpoint"] = True
    if args.rca_aware_checkpoint:
        score_hyper_params["use_rca_aware_checkpoint"] = True
    if args.rca_checkpoint_normalize:
        score_hyper_params["use_rca_aware_checkpoint"] = True
        score_hyper_params["rca_checkpoint_normalize"] = True
    if args.rca_checkpoint_proxy_weight is not None:
        score_hyper_params["rca_checkpoint_proxy_weight"] = max(
            0.0, float(args.rca_checkpoint_proxy_weight)
        )
        if args.rca_checkpoint_proxy_weight > 0:
            score_hyper_params["use_rca_aware_checkpoint"] = True
    if args.rca_checkpoint_proxy_batches is not None:
        score_hyper_params["rca_checkpoint_proxy_batches"] = max(
            0, int(args.rca_checkpoint_proxy_batches)
        )
    if args.rca_checkpoint_min_epoch is not None:
        score_hyper_params["rca_checkpoint_min_epoch"] = max(
            1, int(args.rca_checkpoint_min_epoch)
        )
    if args.enable_visualization_hooks:
        score_hyper_params["enable_visualization_hooks"] = True
    if args.debug_loss_breakdown:
        score_hyper_params["debug_loss_breakdown"] = True
    if args.debug_loss_log_path is not None:
        score_hyper_params["debug_loss_log_path"] = args.debug_loss_log_path
        score_hyper_params["debug_loss_breakdown"] = True
    if args.debug_loss_max_batches is not None:
        score_hyper_params["debug_loss_max_batches"] = max(0, int(args.debug_loss_max_batches))
        score_hyper_params["debug_loss_breakdown"] = True
    if args.debug_train_max_batches is not None:
        score_hyper_params["debug_train_max_batches"] = max(0, int(args.debug_train_max_batches))
    if args.robust_input_preprocess:
        score_hyper_params["use_robust_input_preprocess"] = True
    if args.input_clip_lower_quantile is not None:
        score_hyper_params["use_robust_input_preprocess"] = True
        score_hyper_params["input_clip_lower_quantile"] = min(
            max(float(args.input_clip_lower_quantile), 0.0), 0.5
        )
    if args.input_clip_upper_quantile is not None:
        score_hyper_params["use_robust_input_preprocess"] = True
        score_hyper_params["input_clip_upper_quantile"] = min(
            max(float(args.input_clip_upper_quantile), 0.5), 1.0
        )
    effective_switches = {
        **arch_switches,
        **vq_hyper_params,
        **score_hyper_params,
    }
    if bool(effective_switches.get("use_latest_checkpoint", False)):
        actual_checkpoint_policy = "latest"
    elif bool(effective_switches.get("use_rca_aware_checkpoint", False)):
        if bool(effective_switches.get("rca_checkpoint_normalize", False)):
            actual_checkpoint_policy = "normalized_rca_aware"
        else:
            actual_checkpoint_policy = "rca_aware"
    else:
        actual_checkpoint_policy = "best_val_loss"
    score_hyper_params["checkpoint_policy"] = actual_checkpoint_policy
    effective_switches["checkpoint_policy"] = actual_checkpoint_policy

    model_config = {
        "models": [
            {
                "adapter": None,
                "model_name": "self_impl.LaGraph.LaGraph.LaGraph",
                "model_hyper_params": {
                    # v10 固定训练参数
                    "num_epochs": train_epochs,
                    "n_gpus": n_gpus,
                    "batch_size": per_gpu_batch,
                    "lr": scaled_lr,
                    "warmup_epochs": warmup_epochs,
                    "dataloader_num_workers": dataloader_num_workers,
                    "dataloader_prefetch_factor": dataloader_prefetch_factor,
                    "inference_dataloader_num_workers": inference_dataloader_num_workers,
                    "export_rca": args.export_rca,
                    **arch_switches,
                    **vq_hyper_params,
                    **score_hyper_params,
                    "d_model": d_model_scale,
                    "e_layers": e_layers_scale,
                    "n_heads": n_heads_scale,
                },
            }
        ]
    }

    # 打印 v10 配置摘要
    print(f"  [v10 配置] d_model={d_model_scale}, e_layers={e_layers_scale}, n_heads={n_heads_scale}, batch_per_gpu={per_gpu_batch}")
    print(f"  [v10 配置] LR={scaled_lr:.1e}, warmup={warmup_epochs} epochs")
    print(f"  [v10 数据] workers={dataloader_num_workers}, prefetch={dataloader_prefetch_factor}")
    print(f"  [v10 架构] profile={args.arch_profile}, switches={effective_switches}")
    print()

    print(f"  [protocol] max_epochs={train_epochs}, checkpoint_policy={actual_checkpoint_policy}")

    with open(os.path.join(CONFIG_PATH, args.eval_config), "r") as f:
        evaluation_config = json.load(f)["evaluation_config"]
    evaluation_config["strategy_args"]["seed"] = args.seed

    # ---- 运行前摘要 ----
    t_start = time.time()
    data_src = LocalAnomalyDetectDataSource()
    matched = data_src.dataset.metadata["file_name"].tolist()
    # 排除不兼容数据集后再取值
    if args.datasets:
        matched = [f for f in matched if f in args.datasets]
    else:
        matched = [f for f in matched if not _is_default_excluded(f)]

    strategy_name = evaluation_config["strategy_args"].get("strategy_name", "")
    if strategy_name.startswith("unfixed"):
        meta_by_name = data_src.dataset.metadata.set_index("file_name")
        missing_train_lens = [
            name
            for name in matched
            if "train_lens" not in meta_by_name.columns
            or pd.isna(meta_by_name.loc[name].get("train_lens"))
        ]
        if missing_train_lens:
            raise ValueError(
                "The selected evaluation config requires train_lens metadata, "
                f"but these datasets do not have it: {missing_train_lens}. "
                "Use --eval-config all_detect_label_config.json or choose TE_MM_/HAI_ datasets."
            )

    print()
    print("=" * 58)
    print("  LaGraph")
    print("=" * 58)
    print(f"  epochs        : {train_epochs}")
    print(f"  save dir      : result/{args.save_dir}/")
    print(f"  eval config   : {args.eval_config}")
    print(f"  strategy      : {strategy_name}")
    print(f"  datasets found: {len(matched)}")
    for name in matched:
        row = data_src.dataset.metadata.set_index("file_name").loc[name]
        rows_val = row.get("length", "?")
        if pd.notna(rows_val):
            rows_str = f"{int(rows_val):>8,}"
        else:
            rows_str = f"{'?':>8}"
        uv_val = row.get("if_univariate", "?")
        print(f"    - {name:<20s}  rows={rows_str}  univariate={uv_val}")
    _diagnose_cuda()
    print("=" * 58)
    print()

    ParallelBackend().init(backend="sequential", n_workers=1, n_cpus=1)

    try:
        log_filenames = pipeline(
            data_config, model_config, evaluation_config, save_path=args.save_dir,
        )
    finally:
        ParallelBackend().close(force=True)

    elapsed = time.time() - t_start
    print()
    print("=" * 58)
    print(f"  Finished in {elapsed:.1f}s")
    for fn in log_filenames:
        print(f"    -> result/{args.save_dir}/{os.path.basename(fn)}")
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
