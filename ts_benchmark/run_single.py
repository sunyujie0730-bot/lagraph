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
        "--use-latest-checkpoint",
        action="store_true",
        help="Use the latest trained checkpoint for evaluation instead of the best validation-loss checkpoint.",
    )
    parser.add_argument(
        "--enable-visualization-hooks",
        action="store_true",
        help="Enable diagnostic visualization hooks and vis_data.json export. Disabled by default for fast experiments.",
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
            "rca_graph_penalty_weight": 0.10,
            "rca_mechanism_weight": 0.0,
            "rca_source_weight": 1.0,
            "rca_source_base_weight": 1.0,
            "rca_propagation_weight": 0.0,
            "rca_source_mechanism_weight": 0.25,
            "rca_causal_weight": 0.0,
            "rca_onset_weight": 0.25,
            "rca_onset_baseline_window": 300,
            "rca_onset_z": 2.0,
            "rca_mechanism_residual_window": 300,
            "rca_mechanism_residual_weight": 0.15,
            "rca_event_head_ratio": 1.0,
            "rca_event_head_points": 0,
            "rca_prediction_key": "15",
            "rca_event_local_export": True,
            "rca_event_local_margin": 100,
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
    if args.synthetic_rca_bce_weight is not None:
        score_hyper_params["synthetic_rca_bce_weight"] = max(0.0, float(args.synthetic_rca_bce_weight))
    if args.synthetic_rca_rank_weight is not None:
        score_hyper_params["synthetic_rca_rank_weight"] = max(0.0, float(args.synthetic_rca_rank_weight))
    if args.synthetic_score_weight is not None:
        score_hyper_params["synthetic_score_weight"] = max(0.0, args.synthetic_score_weight)
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
    if args.rca_propagation_weight is not None:
        score_hyper_params["rca_propagation_weight"] = max(0.0, float(args.rca_propagation_weight))
    if args.rca_source_mechanism_weight is not None:
        score_hyper_params["rca_source_mechanism_weight"] = max(0.0, float(args.rca_source_mechanism_weight))
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
    if args.use_latest_checkpoint:
        score_hyper_params["use_latest_checkpoint"] = True
    if args.enable_visualization_hooks:
        score_hyper_params["enable_visualization_hooks"] = True
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
