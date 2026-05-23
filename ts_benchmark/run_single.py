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
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="指定要跑的数据集文件名 (如 swat.csv MSL.csv)，不指定则跑全部",
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
        "--arch-profile",
        choices=["full", "core", "no-vq", "no-boundary", "reconstruction"],
        default="full",
        help="架构配置: full=最终精简架构(已移除 Boundary); no-boundary 为兼容别名; no-vq/core 关闭 VQ; reconstruction 仅保留重建主干",
    )
    parser.add_argument(
        "--pred-head",
        action="store_true",
        help="启用预测分支（Phase 2：解重建悖论）",
    )
    parser.add_argument(
        "--lambda-pred",
        type=float,
        default=0.1,
        help="预测损失权重 (default: 0.1)",
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
        "--ablation",
        type=str,
        nargs="*",
        default=None,
        help="v10 消融模式: 默认全部关闭, 要启用则写 --ablation contrastive/freq/prototype/pred",
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
        "SMD.csv",        # 数据量过大，默认实验先排除；可用 --datasets SMD.csv 单独运行
        "SKAB_all.csv",   # 标签格式不兼容
    }
    if args.datasets:
        data_config["data_name_list"] = args.datasets
    else:
        # 默认排除不适用的数据集
        data_src = LocalAnomalyDetectDataSource()
        all_files = data_src.dataset.metadata["file_name"].tolist()
        filtered = [f for f in all_files if f not in EXCLUDED_FILES]
        data_config["data_name_list"] = filtered
        excluded = [f for f in all_files if f in EXCLUDED_FILES]
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

    # ---- v10: 所有辅助模块默认关闭（必须用户显式 --ablation 才会启用） ----
    use_pred_head = args.pred_head
    lambda_pred = args.lambda_pred

    ablation = args.ablation or []
    # v10 默认所有辅助模块关闭（若要开启必须显式 --ablation 指定）
    use_contrastive = "contrastive" in ablation   # 默认 False，需 --ablation contrastive
    use_prototype = "prototype" in ablation       # 默认 False，需 --ablation prototype
    use_freq_loss = "freq" in ablation            # 默认 False，需 --ablation freq
    if "pred" in ablation:
        use_pred_head = True

    arch_profiles = {
        "full": {
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
                    **arch_switches,
                    **vq_hyper_params,
                    "d_model": d_model_scale,
                    "e_layers": e_layers_scale,
                    "n_heads": n_heads_scale,
                    # v10 默认关闭所有辅助模块
                    "use_prediction_head": use_pred_head,
                    "lambda_pred": lambda_pred,
                    "use_contrastive": use_contrastive,
                    "use_prototype": use_prototype,
                    "use_freq_loss": use_freq_loss,
                },
            }
        ]
    }

    # 打印 v10 配置摘要
    print(f"  [v10 配置] d_model={d_model_scale}, e_layers={e_layers_scale}, n_heads={n_heads_scale}, batch_per_gpu={per_gpu_batch}")
    print(f"  [v10 配置] LR={scaled_lr:.1e}, warmup={warmup_epochs} epochs")
    print(f"  [v10 数据] workers={dataloader_num_workers}, prefetch={dataloader_prefetch_factor}")
    print(f"  [v10 架构] profile={args.arch_profile}, switches={arch_switches}")
    print(f"  [v10 模块] prediction={use_pred_head}, contrastive={use_contrastive}, freq={use_freq_loss}, prototype={use_prototype}")
    print(f"  [v10 提示] 如需启用辅助模块，使用 --ablation contrastive/freq/prototype/pred")
    print()

    with open(os.path.join(CONFIG_PATH, EVAL_CONFIG), "r") as f:
        evaluation_config = json.load(f)["evaluation_config"]

    # ---- 运行前摘要 ----
    t_start = time.time()
    data_src = LocalAnomalyDetectDataSource()
    matched = data_src.dataset.metadata["file_name"].tolist()
    # 排除不兼容数据集后再取值
    if args.datasets:
        matched = [f for f in matched if f in args.datasets]
    else:
        matched = [f for f in matched if f not in EXCLUDED_FILES]

    print()
    print("=" * 58)
    print("  LaGraph")
    print("=" * 58)
    print(f"  epochs        : {train_epochs}")
    print(f"  save dir      : result/{args.save_dir}/")
    print(f"  eval config   : {EVAL_CONFIG}")
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
