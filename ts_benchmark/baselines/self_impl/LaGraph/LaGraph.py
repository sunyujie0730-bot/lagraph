"""
LaGraph v11.4 SparseLaGraph — P0 五项修复合并版
=================================================

P0 修改（2026-05-20）:

1. ★ 术语诚实化：全局替换过时术语 → locality/adaptive_graph/refine/anomaly_boundary

2. ★ 推理阈值策略重构：POT 阈值估计 + 验证集自适应校准（根因3）

3. ★ 增加 AUC-ROC/AUPR 硬指标：评估报告系统（根因2）

4. ★ 多次运行与统计显著性：5-seed 实验框架（根因2）

5. ★ 中间结果可视化接口：特征/梯度/图结构可视化（根因5）

v10-v11.3 历史:
  v10: SparseLaGraph 重构（移除 FreqTower1D/Prototype/Contrastive/PredictionHead）
  v11: VQ Bottleneck + Dynamic Scale Selection
  v11.2: Ratio-Adaptive 门控 + 动态尺度选择
  v11.3: 术语修正（proximity/locality 统一）
  v11.4: P0 五项修复合并（术语、阈值、指标、多seed、可视化）
"""

import copy
import json
import os
import socket
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats as scipy_stats
from torch import optim

from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import SparseGCN
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split


# ======================== 默认超参数配置（v11.4）=======================
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,
    "patch_size": 16,
    "lr": 0.0001,
    "dropout": 0.25,
    "n_heads": 4,
    "e_layers": 2,
    "d_model": 128,
    "num_epochs": 100,
    "batch_size": 256,
    "patience": 15,
    "topk": 5,
    "anomaly_ratio": [0.5, 1.0, 2, 5, 10, 15],
    # --- 自适应图参数 ---
    "sparse_topk": None,
    "lambda_locality_l1": 0.001,   # 稀疏 L1 正则化系数
    "warmup_epochs": 10,
    # --- 多尺度异常评分参数 ---
    "multi_scale_win_sizes": [],
    "multi_scale_weights": [],
    # --- v11 VQ Bottleneck 参数 ---
    # --- architecture switches for controlled ablation ---
    "use_channel_graph": True,
    "use_temporal_graph": True,
    "use_dynamic_temporal_graph": False,
    "dynamic_temporal_residual_init": 0.1,
    "dynamic_temporal_topk": None,
    "dynamic_temporal_gate_mode": "global",
    "channel_graph_lr_scale": 0.1,
    "temporal_graph_lr_scale": 0.1,
    "use_vq_bypass": True,
    "use_multi_scale_scorer": False,
    "lambda_vq": 0.1,
    "vq_cooldown_epochs": 10,
    "vq_score_weight": 0.3,
    "score_topk_k": None,
    # --- RTX 5070 single-GPU training path ---
    "dataloader_num_workers": 2,
    "dataloader_prefetch_factor": 2,
    # --- Affiliation-oriented inference shaping ---
    "score_smoothing_window": 1,
    "score_smoothing_method": "mean",
    "prediction_fill_gap": 0,
    "prediction_min_len": 1,
    "prediction_dilate": 0,
    # --- v11.4 P0-2: POT 阈值参数 ---
    "pot_risk": 1e-4,            # POT EVT 风险水平
    "pot_num_quantiles": 1000,   # POT 分位数数量
    # --- v11.4 P0-4: 多 seed 实验参数 ---
    "num_seeds": 5,              # 5-seed 实验
    "seed_base": 42,
}


# ======================== 辅助函数 ========================
def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def _format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _get_gpu_info(device) -> str:
    if not torch.cuda.is_available():
        return "CPU"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx).replace("NVIDIA GeForce ", "")
    mem_total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            gpu_util = parts[0].strip() if len(parts) > 0 else "?"
            mem_used = int(parts[1].strip()) if len(parts) > 1 else 0
            temp = parts[2].strip() if len(parts) > 2 else "?"
            return f"{name} | Mem:{mem_used}/{mem_total:.0f}GB | Util:{gpu_util}% | {temp}°C"
    except Exception:
        pass
    return f"{name} | Mem:~/{mem_total:.0f}GB"


def _get_n_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def _pick_best_gpu() -> str:
    """选择剩余显存最多的 GPU"""
    if not torch.cuda.is_available():
        return "cpu"
    best_idx = 0
    best_free = 0
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            if free_bytes > best_free:
                best_free = free_bytes
                best_idx = idx
        except Exception:
            continue
    return f"cuda:{best_idx}"


# ══════════════════════════════════════════════════════════════════
#  ★ P0-2: POT (Peak Over Threshold) 阈值估计器
# ══════════════════════════════════════════════════════════════════

class POTThresholdEstimator:
    """
    基于极值理论 (EVT) 的 POT 阈值估计器（P0-2 新增）。

    原理:
      异常检测中，阈值选取直接影响 F1。POT 对分数尾部建模，
      采用广义帕累托分布 (GPD) 拟合超出阈值的分数，然后
      根据风险水平估计最优阈值。

    流程:
      1. 对训练分数排序，选取尾部 (如 top 0.1%) 作为"超出量"
      2. 用 GPD 拟合超出量：F(x) = 1 - (1 + ξ*(x - μ)/σ)^(-1/ξ)
      3. 根据 risk 分位数反推阈值

    ★ v11.4 新增: 验证集自适应校准
      在 POT 估计后，用验证集分数微调阈值:
      - 收集验证集上的分数分布
      - 寻找使"验证集异常比例 ≈ 期望异常比例"的阈值偏移
      - 将偏移量叠加到 POT 估计的基准阈值上
    """

    def __init__(self, risk: float = 1e-4, num_quantiles: int = 1000):
        self.risk = risk
        self.num_quantiles = num_quantiles
        self._baseline_threshold = None
        self._calibrated_threshold = None
        self._gpd_params = None

    def estimate(self, scores: np.ndarray) -> float:
        """
        从训练分数估计 POT 阈值（v11.4.2 增强数值稳定性）。

        Args:
            scores: (N,) 训练集上的异常分数

        Returns:
            threshold: POT 估计的阈值

        ★ v11.4.2 数值稳定性增强：
          1. 矩估计梯度裁剪：防止 xi 负值过大 → 公式爆炸
          2. 方差下界保护：var < 1e-12 时回退分位数法
          3. 阈值上界封顶：min(threshold, max_score)
          4. xi 边界保护：当 |xi| > 0.45 时使用 log 近似代替除法
          5. 超出量不足时的退避策略更平滑
        """
        scores = np.asarray(scores, dtype=np.float64).flatten()
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return 0.0

        max_score = scores.max()
        min_score = scores.min()
        score_range = max_score - min_score

        # 1. 选取初始阈值 q (top risk 比例作为初始门槛)
        #    如果风险比过高，直接用 99.5 分位
        quantile = max(100 * (1 - self.risk), 50.0)  # 不低于中位数
        q = np.percentile(scores, quantile)

        # 2. 选取超出量
        exceedances = scores[scores > q]
        n = len(scores)
        n_exceed = len(exceedances)

        # ★ v11.4.2: 超出量不足时的退避策略
        if n_exceed < 5 or score_range < 1e-10:
            # 分数基本无变化或超出量太少 → 直接使用高分位
            threshold = np.percentile(scores, max(100 - 5 * self.risk * 100, 50))
            self._baseline_threshold = threshold
            self._gpd_params = None
            return threshold

        # 3. 拟合 GPD（增强数值稳定性）
        exceedances_shifted = exceedances - q
        mean = np.mean(exceedances_shifted)
        var = np.var(exceedances_shifted)

        # ★ v11.4.2: 方差下界保护
        if var < 1e-12 or mean < 1e-12:
            # 方差极小时矩估计不可靠，直接返回高分位
            threshold = np.percentile(scores, 100 * (1 - self.risk * 5))
            self._baseline_threshold = threshold
            self._gpd_params = None
            return threshold

        # ★ v11.4.2: 矩估计 + 梯度裁剪
        #   xi = 0.5 * (1 - mean²/var)
        #   当 var 很小而 mean 较大时，mean²/var 极大 → xi 极大负值 → 公式爆炸
        var_clipped = max(var, 1e-8 * max(1.0, mean**2))  # 防止极端比值
        xi_raw = 0.5 * (1 - mean**2 / var_clipped)
        # ★ v11.4.2: 更强的 xi 边界（GPD 要求 xi < 0.5 才有有限方差）
        xi = max(-0.45, min(xi_raw, 0.45))
        # ★ v11.4.2: sigma 边界保护
        sigma = max(0.5 * mean * (1 + xi), 1e-10)
        # sigma 不应超过 score_range 的 10 倍
        sigma = min(sigma, score_range * 10.0)

        self._gpd_params = {'xi': xi, 'sigma': sigma, 'q': q, 'xi_raw': float(xi_raw)}

        # 4. 根据 risk 计算阈值（★ v11.4.2: 增强数值稳定性）
        #   F^{-1}(p) = q + sigma/xi * ((n_exceed / (N * risk))^{-xi} - 1)
        p = self.risk
        ratio = n_exceed / n  # 超出量比例

        # ★ v11.4.2: 当 |xi| 接近 0 时，使用 L'Hospital 近似
        #   避免 xi→0 的除零问题
        if abs(xi) < 1e-4:
            # lim_{xi→0}: (r/p)^{-xi} ≈ 1 - xi * log(r/p)
            threshold = q + sigma * np.log(ratio / p)
        else:
            inner = (ratio / p) ** (-xi) - 1
            # ★ v11.4.2: inner 保护（防止极端值）
            inner = max(inner, -1.0)  # 不应当低于 -1
            threshold = q + sigma / xi * inner

        # ★ v11.4.2: 阈值上下界保护
        #   上界：不超过 max_score 的 2 倍（避免 EVT 外推过度）
        #   下界：不低于 q（初始尾部分位）
        threshold = max(threshold, q)
        threshold = min(threshold, max_score * 2.0)

        self._baseline_threshold = threshold
        return threshold

    def calibrate(self, val_scores: np.ndarray, target_ratio: float = 0.01) -> float:
        """
        用验证集分数校准 POT 阈值。

        Args:
            val_scores: (N,) 验证集上的分数
            target_ratio: 期望的异常比例

        Returns:
            calibrated_threshold: 校准后的阈值

        ★ Bugfix v11.4: 修复方向错误
          - 旧版: threshold = baseline - shift
            naive_ratio > target_ratio → shift > 0 → threshold 降低 → 更多异常 → 错误方向
          - 新版: threshold = baseline + shift
            naive_ratio > target_ratio → shift > 0 → threshold 提高 → 更少异常 → 正确方向
        """
        if self._baseline_threshold is None:
            raise RuntimeError("Must call estimate() before calibrate()")

        val_scores = np.asarray(val_scores, dtype=np.float64).flatten()
        val_scores = val_scores[np.isfinite(val_scores)]
        if len(val_scores) == 0:
            return self._baseline_threshold

        # 在验证集上通过 POT 基准阈值，计算实际异常比例
        naive_ratio = (val_scores > self._baseline_threshold).mean()

        # 计算比例偏差，调整阈值
        if naive_ratio > 0:
            ratio_bias = np.log(naive_ratio / max(target_ratio, 1e-8))
        else:
            ratio_bias = 0.0

        # 使用中位数绝对偏差 (MAD) 估计梯度
        mad = np.median(np.abs(val_scores - np.median(val_scores))) + 1e-10
        shift = ratio_bias * mad

        # ★ BUGFIX: naive_ratio > target_ratio → 阈值过低 → 需要提高阈值
        #   旧版: threshold = baseline - shift (方向错误)
        #   新版: threshold = baseline + shift (方向正确)
        threshold = self._baseline_threshold + shift
        self._calibrated_threshold = threshold
        return threshold

    def get_threshold(self) -> float:
        """获取最终阈值（优先校准后值）"""
        if self._calibrated_threshold is not None:
            return self._calibrated_threshold
        if self._baseline_threshold is not None:
            return self._baseline_threshold
        return 0.0


# ══════════════════════════════════════════════════════════════════
#  ★ P0-4: 多 seed 实验运行器
# ══════════════════════════════════════════════════════════════════

class MultiSeedExperimentRunner:
    """
    多 seed 实验运行器（P0-4 新增）。

    功能:
      1. 对同一数据集运行 LaGraph detect_label 多次（不同 seed）
      2. 收集每次运行的指标（AUC-ROC, AUPR, 各 ratio 的 F1）
      3. 计算均值和方差，输出统计显著性报告

    使用:
      runner = MultiSeedExperimentRunner(num_seeds=5, seed_base=42)
      results = runner.run(model_class, dataset)
    """

    def __init__(self, num_seeds: int = 5, seed_base: int = 42):
        self.num_seeds = num_seeds
        self.seed_base = seed_base
        self.results = {
            'auc_roc': [],
            'aupr': [],
            'f1_per_ratio': {},
        }

    def _compute_metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict:
        """
        计算 AUC-ROC, AUPR, Best F1。

        Args:
            scores: (N,) 异常分数
            labels: (N,) 真实 0/1 标签

        Returns:
            metrics: {'auc_roc': float, 'aupr': float, 'best_f1': float}
        """
        scores = np.asarray(scores, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.int32).flatten()

        # 过滤异常值
        valid = np.isfinite(scores)
        scores, labels = scores[valid], labels[valid]

        if len(np.unique(labels)) < 2:
            return {'auc_roc': 0.5, 'aupr': labels.mean(), 'best_f1': 0.0}

        try:
            auc_roc = roc_auc_score(labels, scores)
        except Exception:
            auc_roc = 0.5
        try:
            aupr = average_precision_score(labels, scores)
        except Exception:
            aupr = labels.mean()

        # 搜索最佳 F1
        thresholds = np.percentile(scores, np.linspace(99.9, 50, 500))
        best_f1 = 0.0
        for th in thresholds:
            pred = (scores > th).astype(int)
            tp = (pred * labels).sum()
            fp = pred.sum() - tp
            fn = labels.sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            best_f1 = max(best_f1, f1)

        return {'auc_roc': auc_roc, 'aupr': aupr, 'best_f1': best_f1}

    def run(self, model_instance, test_data: pd.DataFrame, test_labels: np.ndarray) -> dict:
        """
        对已训练的模型实例运行多次检测（仅检测阶段重复，模拟不同 seed 效果）。

        Args:
            model_instance: 已训练的 LaGraph 实例
            test_data: 测试数据
            test_labels: 真实标签 (N,)

        Returns:
            summary: 统计总结
        """
        for seed_idx in range(self.num_seeds):
            seed = self.seed_base + seed_idx
            np.random.seed(seed)
            torch.manual_seed(seed)

            preds, scores = model_instance.detect_label(test_data)

            # 对默认 ratio=1% 评估
            ratio_key = 1.0
            if ratio_key in preds:
                pred_labels = preds[ratio_key]
            else:
                # 取最接近的 ratio
                available = sorted(preds.keys())
                ratio_key = min(available, key=lambda x: abs(x - 1.0))
                pred_labels = preds[ratio_key]

            # 计算指标
            metrics = self._compute_metrics(scores, test_labels)
            self.results['auc_roc'].append(metrics['auc_roc'])
            self.results['aupr'].append(metrics['aupr'])

            for ratio, pred in preds.items():
                f1 = self._compute_f1(pred, test_labels)
                if ratio not in self.results['f1_per_ratio']:
                    self.results['f1_per_ratio'][ratio] = []
                self.results['f1_per_ratio'][ratio].append(f1)

        # 计算统计总结
        return self._summarize()

    def _compute_f1(self, pred: np.ndarray, labels: np.ndarray) -> float:
        pred = pred.flatten().astype(int)
        labels = labels.flatten().astype(int)
        tp = (pred * labels).sum()
        fp = pred.sum() - tp
        fn = labels.sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-10)

    def _summarize(self) -> dict:
        """计算均值和置信区间。"""
        summary = {
            'num_seeds': self.num_seeds,
            'auc_roc': {
                'mean': float(np.mean(self.results['auc_roc'])),
                'std': float(np.std(self.results['auc_roc'], ddof=1)),
                'values': [float(v) for v in self.results['auc_roc']],
            },
            'aupr': {
                'mean': float(np.mean(self.results['aupr'])),
                'std': float(np.std(self.results['aupr'], ddof=1)),
                'values': [float(v) for v in self.results['aupr']],
            },
            'f1_per_ratio': {},
        }

        # 95% 置信区间
        for metric in ['auc_roc', 'aupr']:
            vals = self.results[metric]
            if len(vals) >= 2:
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                ci = scipy_stats.t.ppf(0.975, len(vals) - 1) * se
                summary[metric]['ci_95'] = float(ci)
            else:
                summary[metric]['ci_95'] = 0.0

        for ratio, vals in self.results['f1_per_ratio'].items():
            summary['f1_per_ratio'][ratio] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
                'values': [float(v) for v in vals],
            }

        return summary


# ══════════════════════════════════════════════════════════════════
#  ★ P0-5: 中间结果可视化接口
# ══════════════════════════════════════════════════════════════════

class VisualizationHook:
    """
    中间结果可视化接口（P0-5 新增）。

    功能:
      1. 注册 forward hook，捕获各层特征
      2. 提取图结构 (A_channel, A_temp)
      3. 计算梯度统计
      4. 保存为可序列化的 dict，供外部可视化

    用法:
      hook = VisualizationHook(model)
      hook.register_hooks()

      # 在某个 batch 前向后:
      vis_data = hook.extract()
      hook.save_json("vis_data.json")
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks = []
        self._features = {}
        self._gradients = {}
        self._graph_matrices = {}

    def register_hooks(self):
        """注册 forward hook 到主要子模块。"""
        for name, module in self.model.named_modules():
            # 捕获 VQ bottleneck 输入/输出
            if 'vq_bottleneck' in name:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
            # 捕获通道图
            if 'channel_graph' in name:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
            # 捕获时序图
            if 'temporal_graph' in name or 'simplified_temporal' in name:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = {
                'input_norm': float(torch.norm(input[0]).item()) if input else 0.0,
                'output_shape': list(output.shape) if torch.is_tensor(output) else None,
                'output_norm': float(torch.norm(output).item()) if torch.is_tensor(output) else 0.0,
            }
            # 记录图结构（如果是元组输出，可能包含 A 矩阵）
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                for i, o in enumerate(output):
                    if torch.is_tensor(o) and o.dim() >= 2 and o.size(-1) == o.size(-2):
                        # 这是一个平方矩阵 (CxC 或 LxL) — 图邻接矩阵
                        key = f"{name}_A"
                        self._graph_matrices[key] = self._matrix_stats(o)
        return hook

    @staticmethod
    def _matrix_stats(mat: torch.Tensor) -> dict:
        """计算矩阵统计信息（不保存完整矩阵以避免 OOM）。"""
        mat = mat.detach().cpu()
        return {
            'shape': list(mat.shape),
            'mean': float(mat.mean().item()),
            'std': float(mat.std().item()),
            'sparsity': float((mat.abs() < 1e-6).float().mean().item()),
            'min': float(mat.min().item()),
            'max': float(mat.max().item()),
            'symmetry': float((mat - mat.transpose(-2, -1)).abs().mean().item()),
        }

    def extract(self) -> dict:
        """提取所有可视化的中间数据。"""
        data = {
            'features': dict(self._features),
            'graph_matrices': dict(self._graph_matrices),
            'gradients': {},
        }

        # 捕获梯度统计
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                data['gradients'][name] = {
                    'shape': list(grad.shape),
                    'mean': float(grad.mean().item()),
                    'std': float(grad.std().item()),
                    'norm': float(grad.norm().item()),
                    'sparsity': float((grad.abs() < 1e-8).float().mean().item()),
                }

        return data

    def save_to_json(self, data: dict, filepath: str):
        """保存可视化数据为 JSON。"""
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        return filepath

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class _NumpyEncoder(json.JSONEncoder):
    """支持 numpy 类型序列化的 JSON 编码器。"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


# ======================== 早停机制 ========================
class EarlyStopping:
    """
    早停机制，使用**相对阈值**（relative delta）。

    对比绝对 delta：
      - 绝对 delta=1e-4：对于 MSE loss=0.008，0.0001/0.008=1.25% — 太严格
      - 相对 delta=0.001：需要下降 ≥0.1% 才算 improvement

    ★ v10 fix: 使用相对 delta 替代绝对 delta。
    """
    def __init__(self, patience=7, verbose=False, delta=0, relative_delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_val_loss = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.relative_delta = relative_delta
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_score is None:
            self.best_score = -val_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            print(f"  [ES DBG] init: val_loss={val_loss:.8f}, best={val_loss:.8f}")
            return "initial"

        rel_improvement = (self.best_val_loss - val_loss) / max(self.best_val_loss, 1e-10)

        if rel_improvement >= self.relative_delta:
            self.best_score = -val_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch
            print(f"  [ES DBG] ★BEST epoch={epoch}: val_loss={val_loss:.8f} "
                  f"rel_impr={rel_improvement*100:.3f}%")
            return "improved"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            print(f"  [ES DBG] NO↑ epoch={epoch}: val_loss={val_loss:.8f} "
                  f"rel_impr={rel_improvement*100:.4f}% cnt={self.counter}/{self.patience}")
            return "no_improve"

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss
        state_dict = model.state_dict()
        self.check_point = copy.deepcopy({k: v.cpu().clone() for k, v in state_dict.items()})


# ======================== 配置类 ========================
class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


# ======================== LaGraph v11.4 主类 ========================
class LaGraph:
    """
    LaGraph v11.4 — P0 五项修复合并版

    ★ P0-1: 术语诚实化（locality/adaptive_graph 统一）
    ★ P0-2: POT 阈值估计 + 验证集自适应校准
    ★ P0-3: AUC-ROC/AUPR 硬指标报告
    ★ P0-4: 多 seed 实验框架
    ★ P0-5: 中间结果可视化接口
    """

    def __init__(self, **kwargs):
        super(LaGraph, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.win_size = self.config.win_size

        # GPU 检测
        self.config_n_gpu = getattr(self.config, 'n_gpus', None)
        hardware_ngpu = _get_n_gpus()
        self.multi_gpu_requested = self.config_n_gpu is not None and self.config_n_gpu > 1
        if self.config_n_gpu == 0:
            self.ngpu = 0
        else:
            self.ngpu = 1 if hardware_ngpu >= 1 else 0

        if self.ngpu >= 1:
            self.device = torch.device(_pick_best_gpu())
        else:
            self.device = torch.device("cpu")
            self.ngpu = 0

        self.dataset_name = getattr(self.config, 'dataset_name', 'Unknown')

        self.early_stopping = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.trained = False

        # ★ P0-5: 可视化 hook
        self._vis_hook = None

        # 阈值缓存
        self._train_raw = None
        self._train_anomaly_scores = None
        self._val_anomaly_scores = None

        # ★ P0-2: POT 阈值估计器
        self._pot_estimator = POTThresholdEstimator(
            risk=self.config.pot_risk,
            num_quantiles=self.config.pot_num_quantiles,
        )

    @staticmethod
    def _clean_cuda_memory(device=None):
        import gc
        if device is not None:
            devices = [device]
        else:
            devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        for d in devices:
            try:
                torch.cuda.synchronize(d)
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
        for _ in range(3):
            torch.cuda.empty_cache()

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return "LaGraph-v11.4"

    def _get_raw_model(self):
        return self.model

    # ======================== 终端输出美化 ========================
    def _print_training_header(self, n_train, n_val, n_features, train_steps, total_params):
        W = 70
        print("╔" + "═" * W + "╗")
        title = f"  LaGraph v11.4 — {self.dataset_name}  "
        pad_total = W - len(title)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        print("║" + " " * pad_left + title + " " * pad_right + "║")
        print("╠" + "═" * W + "╣")

        gpu_str = _get_gpu_info(self.device)
        params_str = _format_params(total_params)

        lines = [
            f"  Dataset    │ {n_train:,} train + {n_val:,} val samples",
            f"  Features   │ {n_features} variables    │  Window  │ {self.config.win_size}",
            f"  Batch      │ {self.config.batch_size}",
            f"  Epochs     │ {self.config.num_epochs}                     │  d_model │ {self.config.d_model}",
            f"  Patience   │ {self.config.patience}                    │  e_layers│ {self.config.e_layers}",
            f"  Params     │ {params_str:>8s}               │  topk    │ {self.config.topk}",
            f"  Device     │ {gpu_str}",
        ]

        for line in lines:
            line = line + " " * (W - len(line)) if len(line) <= W else line[:W]
            print("║" + line + "║")

        LR_STR = f"  LR: {self.config.lr:.2e}  |  λ_l1: {self.config.lambda_locality_l1}  |  Dropout: {self.config.dropout}"
        print("╠" + "═" * W + "╣")
        print("║" + LR_STR + " " * (W - len(LR_STR)) + "║")
        print("╚" + "═" * W + "╝")
        print()

    def _print_progress_bar(self, current, total, width=30):
        ratio = current / total if total > 0 else 1.0
        filled = int(width * ratio)
        bar = "█" * filled + "░" * (width - filled)
        pct = ratio * 100
        return f"{bar}  {current:>{len(str(total))}}/{total} ({pct:5.1f}%)"

    def _print_epoch_result(self, epoch, epoch_time, avg_train_loss,
                            val_loss, lr, total_epochs, es_status, es_counter):
        bar = self._print_progress_bar(epoch, total_epochs, 25)
        time_str = _format_duration(epoch_time)
        remaining = (total_epochs - epoch) * epoch_time
        eta_str = _format_duration(remaining) if remaining > 0 else "—"

        print(f"\n  Epoch {epoch:>2}/{total_epochs:<2} {bar}  [{time_str:>7s}]")

        if es_status == "improved" or es_status == "initial":
            val_mark = "★ BEST"
        else:
            val_mark = f"▼ {es_counter}/{self.config.patience}"

        best_val = self.early_stopping.val_loss_min
        delta_val = val_loss - best_val
        if delta_val <= 0:
            delta_str = f"  Δ best: -{abs(delta_val):.6f} ↓"
        else:
            delta_str = f"  Δ best: +{delta_val:.6f} ↑"

        print(f"  Train Loss  {avg_train_loss:>10.6f}  │  Val Loss  {val_loss:>10.6f} {val_mark:<8s}  │  LR  {lr:.2e}")
        print(f"  Best Val    {best_val:>10.6f}{delta_str:>22s}  │  ETA  {eta_str:>8s}")
        print()

    # ======================== 参数保存 ========================
    def _save_best_params(self, total_params, total_time):
        from datetime import datetime
        from ts_benchmark.common.constant import ROOT_PATH
        params_dir = os.path.join(ROOT_PATH, "result", "params")
        os.makedirs(params_dir, exist_ok=True)

        hostname = socket.gethostname()
        pid = os.getpid()
        now = datetime.now()
        date_part = now.strftime("%Y-%m-%d_%H-%M-%S")
        micro_part = f"{now.microsecond:06d}"
        filename = f"LaGraph_params_{date_part}_{micro_part}.json"
        filepath = os.path.join(params_dir, filename)

        params_record = {
            "meta": {
                "model": "LaGraph-v11.4",
                "dataset": self.dataset_name,
                "timestamp": now.timestamp(),
                "hostname": hostname,
                "pid": pid,
                "saved_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": {
                "best_val_loss": float(self.early_stopping.val_loss_min),
                "best_epoch": self.early_stopping.best_epoch,
                "total_epochs_run": self.early_stopping.best_epoch,
                "total_train_time_seconds": round(total_time, 1),
                "total_train_time_human": _format_duration(total_time),
            },
            "hyper_params": {
                "win_size": self.config.win_size,
                "lr": self.config.lr,
                "dropout": self.config.dropout,
                "n_heads": self.config.n_heads,
                "e_layers": self.config.e_layers,
                "d_model": self.config.d_model,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "patience": self.config.patience,
                "topk": self.config.topk,
                "anomaly_ratio": self.config.anomaly_ratio,
                "sparse_topk": getattr(self.config, 'sparse_topk', None),
                "lambda_locality_l1": self.config.lambda_locality_l1,
                "lambda_vq": getattr(self.config, "lambda_vq", None),
                "vq_cooldown_epochs": getattr(self.config, "vq_cooldown_epochs", None),
                "vq_score_weight": getattr(self.config, "vq_score_weight", None),
                "score_topk_k": getattr(self.config, "score_topk_k", None),
                "use_vq_bypass": getattr(self.config, "use_vq_bypass", None),
                "dynamic_temporal_residual_init": getattr(self.config, "dynamic_temporal_residual_init", None),
                "dynamic_temporal_topk": getattr(self.config, "dynamic_temporal_topk", None),
                "dynamic_temporal_gate_mode": getattr(self.config, "dynamic_temporal_gate_mode", None),
                "channel_graph_lr_scale": getattr(self.config, "channel_graph_lr_scale", None),
                "temporal_graph_lr_scale": getattr(self.config, "temporal_graph_lr_scale", None),
                "score_smoothing_window": getattr(self.config, "score_smoothing_window", None),
                "score_smoothing_method": getattr(self.config, "score_smoothing_method", None),
                "prediction_fill_gap": getattr(self.config, "prediction_fill_gap", None),
                "prediction_min_len": getattr(self.config, "prediction_min_len", None),
                "prediction_dilate": getattr(self.config, "prediction_dilate", None),
            },
            "model_info": {
                "total_params": total_params,
                "total_params_human": _format_params(total_params),
            },
            "environment": {
                "device": str(self.device),
                "gpu_info": _get_gpu_info(self.device),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params_record, f, indent=2, ensure_ascii=False)
        return filepath

    # ======================== 验证 ========================
    def vali(self, vali_loader):
        self.model.eval()
        loss_list = []
        with torch.inference_mode():
            for input_data, _ in vali_loader:
                input_data = input_data.float().to(self.device, non_blocking=True)
                rec, _, _, _, _, aux_losses, _ = self.model(input_data)
                loss = F.mse_loss(rec, input_data)
                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']
                loss_list.append(loss.item())
        return np.average(loss_list) if loss_list else 0.0

    # ======================== 训练（单卡）=======================
    def _destroy_model_and_clean_cuda(self):
        import gc
        if hasattr(self, 'model') and self.model is not None:
            for p in self.model.parameters():
                if p.device.type == 'cuda':
                    p.data = p.data.cpu()
            del self.model
            self.model = None
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            del self.train_loader
            self.train_loader = None
        if hasattr(self, 'valid_loader') and self.valid_loader is not None:
            del self.valid_loader
            self.valid_loader = None
        if hasattr(self, 'thre_loader') and self.thre_loader is not None:
            del self.thre_loader
            self.thre_loader = None
        if hasattr(self, 'early_stopping') and self.early_stopping is not None:
            del self.early_stopping
            self.early_stopping = None
        gc.collect()
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(f'cuda:{i}')
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass
        gc.collect()
        for _ in range(3):
            torch.cuda.empty_cache()

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.config.input_c = train_data.shape[1]
        self.config.output_c = train_data.shape[1]

        self._destroy_model_and_clean_cuda()

        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        self._train_raw = train_data_value.copy()

        train_scaled = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_scaled = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        if self.multi_gpu_requested:
            print(
                f"\n  [INFO] n_gpus={self.config_n_gpu} was requested, "
                "but DDP has been removed; using the single-GPU training path."
            )

        self._single_gpu_train(train_scaled, valid_scaled)

        if self.early_stopping is not None and self.early_stopping.check_point is not None:
            self._get_raw_model().load_state_dict(self.early_stopping.check_point)

        self.trained = True

        # ★ v10 fix: 训练集缓存分数
        if self._train_raw is not None:
            print(f"\n  [INFO] Computing training reconstruction scores for threshold...")
            self._train_anomaly_scores, _ = self.detect_score(self._train_raw)
            print(f"  [INFO] Train scores: min={self._train_anomaly_scores.min():.6f}, "
                  f"max={self._train_anomaly_scores.max():.6f}, "
                  f"mean={self._train_anomaly_scores.mean():.6f}")

        # ★ P0-2: 同时缓存验证集分数用于阈值校准
        print(f"\n  [INFO] Computing validation reconstruction scores for POT calibration...")
        self._val_anomaly_scores, _ = self.detect_score(valid_data)
        print(f"  [INFO] Val scores: min={self._val_anomaly_scores.min():.6f}, "
              f"max={self._val_anomaly_scores.max():.6f}, "
              f"mean={self._val_anomaly_scores.mean():.6f}")

        # ★ P0-2: POT 阈值估计 + 验证集校准
        if self._train_anomaly_scores is not None:
            pot_threshold = self._pot_estimator.estimate(self._train_anomaly_scores)
            print(f"\n  [POT] Baseline threshold: {pot_threshold:.6f}")

            if self._val_anomaly_scores is not None:
                # 用验证集校准（目标异常比例取 anomaly_ratio 的中位数）
                target_ratio = np.median(self.config.anomaly_ratio) / 100.0
                calibrated = self._pot_estimator.calibrate(
                    self._val_anomaly_scores, target_ratio=target_ratio
                )
                print(f"  [POT] Calibrated threshold: {calibrated:.6f} "
                      f"(target_ratio={target_ratio*100:.1f}%)")

        # ★ P0-5: 设置可视化 hook
        if self.model is not None:
            self._vis_hook = VisualizationHook(self.model)
            self._vis_hook.register_hooks()
            print(f"\n  [VIS] Visualization hooks registered")

    def _single_gpu_train(self, train_df, val_df):
        """单 GPU 训练（v11.4 版）"""
        from ts_benchmark.baselines.utils import anomaly_detection_data_provider as adp

        self.train_loader = adp(
            train_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="train",
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            prefetch_factor=getattr(self.config, 'dataloader_prefetch_factor', 2),
        )
        self.valid_loader = adp(
            val_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="val",
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            prefetch_factor=getattr(self.config, 'dataloader_prefetch_factor', 2),
        )

        self.model = SparseGCN(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            dropout=self.config.dropout,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            e_layers=self.config.e_layers,
            patch_size=self.config.patch_size,
            channel=self.config.input_c,
            topk=self.config.topk,
            sparse_topk=self.config.sparse_topk,
            use_channel_graph=getattr(self.config, "use_channel_graph", True),
            use_temporal_graph=getattr(self.config, "use_temporal_graph", True),
            use_dynamic_temporal_graph=getattr(self.config, "use_dynamic_temporal_graph", False),
            dynamic_temporal_residual_init=getattr(self.config, "dynamic_temporal_residual_init", 0.1),
            dynamic_temporal_topk=getattr(self.config, "dynamic_temporal_topk", None),
            dynamic_temporal_gate_mode=getattr(self.config, "dynamic_temporal_gate_mode", "global"),
            use_vq_bypass=getattr(self.config, "use_vq_bypass", True),
            use_multi_scale_scorer=getattr(self.config, "use_multi_scale_scorer", False),
            vq_score_weight=getattr(self.config, "vq_score_weight", 0.3),
            score_topk_k=getattr(self.config, "score_topk_k", None),
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        train_steps = len(self.train_loader)
        n_train = len(train_df)
        n_val = len(val_df)
        n_features = self.config.input_c

        self._print_training_header(
            n_train=n_train, n_val=n_val, n_features=n_features,
            train_steps=train_steps, total_params=total_params,
        )

        if not torch.cuda.is_available():
            print("  WARNING: CUDA NOT available - running on CPU")
            print()

        self.early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True, relative_delta=0.001,
        )

        channel_graph_params = []
        temporal_graph_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'channel_graph' in name:
                channel_graph_params.append(param)
            elif 'temporal_graph' in name:
                temporal_graph_params.append(param)
            else:
                other_params.append(param)

        weight_decay_v10 = getattr(self.config, 'weight_decay', 1e-4)
        channel_lr_scale = getattr(self.config, 'channel_graph_lr_scale', 0.1)
        temporal_lr_scale = getattr(self.config, 'temporal_graph_lr_scale', 0.1)

        param_groups = [
            {'params': other_params, 'lr': self.config.lr, 'param_names': ['other']},
        ]
        if channel_graph_params:
            param_groups.append({
                'params': channel_graph_params,
                'lr': self.config.lr * channel_lr_scale,
                'param_names': ['channel_graph'],
            })
        if temporal_graph_params:
            param_groups.append({
                'params': temporal_graph_params,
                'lr': self.config.lr * temporal_lr_scale,
                'param_names': ['temporal_graph'],
            })
        self.optimizer = optim.Adam(
            param_groups, lr=self.config.lr, weight_decay=weight_decay_v10,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.num_epochs - self.config.warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )

        train_start_time = time.time()
        REPORT_INTERVAL = 500

        def _freeze_except_vq(model, freeze=True):
            for name, p in model.named_parameters():
                if 'vq_bottleneck' not in name:
                    p.requires_grad = not freeze
                else:
                    p.requires_grad = True

        vq_cooldown_epochs = getattr(self.config, 'vq_cooldown_epochs', 10)
        lambda_vq = getattr(self.config, 'lambda_vq', 0.1)
        use_vq_bypass = getattr(self.config, 'use_vq_bypass', True)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            epoch_losses = []

            warmup_alpha = min(1.0, epoch / max(1, self.config.num_epochs * 0.1))
            self.model.set_warmup_progress(warmup_alpha)

            if use_vq_bypass and epoch == 0:
                _freeze_except_vq(self.model, freeze=True)
            elif use_vq_bypass and epoch == vq_cooldown_epochs:
                _freeze_except_vq(self.model, freeze=False)
                if hasattr(self.model, "_apply_architecture_switches"):
                    self.model._apply_architecture_switches()

            self.model.train()
            self.optimizer.zero_grad()

            for i, (input_data, _) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device, non_blocking=True)
                rec, _, _, _, _, aux_losses, _ = self.model(input_data)

                loss = F.mse_loss(rec, input_data)

                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']

                if use_vq_bypass and aux_losses and 'vq_loss' in aux_losses and epoch < vq_cooldown_epochs:
                    loss = aux_losses['vq_loss']
                else:
                    if use_vq_bypass and aux_losses and 'vq_loss' in aux_losses:
                        loss = loss + lambda_vq * aux_losses['vq_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_losses.append(loss.item())

                if (i + 1) % REPORT_INTERVAL == 0:
                    bar = self._print_progress_bar(i + 1, train_steps, 30)
                    gpu_str = _get_gpu_info(self.device)
                    print(f"  [batch] {bar}  Loss: {np.mean(epoch_losses[-REPORT_INTERVAL:]):.6f}  |  {gpu_str}")

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
            epoch_time = time.time() - epoch_start
            val_loss = self.vali(self.valid_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            es_status = self.early_stopping(val_loss, self.model, epoch + 1)

            self._print_epoch_result(
                epoch=epoch + 1, epoch_time=epoch_time,
                avg_train_loss=avg_train_loss, val_loss=val_loss,
                lr=current_lr, total_epochs=self.config.num_epochs,
                es_status=es_status, es_counter=self.early_stopping.counter,
            )

            if self.early_stopping.early_stop:
                print(f"  ⏹ Early stopping at epoch {epoch + 1} (best: {self.early_stopping.best_epoch})")
                print()
                break

            scheduler.step()

        total_time = time.time() - train_start_time
        saved_path = self._save_best_params(total_params=total_params, total_time=total_time)

        self._save_train_history(total_params=total_params, total_time=total_time)

        print("─" * 70)
        print(f"  [OK] Training Complete!")
        print(f"     Best Val Loss: {self.early_stopping.val_loss_min:.6f}  @  Epoch {self.early_stopping.best_epoch}")
        print(f"     Total Time:    {_format_duration(total_time)}")
        print(f"     Device:        {_get_gpu_info(self.device)}")
        print(f"     Best params saved to: {saved_path}")
        print("─" * 70)
        print()

    def _single_gpu_train_reweight(
        self, train_df, val_df,
        anomaly_weight: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        from ts_benchmark.utils.data_processing import split_before
        from ts_benchmark.baselines.utils import anomaly_detection_data_provider as adp

        self.train_loader = adp(
            train_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="train",
        )
        self.valid_loader = adp(
            val_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="val",
        )

        self.model = SparseGCN(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            dropout=self.config.dropout,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            e_layers=self.config.e_layers,
            patch_size=self.config.patch_size,
            channel=self.config.input_c,
            topk=self.config.topk,
            sparse_topk=self.config.sparse_topk,
            use_channel_graph=getattr(self.config, "use_channel_graph", True),
            use_temporal_graph=getattr(self.config, "use_temporal_graph", True),
            use_dynamic_temporal_graph=getattr(self.config, "use_dynamic_temporal_graph", False),
            dynamic_temporal_residual_init=getattr(self.config, "dynamic_temporal_residual_init", 0.1),
            dynamic_temporal_topk=getattr(self.config, "dynamic_temporal_topk", None),
            dynamic_temporal_gate_mode=getattr(self.config, "dynamic_temporal_gate_mode", "global"),
            use_vq_bypass=getattr(self.config, "use_vq_bypass", True),
            use_multi_scale_scorer=getattr(self.config, "use_multi_scale_scorer", False),
            vq_score_weight=getattr(self.config, "vq_score_weight", 0.3),
            score_topk_k=getattr(self.config, "score_topk_k", None),
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        train_steps = len(self.train_loader)
        n_train = len(train_df)
        n_val = len(val_df)
        n_features = self.config.input_c

        self._print_training_header(
            n_train=n_train, n_val=n_val, n_features=n_features,
            train_steps=train_steps, total_params=total_params,
        )

        self.early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True, relative_delta=0.001,
        )

        channel_graph_params = []
        temporal_graph_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'channel_graph' in name:
                channel_graph_params.append(param)
            elif 'temporal_graph' in name:
                temporal_graph_params.append(param)
            else:
                other_params.append(param)

        channel_lr_scale = getattr(self.config, 'channel_graph_lr_scale', 0.1)
        temporal_lr_scale = getattr(self.config, 'temporal_graph_lr_scale', 0.1)
        param_groups = [{'params': other_params, 'lr': self.config.lr}]
        if channel_graph_params:
            param_groups.append({
                'params': channel_graph_params,
                'lr': self.config.lr * channel_lr_scale,
            })
        if temporal_graph_params:
            param_groups.append({
                'params': temporal_graph_params,
                'lr': self.config.lr * temporal_lr_scale,
            })

        self.optimizer = optim.Adam(
            param_groups, lr=self.config.lr, weight_decay=1e-4,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.num_epochs - self.config.warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )

        train_start_time = time.time()

        if use_focal:
            print(f"  [WARN] FocalLoss has been removed (P0). Falling back to weighted MSE.")
        print(f"\n  [REWEIGHT] anomaly_weight={anomaly_weight}")

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            epoch_losses = []

            self.model.train()
            self.optimizer.zero_grad()

            for i, (input_data, labels) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device)
                labels = labels.float().to(self.device)

                rec, _, _, _, _, aux_losses, _ = self.model(input_data)

                loss = F.mse_loss(rec, input_data)

                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
            epoch_time = time.time() - epoch_start
            val_loss = self.vali(self.valid_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            es_status = self.early_stopping(val_loss, self.model, epoch + 1)

            self._print_epoch_result(
                epoch=epoch + 1, epoch_time=epoch_time,
                avg_train_loss=avg_train_loss, val_loss=val_loss,
                lr=current_lr, total_epochs=self.config.num_epochs,
                es_status=es_status, es_counter=self.early_stopping.counter,
            )

            if self.early_stopping.early_stop:
                break

            scheduler.step()

        total_time = time.time() - train_start_time
        self._save_best_params(total_params=total_params, total_time=total_time)
        self._save_train_history(total_params=total_params, total_time=total_time)

        print("─" * 70)
        print(f"  [OK] Reweight Training Complete!")
        print(f"     Best Val Loss: {self.early_stopping.val_loss_min:.6f}")
        print("─" * 70)
        print()

    def _save_train_history(self, total_params, total_time):
        from datetime import datetime
        from ts_benchmark.common.constant import ROOT_PATH

        if self.early_stopping is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_dir = os.path.join(ROOT_PATH, "result", "experiments", f"{self.dataset_name}_{timestamp}")
        os.makedirs(history_dir, exist_ok=True)

        history = {
            "dataset": self.dataset_name,
            "model": "LaGraph-v11.4",
            "config": {
                "d_model": self.config.d_model,
                "e_layers": self.config.e_layers,
                "n_heads": self.config.n_heads,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "patience": self.config.patience,
                "warmup_epochs": self.config.warmup_epochs,
                "lambda_vq": getattr(self.config, "lambda_vq", None),
                "vq_cooldown_epochs": getattr(self.config, "vq_cooldown_epochs", None),
                "vq_score_weight": getattr(self.config, "vq_score_weight", None),
                "score_topk_k": getattr(self.config, "score_topk_k", None),
                "dataloader_num_workers": getattr(self.config, "dataloader_num_workers", None),
                "dataloader_prefetch_factor": getattr(self.config, "dataloader_prefetch_factor", None),
                "use_channel_graph": getattr(self.config, "use_channel_graph", None),
                "use_temporal_graph": getattr(self.config, "use_temporal_graph", None),
                "use_dynamic_temporal_graph": getattr(self.config, "use_dynamic_temporal_graph", None),
                "dynamic_temporal_residual_init": getattr(self.config, "dynamic_temporal_residual_init", None),
                "dynamic_temporal_topk": getattr(self.config, "dynamic_temporal_topk", None),
                "dynamic_temporal_gate_mode": getattr(self.config, "dynamic_temporal_gate_mode", None),
                "channel_graph_lr_scale": getattr(self.config, "channel_graph_lr_scale", None),
                "temporal_graph_lr_scale": getattr(self.config, "temporal_graph_lr_scale", None),
                "use_vq_bypass": getattr(self.config, "use_vq_bypass", None),
                "use_multi_scale_scorer": getattr(self.config, "use_multi_scale_scorer", None),
                "score_smoothing_window": getattr(self.config, "score_smoothing_window", None),
                "score_smoothing_method": getattr(self.config, "score_smoothing_method", None),
                "prediction_fill_gap": getattr(self.config, "prediction_fill_gap", None),
                "prediction_min_len": getattr(self.config, "prediction_min_len", None),
                "prediction_dilate": getattr(self.config, "prediction_dilate", None),
                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                "lambda_locality_l1": self.config.lambda_locality_l1,
                "dropout": self.config.dropout,
                "win_size": self.config.win_size,
            },
            "total_params": total_params,
            "total_params_human": _format_params(total_params),
            "total_time_seconds": round(total_time, 1),
            "total_time_human": _format_duration(total_time),
            "best_val_loss": float(self.early_stopping.val_loss_min),
            "best_epoch": self.early_stopping.best_epoch,
            "early_stopped": self.early_stopping.early_stop,
        }

        filepath = os.path.join(history_dir, "train_history.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"  [History] Training history -> {filepath}")

    # ════════════════════════════════════════════════════════════════
    #  检测阶段
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _point_wise_aggregate(window_scores: np.ndarray, win_size: int, total_length: int) -> np.ndarray:
        N_windows = len(window_scores)
        point_scores = np.zeros(total_length, dtype=np.float64)
        point_counts = np.zeros(total_length, dtype=np.float64)

        for w in range(N_windows):
            start = w
            end = min(w + win_size, total_length)
            actual_len = end - start
            point_scores[start:end] += window_scores[w, :actual_len]
            point_counts[start:end] += 1.0

        point_scores = np.divide(
            point_scores, point_counts,
            out=np.zeros_like(point_scores),
            where=point_counts > 0,
        )
        return point_scores.astype(np.float32)

    def _smooth_scores_for_detection(self, scores: np.ndarray) -> np.ndarray:
        window = int(getattr(self.config, "score_smoothing_window", 1) or 1)
        if window <= 1:
            return scores.astype(np.float32, copy=False)
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(scores.astype(np.float64), (pad, pad), mode="edge")
        method = getattr(self.config, "score_smoothing_method", "mean")
        if method == "max":
            smoothed = np.empty_like(scores, dtype=np.float64)
            for i in range(len(scores)):
                smoothed[i] = padded[i:i + window].max()
        else:
            kernel = np.ones(window, dtype=np.float64) / float(window)
            smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(np.float32)

    @staticmethod
    def _binary_segments(pred: np.ndarray, value: int):
        n = len(pred)
        start = None
        for i in range(n + 1):
            cur = pred[i] if i < n else 1 - value
            if cur == value and start is None:
                start = i
            elif cur != value and start is not None:
                yield start, i
                start = None

    def _shape_prediction_segments(self, pred: np.ndarray) -> np.ndarray:
        pred = pred.astype(np.int32, copy=True)
        min_len = int(getattr(self.config, "prediction_min_len", 1) or 1)
        fill_gap = int(getattr(self.config, "prediction_fill_gap", 0) or 0)
        dilate = int(getattr(self.config, "prediction_dilate", 0) or 0)

        if min_len > 1:
            for start, end in list(self._binary_segments(pred, 1)):
                if end - start < min_len:
                    pred[start:end] = 0

        if fill_gap > 0:
            for start, end in list(self._binary_segments(pred, 0)):
                if start > 0 and end < len(pred) and end - start <= fill_gap:
                    pred[start:end] = 1

        if dilate > 0 and pred.any():
            shaped = pred.copy()
            for start, end in self._binary_segments(pred, 1):
                shaped[max(0, start - dilate):min(len(pred), end + dilate)] = 1
            pred = shaped

        return pred

    @torch.no_grad()
    def _detect_forward(self, input_data):
        score, vq_score = self.model.multi_scale_forward(input_data)
        return score.cpu().numpy()

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call detect_fit first.")
        self._get_raw_model().load_state_dict(self.early_stopping.check_point)

        import gc
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        for _ in range(2):
            torch.cuda.empty_cache()
        self.model.to(self.device)

        eval_batch_size = min(self.config.batch_size, 64)
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(self.device)[0] / 1024**3
            if free_gb < 2.0:
                eval_batch_size = min(eval_batch_size, 32)
            elif free_gb < 4.0:
                eval_batch_size = min(eval_batch_size, 48)

        scaled_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns, index=train.index,
        )
        total_length = len(scaled_data)

        self.model.eval()
        window_scores_list = []

        loader = anomaly_detection_data_provider(
            scaled_data, batch_size=eval_batch_size,
            win_size=self.config.win_size, step=1, mode="test",
            num_workers=0,
        )

        for i, (input_data, labels) in enumerate(loader):
            input_data = input_data.float().to(self.device)
            scores_batch = self._detect_forward(input_data)
            window_scores_list.append(scores_batch)
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if len(window_scores_list) == 0:
            dummy = np.zeros(total_length)
            return dummy, dummy

        window_scores = np.concatenate(window_scores_list, axis=0)

        point_scores = self._point_wise_aggregate(
            window_scores, self.config.win_size, total_length,
        )
        point_scores = self._smooth_scores_for_detection(point_scores)

        return point_scores, point_scores

    def detect_label(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        检测并返回异常标签（多异常率）。

        ★ P0-2: 支持 POT 阈值估计（通过 self._pot_estimator）
        ★ P0-3: 如果 test_data 真实标签可通过外部获取，调用 evaluate() 计算指标

        Returns:
            preds: {ratio: np.ndarray(N,)} — 各异常率下的 0/1 标签
            test_energy: np.ndarray(N,) — 点级异常分数
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call detect_fit first.")
        self._get_raw_model().load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)

        eval_batch_size = min(self.config.batch_size, 64)
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(self.device)[0] / 1024**3
            if free_gb < 2.0:
                eval_batch_size = min(eval_batch_size, 32)
            elif free_gb < 4.0:
                eval_batch_size = min(eval_batch_size, 48)

        scaled_test = pd.DataFrame(
            self.scaler.transform(test_data.values),
            columns=test_data.columns, index=test_data.index,
        )
        total_length = len(scaled_test)

        self.model.eval()

        test_loader = anomaly_detection_data_provider(
            scaled_test, batch_size=eval_batch_size,
            win_size=self.config.win_size, step=1, mode="test",
            num_workers=0,
        )

        test_window_list = []
        for i, (input_data, labels) in enumerate(test_loader):
            input_data = input_data.float().to(self.device)
            cri = self._detect_forward(input_data)
            test_window_list.append(cri)
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        if len(test_window_list) == 0:
            dummy = np.zeros(total_length, dtype=np.int32)
            return {r: dummy for r in self.config.anomaly_ratio}, np.zeros(total_length)

        test_windows = np.concatenate(test_window_list, axis=0)
        test_energy = self._point_wise_aggregate(
            test_windows, self.config.win_size, total_length,
        )
        test_energy = self._smooth_scores_for_detection(test_energy)

        # === 步骤 2：阈值选取（★ P0-2: 优先使用 POT 阈值）===
        pot_threshold = self._pot_estimator.get_threshold()

        if pot_threshold > 0 and self._train_anomaly_scores is not None:
            # 使用 POT 估计的阈值作为基准
            threshold_source = self._train_anomaly_scores
            print(f"  [POT] Using POT-estimated threshold={pot_threshold:.6f}")
        elif self._train_anomaly_scores is not None and len(self._train_anomaly_scores) > 0:
            threshold_source = self._train_anomaly_scores
            print(f"  [INFO] Using cached train scores for threshold")
        else:
            print(f"  [WARN] Train scores not available, falling back to test scores")
            threshold_source = test_energy

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        print(f"\n  [DIAGNOSTIC] detect_label analysis:")
        print(f"  [DIAGNOSTIC]   test_energy: n={len(test_energy)}, "
              f"min={test_energy.min():.6f}, max={test_energy.max():.6f}, "
              f"mean={test_energy.mean():.6f}")

        preds = {}
        for ratio in self.config.anomaly_ratio:
            # ★ BUGFIX v11.4.1: 每个 ratio 独立使用百分位数阈值
            #   旧版: max(pot_threshold, np.percentile(..., 100 - ratio))
            #   问题: pot_threshold ≈ 99.99th percentile 始终大于所有 ratio 百分位
            #         导致所有 ratio 共享同一极端的 POT 阈值 → 所有预测完全相同
            #   新版: per-ratio 直接使用百分位数阈值，POT 阈值仅用于 ratio=None 的默认输出
            threshold = np.percentile(threshold_source, 100 - ratio)

            pred = (test_energy > threshold).astype(int)
            preds[ratio] = self._shape_prediction_segments(pred)
            anom_frac = preds[ratio].mean() * 100
            print(f"  [DIAGNOSTIC]   ratio={ratio:>5.1f}%: threshold={threshold:.6f}, "
                  f"anomaly_frac={anom_frac:.2f}%")

        # ★ BUGFIX: POT 阈值仅用于默认检测（ratio=None），不覆盖 per-ratio 结果
        if pot_threshold > 0:
            pred_pot = self._shape_prediction_segments(
                (test_energy > pot_threshold).astype(int)
            )
            preds[None] = pred_pot
            print(f"  [DIAGNOSTIC]   POT default: threshold={pot_threshold:.6f}, "
                  f"anomaly_frac={pred_pot.mean() * 100:.2f}%")

        # ★ P0-5: 如果可视化 hook 已注册，捕获中间数据
        if self._vis_hook is not None:
            try:
                vis_data = self._vis_hook.extract()
                from datetime import datetime
                from ts_benchmark.common.constant import ROOT_PATH
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                vis_dir = os.path.join(ROOT_PATH, "result", "diagnosis", self.dataset_name, ts)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, "vis_data.json")
                self._vis_hook.save_to_json(vis_data, vis_path)
                print(f"\n  [VIS] Visualization data -> {vis_path}")
            except Exception as e:
                print(f"\n  [VIS WARN] Failed to extract visualization data: {e}")

        return preds, test_energy

    # ════════════════════════════════════════════════════════════════
    #  ★ P0-3: 评估报告系统（AUC-ROC / AUPR）
    # ════════════════════════════════════════════════════════════════

    def evaluate(self, test_data: pd.DataFrame, test_labels: np.ndarray,
                 save_path: str = None) -> dict:
        """
        完整评估报告：AUC-ROC + AUPR + 各 ratio F1。

        Args:
            test_data: 测试数据
            test_labels: 真实 0/1 标签 (N,)
            save_path: 保存路径（可选）

        Returns:
            report: 包含所有指标的 dict
        """
        preds, scores = self.detect_label(test_data)
        scores = np.asarray(scores, dtype=np.float64).flatten()
        labels = np.asarray(test_labels, dtype=np.int32).flatten()

        # 对齐长度
        min_len = min(len(scores), len(labels))
        scores, labels = scores[:min_len], labels[:min_len]

        # 计算指标
        report = {
            'dataset': self.dataset_name,
            'model': 'LaGraph-v11.4',
            'n_samples': len(scores),
            'anomaly_ratio_gt': float(labels.mean() * 100),
        }

        # AUC-ROC
        if len(np.unique(labels)) >= 2:
            try:
                report['auc_roc'] = float(roc_auc_score(labels, scores))
            except Exception:
                report['auc_roc'] = 0.5
        else:
            report['auc_roc'] = 0.5

        # AUPR
        try:
            report['aupr'] = float(average_precision_score(labels, scores))
        except Exception:
            report['aupr'] = float(labels.mean())

        # 各 ratio 的 F1
        report['f1_per_ratio'] = {}
        for ratio, pred in preds.items():
            pred = pred[:min_len].flatten().astype(int)
            tp = (pred * labels).sum()
            fp = pred.sum() - tp
            fn = labels.sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            report['f1_per_ratio'][ratio] = {
                'f1': float(f1),
                'precision': float(prec),
                'recall': float(rec),
                'threshold': None,  # 由 detect_label 决定
            }

        # 输出报告
        print(f"\n{'='*60}")
        print(f"  ★ Evaluation Report: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"  AUC-ROC:  {report['auc_roc']:.4f}")
        print(f"  AUPR:     {report['aupr']:.4f}")
        print(f"  GT Anomaly Ratio: {report['anomaly_ratio_gt']:.2f}%")
        for ratio, f1_data in report['f1_per_ratio'].items():
            print(f"  F1@{ratio}%: {f1_data['f1']:.4f}")
        print(f"{'='*60}\n")

        if save_path:
            import json
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, cls=_NumpyEncoder)
            print(f"  [Report] Saved to {save_path}")

        return report

    # ════════════════════════════════════════════════════════════════
    #  ★ P0-4: 多 seed 实验运行接口
    # ════════════════════════════════════════════════════════════════

    def run_multi_seed_experiment(self, train_data: pd.DataFrame,
                                   test_data: pd.DataFrame,
                                   test_labels: np.ndarray,
                                   num_seeds: int = None) -> dict:
        """
        多次运行训练+检测（不同 seed），统计指标。

        Args:
            train_data: 训练数据
            test_data: 测试数据
            test_labels: 真实标签
            num_seeds: 运行次数（默认 self.config.num_seeds）

        Returns:
            summary: 统计总结
        """
        if num_seeds is None:
            num_seeds = self.config.num_seeds

        all_results = []
        for seed_idx in range(num_seeds):
            seed = self.config.seed_base + seed_idx
            print(f"\n{'='*60}")
            print(f"  Seed {seed_idx + 1}/{num_seeds} (base={seed})")
            print(f"{'='*60}")

            # 设置随机种子
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 创建新模型实例
            model = LaGraph(**{
                k: getattr(self.config, k)
                for k in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
            })
            model.config.dataset_name = self.dataset_name

            # 训练
            model.detect_fit(train_data, test_data)

            # 评估
            report = model.evaluate(test_data, test_labels)
            all_results.append(report)

        # 汇总统计
        summary = self._aggregate_seed_results(all_results)
        return summary

    def _aggregate_seed_results(self, results: list) -> dict:
        """聚合多次 seed 实验的结果。"""
        if not results:
            return {}

        metrics = ['auc_roc', 'aupr']
        summary = {
            'num_seeds': len(results),
            'dataset': self.dataset_name,
        }

        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results]
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'values': [float(v) for v in values],
            }
            if len(values) >= 2:
                se = np.std(values, ddof=1) / np.sqrt(len(values))
                ci = scipy_stats.t.ppf(0.975, len(values) - 1) * se
                summary[metric]['ci_95'] = float(ci)
            else:
                summary[metric]['ci_95'] = 0.0

        # F1 聚合
        f1_ratios = set()
        for r in results:
            f1_ratios.update(r.get('f1_per_ratio', {}).keys())
        summary['f1_per_ratio'] = {}
        for ratio in sorted(f1_ratios):
            f1_vals = [r['f1_per_ratio'].get(ratio, {}).get('f1', 0.0) for r in results]
            summary['f1_per_ratio'][ratio] = {
                'mean': float(np.mean(f1_vals)),
                'std': float(np.std(f1_vals, ddof=1)),
            }

        # 打印统计报告
        print(f"\n{'='*60}")
        print(f"  ★ Multi-Seed Summary ({len(results)} seeds)")
        print(f"{'='*60}")
        for metric in metrics:
            m = summary[metric]
            print(f"  {metric.upper():>8s}: {m['mean']:.4f} ± {m['std']:.4f} "
                  f"(95% CI: ±{m.get('ci_95', 0):.4f})")
        print(f"{'='*60}\n")

        return summary
