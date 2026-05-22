# -*- coding: utf-8 -*-
"""
LaGraph v10 DDP 训练 worker（torchrun 启动）
=============================================
用法：torchrun --nproc_per_node=N distributed_worker_v10.py --config <json_path>

基于 v10 SparseGCN 的 DDP 多卡训练。
与 v9 版的关键区别：
  1. 使用 SparseGCN 而非旧版 GCN_model
  2. 仅包含 MSE + L1 sparse 损失（无 freq/contrastive/prototype/prediction）
  3. 无需渐进式损失激活
  4. 精简验证和 checkpoint 逻辑
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ========== GPU 利用率优化 ==========
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# 确保包路径正确
_worker_file = os.path.abspath(__file__)
_project_root = _worker_file
for _ in range(10):
    _parent = os.path.dirname(_project_root)
    if os.path.exists(os.path.join(_parent, "ts_benchmark")):
        _project_root = _parent
        break
    _project_root = _parent
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import SparseGCN
from ts_benchmark.baselines.utils import SegLoader


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


class EarlyStopping:
    """
    使用**相对阈值**的早停机制，按 loss 量级自适应。
    relative_delta=0.001 表示需要改善 ≥0.1% 才算 improvement。
    """
    def __init__(self, patience=7, verbose=False, delta=0, relative_delta=0.01, min_delta=1e-5):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_val_loss = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.relative_delta = relative_delta  # ★ 相对阈值（默认1%）
        self.min_delta = min_delta            # ★ 绝对阈值兜底
        self.best_epoch = 0
        self.check_point = None

    def __call__(self, val_loss, model, epoch):
        if self.best_score is None:
            self.best_score = -val_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            return "initial"

        abs_improvement = self.best_val_loss - val_loss
        rel_improvement = abs_improvement / max(self.best_val_loss, 1e-10)

        if rel_improvement >= self.relative_delta and abs_improvement >= self.min_delta:
            # ★ 真正改善超过 relative_delta 倍
            self.best_score = -val_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch
            return "improved"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return "no_improve"

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss
        state_dict = model.state_dict()
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.check_point = {k: v.cpu().clone() for k, v in state_dict.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--data', type=str, required=True, help='Path to data pickle')
    parser.add_argument('--checkpoint_out', type=str, required=True, help='Path to save best checkpoint')
    parser.add_argument('--result_out', type=str, required=True, help='Path to save result JSON')
    parser.add_argument('--history_out', type=str, default=None, help='Path to save epoch history JSON')
    args = parser.parse_args()

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = rank

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend='nccl')

    with open(args.config, 'r') as f:
        config = argparse.Namespace()
        for k, v in json.load(f).items():
            setattr(config, k, v)

    with open(args.data, 'rb') as f:
        data_dict = pickle.load(f)

    train_np = data_dict['train_np']
    val_np = data_dict['val_np']

    train_df = pd.DataFrame(train_np)
    val_df = pd.DataFrame(val_np)

    train_dataset = SegLoader(train_df.values, config.win_size, step=1, mode="train")
    val_dataset = SegLoader(val_df.values, config.win_size, step=1, mode="val")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
    )

    # ★ GPU 利用率优化：SegLoader 使用 numpy 索引（无 CUDA 张量），
    #   设置 num_workers=2 安全，可大幅减少滑动窗口生成的 CPU 开销。
    #   每个 GPU 独立使用 2 个 worker，配合 prefetch_factor=2 可 pipeline 数据加载。
    #   effective_batch=2048×8=16384，每个 batch 计算 ~0.5ms，数据加载若不并行
    #   则是关键瓶颈。
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=True,
        prefetch_factor=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=2, pin_memory=True, drop_last=False,
        prefetch_factor=2, persistent_workers=True,
    )

    # ★ v10: 使用 SparseGCN
    model = SparseGCN(
        win_size=config.win_size,
        enc_in=config.input_c,
        c_out=config.output_c,
        dropout=config.dropout,
        n_heads=config.n_heads,
        d_model=config.d_model,
        e_layers=config.e_layers,
        patch_size=config.patch_size,
        channel=config.input_c,
        topk=config.topk,
        sparse_topk=getattr(config, 'sparse_topk', None),
    )
    model.to(device)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # ★ v11: VQ cooldown 期间冻结非 VQ 参数，需要 find_unused_parameters=True
    #   否则 DDP reducer 会因部分参数未收到梯度而崩溃
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True, broadcast_buffers=True,
                gradient_as_bucket_view=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lambda_locality_l1 = getattr(config, 'lambda_locality_l1', 0.001)
    warmup_epochs = getattr(config, 'warmup_epochs', 5)
    num_epochs = config.num_epochs

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  LaGraph v10 DDP Training  |  {world_size} GPUs  |  {_format_params(total_params)} params")
        print(f"  Batch: {config.batch_size}/GPU = {config.batch_size * world_size} eff.")
        print(f"  Steps/GPU/epoch: {len(train_loader)}")
        print(f"  d_model={config.d_model}, e_layers={config.e_layers}, n_heads={config.n_heads}")
        print(f"  lambda_locality_l1={lambda_locality_l1}, warmup={warmup_epochs}")
        print(f"{'='*60}\n", flush=True)

    # 差分学习率：图结构参数使用较低 LR
    lr = config.lr
    graph_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'channel_graph' in name or 'temporal_graph' in name:
            graph_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {'params': other_params, 'lr': lr, 'param_names': ['other']},
        {'params': graph_params, 'lr': lr * 0.1, 'param_names': ['channel_graph', 'temporal_graph']},
    ], lr=lr, weight_decay=1e-4)  # 1e-5 → 1e-4：增强 L2 正则化

    # Warmup + Cosine Annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_epochs),
        eta_min=1e-6,  # 1e-7 → 1e-6：防止 LR 降到极低时被困在局部极小值
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # 验证函数
    def validate(val_loader_local):
        model.eval()
        loss_list = []
        with torch.inference_mode():
            for input_data, _ in val_loader_local:
                input_data = input_data.float().to(device, non_blocking=True)
                rec, _, _, _, _, aux_losses, _ = model(input_data)
                loss = F.mse_loss(rec, input_data)
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + lambda_locality_l1 * aux_losses['sparse_loss']
                loss_list.append(loss.item())
        return np.average(loss_list) if loss_list else 0.0

    early_stopping = EarlyStopping(patience=getattr(config, 'patience', 7), verbose=(rank == 0))

    train_start = time.time()
    epoch_records = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_sampler.set_epoch(epoch)

        # ★ v11: VQ cooldown — 前 vq_cooldown_epochs 轮冻结非 VQ 参数
        #   只在 rank 0 控制冻结（DDP 会自动同步 requires_grad）
        vq_cooldown_epochs = getattr(config, 'vq_cooldown_epochs', 10)
        lambda_vq = getattr(config, 'lambda_vq', 0.01)

        def _freeze_except_vq(model_obj, freeze=True):
            for name, p in model_obj.named_parameters():
                if 'vq_bottleneck' not in name:
                    p.requires_grad = not freeze
                else:
                    p.requires_grad = True
            if freeze and rank == 0:
                print(f"  [VQ] Cooldown active: freezing non-VQ params")
            elif not freeze and rank == 0:
                print(f"  [VQ] Cooldown ended: all params trainable")

        if epoch == 0:
            _freeze_except_vq(model.module if hasattr(model, 'module') else model, freeze=True)
        elif epoch == vq_cooldown_epochs:
            _freeze_except_vq(model.module if hasattr(model, 'module') else model, freeze=False)

        # Warmup 进度
        warmup_alpha = min(1.0, epoch / max(1, num_epochs * 0.1))
        raw_model = model.module if hasattr(model, 'module') else model
        if hasattr(raw_model, 'set_warmup_progress'):
            raw_model.set_warmup_progress(warmup_alpha)

        epoch_losses = []
        optimizer.zero_grad()

        for i, (input_data, _) in enumerate(train_loader):
            input_data = input_data.float().to(device, non_blocking=True)
            rec, _, _, _, _, aux_losses, _ = model(input_data)

            # ★ v11 FIX: 冷却期内仅使用 VQ loss（其他参数冻结，MSE 无梯度流动）
            if aux_losses and 'vq_loss' in aux_losses and epoch < vq_cooldown_epochs:
                loss = aux_losses['vq_loss']
            else:
                loss = F.mse_loss(rec, input_data)
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + lambda_locality_l1 * aux_losses['sparse_loss']
                # ★ v11: VQ commitment loss（冷却期后才启用）
                if aux_losses and 'vq_loss' in aux_losses:
                    loss = loss + lambda_vq * aux_losses['vq_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0

        # 验证
        local_val_loss = validate(val_loader)
        val_tensor = torch.tensor([local_val_loss], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_tensor.item()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            es_status = early_stopping(val_loss, model, epoch + 1)
            best_v = early_stopping.val_loss_min
            remaining = (num_epochs - (epoch + 1)) * epoch_time
            eta_str = _format_duration(remaining) if remaining > 0 else "---"

            if es_status == "improved":
                val_mark = "* BEST"
            elif es_status == "initial":
                val_mark = "* INIT"
            else:
                val_mark = f"v {early_stopping.counter}/{early_stopping.patience}"

            print(f"  Epoch {epoch+1:>3}/{num_epochs}  "
                  f"Train: {avg_train_loss:.6f}  "
                  f"Val: {val_loss:.6f} {val_mark}  "
                  f"Best: {best_v:.6f}  "
                  f"LR: {current_lr:.2e}  "
                  f"Time: {_format_duration(epoch_time)}  "
                  f"ETA: {eta_str}",
                  flush=True)

            epoch_records.append({
                "epoch": epoch + 1,
                "train_loss": round(float(avg_train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "best_val_loss": round(float(best_v), 6),
                "lr": float(current_lr),
                "status": es_status,
                "epoch_time_seconds": round(epoch_time, 2),
            })

        # 广播 early stop 信号
        stop_flag = torch.tensor([1.0 if (rank == 0 and early_stopping.early_stop) else 0.0], device=device)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() > 0.5:
            if rank == 0:
                print(f"  ⏹ Early stopping at epoch {epoch+1} (best: {early_stopping.best_epoch})", flush=True)
            break

        scheduler.step()

    total_time = time.time() - train_start

    if rank == 0:
        checkpoint = early_stopping.check_point
        torch.save(checkpoint, args.checkpoint_out)

        result = {
        'best_val_loss': float(early_stopping.val_loss_min),
        'best_epoch': early_stopping.best_epoch,
        'total_time': total_time,
        'total_params': total_params,
        'world_size': world_size,
        'config': {
            'd_model': config.d_model,
            'e_layers': config.e_layers,
            'n_heads': getattr(config, 'n_heads', None),
            'batch_size': config.batch_size,
            'lr': config.lr,
            'lambda_locality_l1': lambda_locality_l1,
            'warmup_epochs': warmup_epochs,
        },
        }
        with open(args.result_out, 'w') as f:
            json.dump(result, f)

        if args.history_out:
            history = {
                "dataset": getattr(config, 'dataset_name', 'Unknown'),
                "epochs": epoch_records,
                "best_epoch": early_stopping.best_epoch,
                "best_val_loss": float(early_stopping.val_loss_min),
                "total_time_seconds": total_time,
                "total_time_human": _format_duration(total_time),
            }
            with open(args.history_out, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"\n  ✓ DDP Training Complete! ({world_size} GPUs, {_format_duration(total_time)})")
        print(f"    Best Val Loss: {early_stopping.val_loss_min:.6f} @ Epoch {early_stopping.best_epoch}")
        print(f"    Checkpoint -> {args.checkpoint_out}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
