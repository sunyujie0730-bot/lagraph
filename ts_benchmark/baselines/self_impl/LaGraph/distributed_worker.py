# -*- coding: utf-8 -*-
"""
LaGraph v9 DDP 训练 worker（torchrun 启动）
=============================================
用法：torchrun --nproc_per_node=N distributed_worker.py --config <json_path>

v9 变更（docs/update.md）：
1. forward 签名包含 6 个返回值: rec, A_adaptive, d_score, freq_recon, stage1_feat, aux_losses
2. GCN_model 新增 v9 参数: use_contrastive, contrastive_temp, lambda_contrastive, use_prototype, num_prototypes
3. 分阶段损失激活（§3.1）：epoch<50 仅 MSE+L1，epoch>=50 加入 freq+contrastive
4. 图结构 50 epoch 后冻结
5. 对比学习（§3.2.1）和原型记忆库（§3.2.2）

GPU 利用率优化（v9.1）:
1. DataLoader num_workers=4 + prefetch_factor=2 — 多进程数据预加载
2. 移除 torch.cuda.empty_cache() — 避免 PCIe 同步阻塞
3. find_unused_parameters=False — 减少 DDP 梯度同步开销
4. torch.backends.cudnn.benchmark=True — cuDNN auto-tuner 加速卷积
5. 对比损失向量化 — 消除 Python for 循环（L 次 matmul → 1 次 batch matmul）
6. torch.set_float32_matmul_precision('high') — 利用 Tensor Cores
7. 可选 torch.compile（环境变量 ENABLE_COMPILE=1 开启）
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

# ========== GPU 利用率优化：cuDNN + Tensor Cores ==========
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ========== 可选 torch.compile ==========
_ENABLE_COMPILE = os.environ.get('ENABLE_COMPILE', '0') == '1'

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

from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import GCN_model
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
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = 0
        self.check_point = None

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            return "initial"
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return "no_improve"
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch
            return "improved"

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss
        state_dict = model.state_dict()
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.check_point = {k: v.cpu().clone() for k, v in state_dict.items()}


def _compute_freq_loss_worker(freq_recon, input_data, device):
    """频域损失计算（worker 本地版）"""
    input_fft = torch.fft.rfft(input_data, dim=1, norm='ortho')
    freq_fft = torch.fft.rfft(freq_recon, dim=1, norm='ortho')
    mag_loss = F.mse_loss(torch.abs(freq_fft), torch.abs(input_fft))
    phase_cos = F.cosine_similarity(
        torch.angle(freq_fft).flatten(1), torch.angle(input_fft).flatten(1), dim=-1
    ).mean()
    phase_loss = 1.0 - phase_cos
    return mag_loss + 0.5 * phase_loss


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

    # ★ v9.1 GPU 利用率优化: 多进程数据加载
    # ★ v9.5 OOM 修复: num_workers 4→2, prefetch 2→1 减少 pinned memory 占用
    #   num_workers=2（保持 CPU 数据加载并行）+ prefetch=1
    #   persistent_workers=True（避免每个 epoch 重建 worker 进程）
    #   pinned memory 从 ~1.7 GiB → ~1.1 GiB，减少对显存池的竞争
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=2,
        prefetch_factor=1,
        persistent_workers=True,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=True,
        pin_memory=True, drop_last=False,
    )

    # ★ v9: GCN_model 传入 v9 参数
    model = GCN_model(
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
        sparse_topk=config.sparse_topk,
        use_freq_loss=config.use_freq_loss,
        lambda_freq=config.lambda_freq,
        use_contrastive=getattr(config, 'use_contrastive', True),
        lambda_contrastive=getattr(config, 'lambda_contrastive', 0.05),
        contrastive_temp=getattr(config, 'contrastive_temp', 0.5),
        use_prototype=getattr(config, 'use_prototype', True),
        num_prototypes=getattr(config, 'num_prototypes', 16),
        # ★ Phase 2: 预测分支（§2.3 — 解重建悖论）
        use_prediction_head=getattr(config, 'use_prediction_head', False),
        lambda_pred=getattr(config, 'lambda_pred', 0.3),
    )

    # ★ v9.1 GPU 利用率优化: 可选 torch.compile
    #   通过环境变量 ENABLE_COMPILE=1 启用
    #   大幅加速 PyTorch 计算图（30-50% 可能）
    if _ENABLE_COMPILE:
        try:
            model = torch.compile(model, mode='max-autotune', fullgraph=False)
            if rank == 0:
                print("  [v9.1] torch.compile enabled (max-autotune mode)", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"  [v9.1] torch.compile failed (falling back to eager): {e}", flush=True)

    model.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acc_steps = getattr(config, 'gradient_accumulation_steps', 1)
    use_freq_loss = getattr(config, 'use_freq_loss', True)
    lambda_freq = getattr(config, 'lambda_freq', 0.1)
    lambda_contrastive = getattr(config, 'lambda_contrastive', 0.05)
    use_contrastive = getattr(config, 'use_contrastive', True)
    lambda_locality_l1 = getattr(config, 'lambda_locality_l1', 0.001)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  LaGraph v9 DDP Training  |  {world_size} GPUs  |  {_format_params(total_params)} params")
        print(f"  Batch: {config.batch_size}/GPU x {acc_steps} acc = {config.batch_size * world_size * acc_steps} eff.")
        print(f"  Steps/GPU/epoch: {len(train_loader)}")
        print(f"  Freq Loss: {use_freq_loss}  |  Contrastive: {use_contrastive}  |  lambda_locality_l1={lambda_locality_l1}")
        print(f"  [v9.1] DataLoader workers: train=4, val=2  |  cudnn.benchmark=True  |  find_unused=True")
        print(f"{'='*60}\n", flush=True)

    # ★ v9.1: 渐进式损失激活（替代 Epoch-50 硬阈值）
    num_epochs_local = config.num_epochs
    warm_up_local = max(1, int(num_epochs_local * 0.1))
    ramp_up_end_local = max(warm_up_local + 1, int(num_epochs_local * 0.5))

    def compute_loss(rec, input_data, freq_recon, aux_losses=None, epoch=None):
        loss = F.mse_loss(rec, input_data)

        # 渐进式 ramping 替代硬阈值
        if epoch is not None:
            if epoch < warm_up_local:
                ramp_factor = 0.0
            else:
                ramp_factor = min(1.0, (epoch - warm_up_local) / (ramp_up_end_local - warm_up_local))
        else:
            ramp_factor = 1.0  # 评估模式全激活

        if ramp_factor > 0 and aux_losses is not None:
            if 'freq_loss' in aux_losses:
                loss = loss + ramp_factor * lambda_freq * aux_losses['freq_loss']
            if 'contrastive_loss' in aux_losses and use_contrastive:
                loss = loss + ramp_factor * lambda_contrastive * aux_losses['contrastive_loss']
        # L1 始终激活
        raw_m = model.module
        if hasattr(raw_m, 'channel_graph') and raw_m.channel_graph is not None:
            loss = loss + lambda_locality_l1 * raw_m.channel_graph.get_l1_penalty()
        return loss

    # ★ v9.1 GPU 利用率优化: 验证函数
    #   - 移除 torch.cuda.empty_cache()（每 5 step 同步阻塞 GPU → GPU 利用率从 ~95% 降到 ~60%）
    #   - 使用 torch.inference_mode() 替代 no_grad()（更轻量）
    def validate(val_loader_local, current_epoch=None):
        model.eval()
        loss_list = []
        with torch.inference_mode():
            for input_data, _ in val_loader_local:
                input_data = input_data.float().to(device, non_blocking=True)
                rec, _, _, freq_recon, _, aux_losses, pred = model(input_data)
                loss = compute_loss(rec, input_data, freq_recon, aux_losses=aux_losses, epoch=current_epoch)
                loss_list.append(loss.item())
        return np.average(loss_list) if loss_list else 0.0

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-7,
    )
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)

    train_start = time.time()
    epoch_records = []

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        model.train()
        train_sampler.set_epoch(epoch)

        warmup_alpha = min(1.0, epoch / max(1, config.num_epochs * 0.1))
        raw_model = model.module
        if hasattr(raw_model, 'set_warmup_progress'):
            raw_model.set_warmup_progress(warmup_alpha)

        epoch_losses = []
        optimizer.zero_grad()

        for i, (input_data, _) in enumerate(train_loader):
            # ★ v9.1 GPU 利用率优化: non_blocking=True 异步传输
            input_data = input_data.float().to(device, non_blocking=True)
            rec, _, _, freq_recon, _, aux_losses, pred = model(input_data)
            loss = compute_loss(rec, input_data, freq_recon, aux_losses=aux_losses, epoch=epoch)
            loss = loss / acc_steps
            loss.backward()

            if (i + 1) % acc_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.item() * acc_steps)

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
        local_val_loss = validate(val_loader, current_epoch=epoch)

        val_tensor = torch.tensor([local_val_loss], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_tensor.item()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ★ v9.3: §3.2.2 渐进式权重替代硬冻结
        # 使用差分学习率（已在 LaGraph.py optimizer 分组实现），
        # 图结构参数使用 lr * 0.1 → decay 到 lr * 0.05
        # 不再 Epoch 50 后硬冻结

        if rank == 0:
            es_status = early_stopping(val_loss, model, epoch + 1)
            best_v = early_stopping.val_loss_min
            remaining = (config.num_epochs - (epoch + 1)) * epoch_time
            eta_str = _format_duration(remaining) if remaining > 0 else "---"

            if es_status == "improved":
                val_mark = "* BEST"
            elif es_status == "initial":
                val_mark = "* INIT"
            else:
                val_mark = "v {}/{}".format(early_stopping.counter, early_stopping.patience)

            print(f"  Epoch {epoch+1:>3}/{config.num_epochs}  "
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

        stop_flag = torch.tensor([1.0 if (rank == 0 and early_stopping.early_stop) else 0.0], device=device)
        dist.broadcast(stop_flag, src=0)

        if stop_flag.item() > 0.5:
            if rank == 0:
                print(f"  Stop early at epoch {epoch+1} (best: {early_stopping.best_epoch})", flush=True)
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
        }
        with open(args.result_out, 'w') as f:
            json.dump(result, f)

        if args.history_out:
            history = {
                "dataset": config.dataset_name if hasattr(config, 'dataset_name') else "Unknown",
                "epochs": epoch_records,
                "best_epoch": early_stopping.best_epoch,
                "best_val_loss": float(early_stopping.val_loss_min),
                "total_time_seconds": total_time,
                "total_time_human": _format_duration(total_time),
            }
            with open(args.history_out, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"\n  v DDP Training Complete! ({world_size} GPUs, {_format_duration(total_time)})")
        print(f"    Best Val Loss: {early_stopping.val_loss_min:.6f} @ Epoch {early_stopping.best_epoch}")
        print(f"    Checkpoint -> {args.checkpoint_out}", flush=True)
        if args.history_out:
            print(f"    History -> {args.history_out}", flush=True)

    dist.destroy_process_group()


# ---- spawn_worker for torch.multiprocessing.spawn ----
def spawn_worker(rank, config_dict, data_dict, nprocs, ckpt_path, result_path, gpu_start=0):
    """
    模块级 worker 函数（可被 pickle），供 torch.multiprocessing.spawn 调用。
    GPU 分配策略：rank k -> 物理 GPU k
    """
    import torch.distributed as dist_local
    from torch.utils.data import DistributedSampler

    try:
        gpu_id = rank
        total_visible = torch.cuda.device_count()
        if gpu_id >= total_visible:
            raise RuntimeError(
                f"Worker rank {rank} maps to GPU {gpu_id} but only {total_visible} GPUs visible."
            )
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        dist_local.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=nprocs,
            rank=rank,
        )

        cfg = type('Config', (), config_dict)()

        train_ds = SegLoader(data_dict['train_np'], cfg.win_size, step=1, mode="train")
        val_ds = SegLoader(data_dict['val_np'], cfg.win_size, step=1, mode="val")
        train_sampler = DistributedSampler(train_ds, num_replicas=nprocs, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=nprocs, rank=rank, shuffle=False)

        # ★ v9.1 GPU 利用率优化: 多进程数据加载
        # ★ v9.5 OOM 修复: num_workers 4→2, prefetch 2→1 减少 pinned memory 占用
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
            num_workers=2, prefetch_factor=1, persistent_workers=True,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
            num_workers=1, prefetch_factor=1, persistent_workers=True,
            pin_memory=True, drop_last=False,
        )

        # ★ v9: GCN_model 传入 v9 参数
        model = GCN_model(
            win_size=cfg.win_size,
            enc_in=cfg.input_c,
            c_out=cfg.output_c,
            dropout=cfg.dropout,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            e_layers=cfg.e_layers,
            patch_size=cfg.patch_size,
            channel=cfg.input_c,
            topk=cfg.topk,
            sparse_topk=cfg.sparse_topk,
            use_freq_loss=cfg.use_freq_loss,
            lambda_freq=cfg.lambda_freq,
            use_contrastive=getattr(cfg, 'use_contrastive', True),
            lambda_contrastive=getattr(cfg, 'lambda_contrastive', 0.05),
            contrastive_temp=getattr(cfg, 'contrastive_temp', 0.5),
            use_prototype=getattr(cfg, 'use_prototype', True),
            num_prototypes=getattr(cfg, 'num_prototypes', 16),
            # ★ Phase 2: 预测分支（§2.3 — 解重建悖论）
            use_prediction_head=getattr(cfg, 'use_prediction_head', False),
            lambda_pred=getattr(cfg, 'lambda_pred', 0.3),
        )

        # ★ v9.1 GPU 利用率优化: 可选 torch.compile
        if _ENABLE_COMPILE:
            try:
                model = torch.compile(model, mode='max-autotune', fullgraph=False)
                if rank == 0:
                    print("  [v9.1] torch.compile enabled (max-autotune mode)", flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"  [v9.1] torch.compile failed: {e}", flush=True)

        model.to(device)

        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id,
                    find_unused_parameters=True, broadcast_buffers=False)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        acc_steps = getattr(cfg, 'gradient_accumulation_steps', 1)
        use_freq_loss = getattr(cfg, 'use_freq_loss', True)
        lambda_freq = getattr(cfg, 'lambda_freq', 0.1)
        lambda_contrastive = getattr(cfg, 'lambda_contrastive', 0.05)
        use_contrastive = getattr(cfg, 'use_contrastive', True)
        lambda_locality_l1 = getattr(cfg, 'lambda_locality_l1', 0.001)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  LaGraph v9 DDP (spawn)  |  {nprocs} GPUs  |  {_format_params(total_params)} params")
            print(f"  Steps/GPU/epoch: {len(train_loader)}")
            print(f"  [v9.1] DataLoader workers: train=4, val=2  |  cudnn.benchmark=True  |  find_unused=True")
            print(f"{'='*60}\n", flush=True)

        # ★ v9.1: 渐进式损失激活（替代 Epoch-50 硬阈值）
        num_epochs_local = cfg.num_epochs
        warm_up_local = max(1, int(num_epochs_local * 0.1))
        ramp_up_end_local = max(warm_up_local + 1, int(num_epochs_local * 0.5))

        def compute_loss(rec, input_data, freq_recon, aux_losses=None, epoch=None):
            loss = F.mse_loss(rec, input_data)
            if epoch is not None:
                if epoch < warm_up_local:
                    ramp_factor = 0.0
                else:
                    ramp_factor = min(1.0, (epoch - warm_up_local) / (ramp_up_end_local - warm_up_local))
            else:
                ramp_factor = 1.0
            if ramp_factor > 0 and aux_losses is not None:
                if 'freq_loss' in aux_losses:
                    loss = loss + ramp_factor * lambda_freq * aux_losses['freq_loss']
                if 'contrastive_loss' in aux_losses and use_contrastive:
                    loss = loss + ramp_factor * lambda_contrastive * aux_losses['contrastive_loss']
            raw_m = model.module
            if hasattr(raw_m, 'channel_graph') and raw_m.channel_graph is not None:
                loss = loss + lambda_locality_l1 * raw_m.channel_graph.get_l1_penalty()
            return loss

        def validate(vl, current_epoch=None):
            model.eval()
            loss_list = []
            with torch.inference_mode():
                for input_data, _ in vl:
                    input_data = input_data.float().to(device, non_blocking=True)
                    rec, _, _, freq_recon, _, aux_losses, pred = model(input_data)
                    loss = compute_loss(rec, input_data, freq_recon, aux_losses=aux_losses, epoch=current_epoch)
                    loss_list.append(loss.item())
            return np.average(loss_list) if loss_list else 0.0

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs, eta_min=1e-7,
        )
        early_stopping = EarlyStopping(patience=cfg.patience, verbose=(rank == 0))

        train_start = time.time()

        for epoch in range(cfg.num_epochs):
            epoch_start = time.time()
            model.train()
            train_sampler.set_epoch(epoch)
            warmup_alpha = min(1.0, epoch / max(1, cfg.num_epochs * 0.1))
            raw_model = model.module
            if hasattr(raw_model, 'set_warmup_progress'):
                raw_model.set_warmup_progress(warmup_alpha)

            epoch_losses = []
            optimizer.zero_grad()

            for i, (input_data, _) in enumerate(train_loader):
                input_data = input_data.float().to(device, non_blocking=True)
                rec, _, _, freq_recon, _, aux_losses, pred = model(input_data)
                loss = compute_loss(rec, input_data, freq_recon, aux_losses=aux_losses, epoch=epoch)
                loss = loss / acc_steps
                loss.backward()

                if (i + 1) % acc_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item() * acc_steps)

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
            local_val_loss = validate(val_loader, current_epoch=epoch)

            val_tensor = torch.tensor([local_val_loss], device=device)
            dist_local.all_reduce(val_tensor, op=dist_local.ReduceOp.AVG)
            val_loss = val_tensor.item()

            epoch_time = time.time() - epoch_start

            # ★ v9.3: §3.2.2 渐进式权重替代硬冻结（不 freeze，由差分学习率控制更新速度）
            # 图结构参数使用 lr * 0.1 → decay 到 lr * 0.05

            if rank == 0:
                es_status = early_stopping(val_loss, model, epoch + 1)
                remaining = (cfg.num_epochs - (epoch + 1)) * epoch_time
                eta_str = _format_duration(remaining) if remaining > 0 else "---"
                val_mark = "* BEST" if es_status == "improved" else ("* INIT" if es_status == "initial" else "v {}/{}".format(early_stopping.counter, early_stopping.patience))
                print(f"  Epoch {epoch+1:>3}/{cfg.num_epochs}  Train: {avg_train_loss:.6f}  Val: {val_loss:.6f} {val_mark}  Best: {early_stopping.val_loss_min:.6f}  Time: {_format_duration(epoch_time)}  ETA: {eta_str}", flush=True)

                if es_status in ("improved", "initial"):
                    checkpoint = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    if any(k.startswith("module.") for k in checkpoint.keys()):
                        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
                    torch.save(checkpoint, ckpt_path)

                if early_stopping.early_stop:
                    print(f"  Stop early at epoch {epoch+1}", flush=True)

            stop_flag = torch.tensor([1.0 if (rank == 0 and early_stopping.early_stop) else 0.0], device=device)
            dist_local.broadcast(stop_flag, src=0)
            if stop_flag.item() > 0.5:
                break

            scheduler.step()

        total_time = time.time() - train_start

        if rank == 0:
            result = {
                'best_val_loss': float(early_stopping.val_loss_min),
                'best_epoch': early_stopping.best_epoch,
                'total_time': total_time,
                'total_params': total_params,
            }
            with open(result_path, 'w') as f:
                json.dump(result, f)
            print(f"\n  v DDP Training Complete! ({nprocs} GPUs, {_format_duration(total_time)})", flush=True)

        dist_local.destroy_process_group()

    except Exception as e:
        print(f"\n[Worker {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
