"""Controlled source-response benchmark for the strict cross-channel mechanism.

This is deliberately small and uses known delayed propagation paths.  It is a
mechanism check, not a replacement for WADI or SWaT evaluation.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import (  # noqa: E402
    SPSRoleHead,
    StrictCrossLagMechanism,
)


CHANNELS = 12
LENGTH = 48
ROOT_CHANNELS = 6
EDGES = (
    # The downstream response is intentionally amplified.  This prevents a
    # trivial onset-amplitude ranking from solving the benchmark by itself.
    (0, 6, 1, 1.20),
    (1, 7, 1, 1.10),
    (2, 8, 2, 1.20),
    (3, 9, 1, 1.05),
    (4, 10, 2, 1.10),
    (5, 11, 1, 1.20),
)


def make_batch(batch_size, device):
    noise = 0.15 * torch.randn(batch_size, LENGTH, CHANNELS, device=device)
    normal = torch.zeros_like(noise)
    for time_idx in range(LENGTH):
        value = noise[:, time_idx]
        if time_idx > 0:
            value = value + 0.55 * normal[:, time_idx - 1]
        for source, target, lag, weight in EDGES:
            if time_idx >= lag:
                value[:, target] = value[:, target] + weight * normal[:, time_idx - lag, source]
        normal[:, time_idx] = value

    roots = torch.randint(ROOT_CHANNELS, (batch_size,), device=device)
    start = torch.randint(10, 18, (batch_size,), device=device)
    source_delta = torch.zeros_like(normal)
    source_mask = torch.zeros(batch_size, CHANNELS, device=device)
    source_onset = torch.zeros(batch_size, LENGTH, device=device)
    for batch_idx in range(batch_size):
        root = int(roots[batch_idx].item())
        begin = int(start[batch_idx].item())
        amplitude = 2.5 if batch_idx % 2 == 0 else -2.5
        source_delta[batch_idx, begin:begin + 4, root] = amplitude
        source_mask[batch_idx, root] = 1.0
        source_onset[batch_idx, begin:begin + 2] = 1.0

    total_delta = source_delta.clone()
    for time_idx in range(LENGTH):
        propagated = torch.zeros(batch_size, CHANNELS, device=device)
        for source, target, lag, weight in EDGES:
            if time_idx >= lag:
                propagated[:, target] += weight * total_delta[:, time_idx - lag, source]
        total_delta[:, time_idx] += propagated

    response_delta = total_delta - source_delta
    effect_mask = (response_delta.abs().amax(dim=1) > 0.20).float()
    effect_mask = effect_mask * (1.0 - source_mask)
    effect_time = (
        (response_delta.abs() * effect_mask.unsqueeze(1)).sum(dim=-1) > 0.10
    ).float()
    return normal + total_delta, source_mask, effect_mask, source_onset, effect_time


def aggregate(values, mask):
    return (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


def relative_channel_feature(values):
    center = values.mean(dim=-1, keepdim=True)
    scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (values - center) / scale


def source_metrics(scores, source_mask):
    order = torch.argsort(scores, dim=-1, descending=True)
    roots = source_mask.argmax(dim=-1)
    ranks = (order == roots.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1
    ranks = ranks.float()
    return {
        "mrr": float((1.0 / ranks).mean().item()),
        "hit1": float((ranks <= 1).float().mean().item()),
        "hit3": float((ranks <= 3).float().mean().item()),
    }


def main():
    torch.manual_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mechanism = StrictCrossLagMechanism(
        channel=CHANNELS,
        lags=(1, 2),
        topk=3,
        dropout=0.0,
        use_static_prior=False,
    ).to(device)
    role_head = SPSRoleHead(feature_dim=4, hidden=24, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(
        list(mechanism.parameters()) + list(role_head.parameters()),
        lr=2e-3,
    )

    for _ in range(300):
        x, source_mask, effect_mask, source_onset, effect_time = make_batch(128, device)
        full_pred, self_pred, _, _ = mechanism(x)
        self_error = F.smooth_l1_loss(self_pred, x, reduction="none")
        full_error = F.smooth_l1_loss(full_pred, x, reduction="none")
        gain = (self_error - full_error).clamp_min(0.0)
        prev_error = torch.cat([torch.zeros_like(self_error[:, :1]), self_error[:, :-1]], dim=1)
        onset = (self_error - prev_error).clamp_min(0.0)
        features = torch.stack(
            [
                relative_channel_feature(torch.log1p(self_error)),
                relative_channel_feature(torch.log1p(gain)),
                relative_channel_feature(torch.log1p(onset)),
                relative_channel_feature(torch.log1p(x.abs())),
            ],
            dim=-1,
        )
        root_logits, response_logits = role_head(features)
        root_scores = aggregate(root_logits, source_onset)
        root_target = source_mask / source_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        root_loss = -(root_target * F.log_softmax(root_scores, dim=-1)).sum(dim=-1).mean()
        response_scores = aggregate(response_logits, effect_time)
        response_loss = F.binary_cross_entropy_with_logits(response_scores, effect_mask)
        source_gain = aggregate(gain, source_onset)
        effect_gain = aggregate(gain, effect_time)
        root_gain = (source_gain * source_mask).sum(dim=-1) / source_mask.sum(dim=-1).clamp_min(1.0)
        effect_gain_mean = (effect_gain * effect_mask).sum(dim=-1) / effect_mask.sum(dim=-1).clamp_min(1.0)
        separation_loss = F.relu(0.10 + root_gain - effect_gain_mean).mean()
        prediction_loss = F.smooth_l1_loss(full_pred[:, 2:], x[:, 2:])
        loss = prediction_loss + root_loss + 0.5 * response_loss + 0.5 * separation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    mechanism.eval()
    role_head.eval()
    with torch.no_grad():
        x, source_mask, effect_mask, source_onset, effect_time = make_batch(1024, device)
        full_pred, self_pred, _, _ = mechanism(x)
        self_error = F.smooth_l1_loss(self_pred, x, reduction="none")
        full_error = F.smooth_l1_loss(full_pred, x, reduction="none")
        gain = (self_error - full_error).clamp_min(0.0)
        prev_error = torch.cat([torch.zeros_like(self_error[:, :1]), self_error[:, :-1]], dim=1)
        onset = (self_error - prev_error).clamp_min(0.0)
        features = torch.stack(
            [
                relative_channel_feature(torch.log1p(self_error)),
                relative_channel_feature(torch.log1p(gain)),
                relative_channel_feature(torch.log1p(onset)),
                relative_channel_feature(torch.log1p(x.abs())),
            ],
            dim=-1,
        )
        root_logits, _ = role_head(features)
        learned_scores = aggregate(root_logits, source_onset)
        baseline_scores = aggregate(onset, source_onset)
        learned = source_metrics(learned_scores, source_mask)
        baseline = source_metrics(baseline_scores, source_mask)
        source_gain = aggregate(gain, source_onset)
        effect_gain = aggregate(gain, effect_time)
        source_gain_mean = (source_gain * source_mask).sum() / source_mask.sum().clamp_min(1.0)
        effect_gain_mean = (effect_gain * effect_mask).sum() / effect_mask.sum().clamp_min(1.0)

    print(f"device={device.type}, learned={learned}, onset_baseline={baseline}")
    print(
        "mean_propagation_gain "
        f"source={float(source_gain_mean):.4f}, effect={float(effect_gain_mean):.4f}"
    )
    if learned["mrr"] <= baseline["mrr"]:
        raise RuntimeError("SPS head did not beat the onset-only source ranking on the controlled benchmark")
    if effect_gain_mean <= source_gain_mean:
        raise RuntimeError("Cross-channel gain did not separate effects from sources")


if __name__ == "__main__":
    main()
