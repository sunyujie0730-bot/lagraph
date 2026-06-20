"""Small invariant checks for the strict cross-channel SPS building blocks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import (
    SPSRoleHead,
    SparseGCN,
    StrictCrossLagMechanism,
)
from ts_benchmark.baselines.self_impl.LaGraph.LaGraph import LaGraph


def main():
    torch.manual_seed(7)
    mechanism = StrictCrossLagMechanism(
        channel=5,
        lags=(1, 3),
        topk=2,
        use_static_prior=True,
    )
    prior = torch.ones(5, 5) - torch.eye(5)
    mechanism.set_static_prior(prior)
    x = torch.randn(3, 12, 5)

    full_pred, self_pred, attention, edge_weight = mechanism(x)
    assert full_pred.shape == x.shape
    assert self_pred.shape == x.shape
    assert torch.allclose(
        torch.diagonal(attention, dim1=1, dim2=2),
        torch.zeros_like(torch.diagonal(attention, dim1=1, dim2=2)),
        atol=1e-7,
    )

    # A target's observed current value cannot affect its same-step prediction.
    changed = x.clone()
    target = 2
    current_time = mechanism.max_lag
    changed[:, current_time, target] += 100.0
    changed_full, changed_self, _, _ = mechanism(changed)
    assert torch.allclose(
        full_pred[:, current_time, target],
        changed_full[:, current_time, target],
        atol=1e-6,
    )
    assert torch.allclose(
        self_pred[:, current_time, target],
        changed_self[:, current_time, target],
        atol=1e-6,
    )

    with torch.no_grad():
        mechanism.cross_transfer_raw.zero_()
    full_without_transfer, self_without_transfer, _, _ = mechanism(x)
    assert torch.allclose(full_without_transfer, self_without_transfer, atol=1e-6)
    assert torch.allclose(edge_weight.diagonal(dim1=1, dim2=2), torch.zeros_like(edge_weight.diagonal(dim1=1, dim2=2)))

    role_head = SPSRoleHead(feature_dim=4, hidden=12)
    root_logits, response_logits = role_head(torch.randn(3, 12, 5, 4))
    assert root_logits.shape == (3, 12, 5)
    assert response_logits.shape == (3, 12, 5)

    model = SparseGCN(
        win_size=12,
        enc_in=5,
        c_out=5,
        dropout=0.0,
        n_heads=1,
        d_model=16,
        e_layers=1,
        patch_size=4,
        channel=5,
        use_channel_graph=False,
        use_temporal_graph=False,
        use_vq_bypass=False,
        use_multi_scale_scorer=False,
        use_strict_cross_mechanism=True,
        strict_cross_lags=(1, 3),
        strict_cross_topk=2,
        strict_cross_use_channel_prior=True,
        use_sps_role_head=True,
    )
    model.set_strict_cross_static_prior(prior)
    rec, _, _, _, _, aux, _ = model(x)
    assert aux["strict_cross_response_gain"].shape == x.shape
    assert aux["sps_root_logits"].shape == x.shape
    loss = (rec - x).abs().mean() + aux["strict_cross_mechanism_loss"]
    loss = loss + aux["sps_root_logits"].square().mean()
    loss.backward()
    assert model.strict_cross_mechanism.cross_edge_logits.grad is not None
    assert model.sps_role_head.source.weight.grad is not None

    rng = np.random.default_rng(7)
    normal = rng.normal(size=(160, 5)).astype(np.float32)
    normal[1:, 1] += 0.55 * normal[:-1, 0]
    normal[1:, 2] += 0.45 * normal[:-1, 1]
    trainer = LaGraph(
        n_gpus=0,
        use_sps_teacher=True,
        sps_teacher_lags=[1],
        sps_teacher_topk=2,
        sps_teacher_response_ratio=0.02,
        source_effect_min_len=5,
        source_effect_max_len=8,
        use_source_effect_synthetic=True,
        lambda_source_effect=1.0,
        lambda_sps_source=1.0,
        lambda_sps_response=0.5,
        lambda_sps_separation=0.5,
    )
    trainer._fit_sps_teacher(pd.DataFrame(normal))
    generated = trainer._make_teacher_source_effect_synthetic_batch(
        torch.as_tensor(normal[:12]).unsqueeze(0).repeat(3, 1, 1)
    )
    synth, event_mask, source_mask, effect_mask, _, effect_time, propagated = generated
    assert synth.shape == (3, 12, 5)
    assert torch.all(source_mask.sum(dim=-1) >= 1)
    assert torch.all(event_mask.sum(dim=-1) >= 1)
    assert torch.all(propagated >= 0)
    assert torch.all(effect_mask * source_mask == 0)
    assert torch.all(effect_time.sum(dim=-1) >= 0)

    trainer.model = model
    model.zero_grad(set_to_none=True)
    model.train()
    sps_loss = trainer._source_effect_synthetic_loss(
        torch.as_tensor(normal[:12]).unsqueeze(0).repeat(3, 1, 1)
    )
    assert torch.isfinite(sps_loss)
    sps_loss.backward()
    assert model.sps_role_head.response.weight.grad is not None
    print("Strict cross SPS invariants passed.")


if __name__ == "__main__":
    main()
