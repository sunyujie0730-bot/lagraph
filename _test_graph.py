"""Quick test of graph_learner + gcn_model — v8 edition (5 output API, no VQ)"""
import sys, torch, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from ts_benchmark.baselines.self_impl.LaGraph.graph_learner import (
    CausalConditionedTempGraph, ChannelAdaptiveGraph, GateFusion
)
from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import GCN_model

B, L, C = 4, 100, 25
x = torch.randn(B, L, C)

# 1. ChannelAdaptiveGraph (v8 data-driven, no DAG)
ccg = ChannelAdaptiveGraph(num_nodes=C, topk=5, dropout=0.1)
x_c, A_c = ccg(x)
diff = (A_c - A_c.transpose(-1, -2)).abs().max().item()
assert diff > 1e-5, f'Causal is symmetric (diff={diff})'
print(f'1. Adaptive   x={list(x_c.shape)}  A={list(A_c.shape)}  diff={diff:.4f} OK')

# 2. CausalConditionedTempGraph (v8 Cross-Attention)
cctg = CausalConditionedTempGraph(win_size=L, num_nodes=C, topk=5, dropout=0.1)
x_temp, A_temp = cctg(x_c, A_c)
max_asym = (A_temp - A_temp.transpose(-1, -2)).abs().max().item()
assert max_asym < 1e-4, f'Temp A asymmetry={max_asym:.2e}'
print(f'2. CCTemp     x={list(x_temp.shape)}  A={list(A_temp.shape)}  max_asym={max_asym:.2e} OK')

# 3. GateFusion
gf = GateFusion(num_nodes=C)
x_f, gate = gf(x_temp, x_c)
print(f'3. Fusion     x={list(x_f.shape)}  gate={list(gate.shape)} OK')

# 4. Full model forward + backward (v8: no VQ, no DAG, 5 outputs)
model = GCN_model(win_size=L, enc_in=C, c_out=C, dropout=0.1, n_heads=4,
                  d_model=256, e_layers=2, d_ff=512, patch_size=16,
                  channel=C, topk=5, use_freq_loss=True, lambda_freq=0.1)
x_rec, A_causal, gate_val, freq_recon, stage1_feat = model(x)
print(f'4. Model outputs:')
print(f'     rec={list(x_rec.shape)}  A_cause={list(A_causal.shape)}')
print(f'     gate={list(gate_val.shape)}  freq_recon={list(freq_recon.shape)}')
print(f'     stage1_feat={list(stage1_feat.shape)}')

# Compute loss v8 style (no VQ, no DAG)
mse = torch.nn.functional.mse_loss(x_rec, x)

# L1 sparse only
cause_loss = model.get_sparse_loss()  # L1 only (no DAG term)

# freq loss
freq_fft = torch.fft.rfft(freq_recon, dim=1, norm='ortho')
input_fft = torch.fft.rfft(x, dim=1, norm='ortho')
freq_mag_loss = torch.nn.functional.mse_loss(torch.abs(freq_fft), torch.abs(input_fft))
freq_phase_cos = torch.nn.functional.cosine_similarity(
    torch.angle(freq_fft).flatten(1), torch.angle(input_fft).flatten(1), dim=-1
).mean()
freq_loss = freq_mag_loss + 0.5 * (1.0 - freq_phase_cos)

loss = mse + 0.1 * freq_loss + cause_loss
loss.backward()

p = sum(p.numel() for p in model.parameters())
print(f'     loss={loss.item():.4f}  mse={mse.item():.4f}  freq={freq_loss.item():.4f}  params={p:,} OK')

print()
print('ALL 4 TESTS PASSED (v8)')
