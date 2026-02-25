# run_all.py
"""End-to-end pipeline: train → LIF validate → NEST → plot."""
import os
import sys
import numpy as np

CKPT = "checkpoints"
os.makedirs(CKPT, exist_ok=True)

# ── Stage 1: Train rate network ───────────────────────────────────────────────
print("=" * 60)
print("STAGE 1: Training rate RNN (PyTorch)")
print("=" * 60)
from train_rate import train
model, rate_perf = train(save_dir=CKPT)

if rate_perf < 0.90:
    print("\nRate network did not reach 90%. Try increasing N_TRAIN in config.py.")
    sys.exit(1)

# ── Stage 2: Transfer + Python LIF validation ─────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: Weight transfer + Python LIF validation")
print("=" * 60)
from transfer import get_lif_weights
W_spk, W_out_spk, W_in, d_signs, lam = get_lif_weights(CKPT)

import config as cfg
from lif_network import LIFNetwork
from task import generate_batch, evaluate_output

rng = np.random.default_rng(42)
net_lif = LIFNetwork(W_spk, W_out_spk, W_in)
outputs_list, is_go_list = [], []
print("Evaluating Python LIF on 100 trials...")
for _ in range(100):
    inputs, targets, is_go = generate_batch(1, rng=rng)
    result = net_lif.run_trial(inputs[0])
    outputs_list.append(result["output"][np.newaxis])
    is_go_list.append(is_go)

outputs_all = np.concatenate(outputs_list, axis=0)
is_go_all   = np.concatenate(is_go_list)
targets_all = np.zeros_like(outputs_all)
tgt_s = int(cfg.TARGET_START / cfg.DT)
targets_all[is_go_all, tgt_s:, 0] = 1.0
lif_perf = evaluate_output(outputs_all, targets_all, is_go_all)
print(f"Python LIF accuracy: {lif_perf*100:.1f}%")

if lif_perf < 0.90:
    print("LIF accuracy below 90% — continuing to NEST for diagnostics.")

# ── Stage 3: NEST simulation ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3: NEST simulation")
print("=" * 60)
from nest_simulation import run_gonogo_nest
nest_results = run_gonogo_nest(CKPT, n_go=5, n_nogo=5)

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)
from evaluate import plot_all
plot_all(outputs_all, is_go_all, nest_results, save_dir=CKPT)

print("\nDone. Check the 'checkpoints/' directory for:")
print("  - results_summary.png")
print("  - accuracy_summary.txt")
