# evaluate.py
"""Plotting utilities for the Go-NoGo POC."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import config as cfg


def plot_all(lif_outputs_all, is_go_all, nest_results,
             save_dir="checkpoints"):

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    t_axis = np.arange(cfg.T_STEPS) * cfg.DT / 1000.0   # seconds

    # ── Panel 1: Python LIF output traces ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    go_outputs   = lif_outputs_all[is_go_all,  :, 0]
    nogo_outputs = lif_outputs_all[~is_go_all, :, 0]

    if go_outputs.shape[0] > 0:
        ax1.plot(t_axis, go_outputs.mean(0), color="#ef4444", lw=2, label="Go (mean)")
        ax1.fill_between(t_axis,
                         go_outputs.mean(0) - go_outputs.std(0),
                         go_outputs.mean(0) + go_outputs.std(0),
                         alpha=0.2, color="#ef4444")
    if nogo_outputs.shape[0] > 0:
        ax1.plot(t_axis, nogo_outputs.mean(0), color="#3b82f6", lw=2, label="NoGo (mean)")
        ax1.fill_between(t_axis,
                         nogo_outputs.mean(0) - nogo_outputs.std(0),
                         nogo_outputs.mean(0) + nogo_outputs.std(0),
                         alpha=0.2, color="#3b82f6")
    ax1.axvspan(cfg.PULSE_START / 1000, cfg.PULSE_END / 1000,
                alpha=0.15, color="green", label="Input pulse")
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_title("Python LIF Output Traces", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Output (a.u.)")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.2, 1.3)

    # ── Panel 2: NEST output traces ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for r in nest_results["go"]:
        ax2.plot(t_axis, r["output"][:, 0], color="#ef4444", alpha=0.6, lw=1)
    for r in nest_results["nogo"]:
        ax2.plot(t_axis, r["output"][:, 0], color="#3b82f6", alpha=0.6, lw=1)
    ax2.axvspan(cfg.PULSE_START / 1000, cfg.PULSE_END / 1000,
                alpha=0.15, color="green")
    ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax2.set_title("NEST Output Traces", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Output (a.u.)")

    # ── Panels 3+4: NEST spike rasters ─────────────────────────────────────────
    for col, (cond, results, color) in enumerate([
        ("Go",   nest_results["go"],   "#ef4444"),
        ("NoGo", nest_results["nogo"], "#3b82f6"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        spikes = results[0]["spikes"] if results else []
        N = cfg.N
        for (unit_idx, t_ms) in spikes:
            if 0 <= unit_idx < N:
                ax.plot(t_ms / 1000.0, unit_idx, "|",
                        color=color if unit_idx < cfg.N_EXC else "#818cf8",
                        markersize=2, markeredgewidth=0.5, alpha=0.7)
        ax.axvspan(cfg.PULSE_START / 1000, cfg.PULSE_END / 1000,
                   alpha=0.1, color="green")
        ax.set_xlim(0, cfg.TRIAL_MS / 1000.0)
        ax.set_ylim(0, N)
        ax.set_title(f"NEST Spike Raster — {cond}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron index")
        ax.axhline(cfg.N_EXC, color="white", lw=0.5, alpha=0.4)

    # ── Panel 5: Firing rate histogram ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    all_go_spikes = nest_results["go"][0]["spikes"] if nest_results["go"] else []
    counts = np.zeros(cfg.N)
    for (uid, _) in all_go_spikes:
        if 0 <= uid < cfg.N:
            counts[uid] += 1
    rates = counts / (cfg.TRIAL_MS / 1000.0)
    active_rates = rates[rates > 0]
    if len(active_rates) > 0:
        ax5.hist(active_rates, bins=20, color="#2dd4bf", edgecolor="black",
                 linewidth=0.5, alpha=0.8)
        mean_r = active_rates.mean()
        ax5.axvline(mean_r, color="red", ls="--", lw=1.5,
                    label=f"Mean = {mean_r:.1f} Hz")
        ax5.legend(fontsize=8)
    ax5.set_title("Firing Rate Distribution\n(Go trial 1)", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Firing rate (Hz)")
    ax5.set_ylabel("Count")

    plt.suptitle("Kim et al. 2019 POC — Go-NoGo Task Results",
                 fontsize=14, fontweight="bold", y=1.01)

    out_path = os.path.join(save_dir, "results_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)

    # Text summary
    go_correct = sum(
        1 for r in nest_results["go"]
        if float(r["output"][-10:, 0].mean()) > 0.5
    )
    nogo_correct = sum(
        1 for r in nest_results["nogo"]
        if float(r["output"][-10:, 0].mean()) < 0.5
    )
    n_go   = len(nest_results["go"])
    n_nogo = len(nest_results["nogo"])
    nest_acc = (go_correct + nogo_correct) / max(n_go + n_nogo, 1)

    summary = f"""
Kim et al. 2019 POC — Go-NoGo Results Summary
==============================================
Lambda (λ):               {nest_results['lambda']:.5f}  (1/λ ≈ {1/nest_results['lambda']:.0f})

NEST accuracy (approx):   {nest_acc*100:.0f}%  ({go_correct}/{n_go} Go, {nogo_correct}/{n_nogo} NoGo correct)
NEST Go    spikes (trial 1): {len(nest_results['go'][0]['spikes'])   if nest_results['go']   else 'N/A'}
NEST NoGo  spikes (trial 1): {len(nest_results['nogo'][0]['spikes']) if nest_results['nogo'] else 'N/A'}
"""
    summary_path = os.path.join(save_dir, "accuracy_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(summary)
