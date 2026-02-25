# transfer.py
"""
Weight transfer from rate RNN to LIF network.
Grid search over 1/lambda to find the optimal scaling factor.
"""
import numpy as np
import os
import config as cfg
from task import generate_batch, evaluate_output
from lif_network import LIFNetwork


def load_rate_weights(checkpoint_dir: str = "checkpoints"):
    data = np.load(os.path.join(checkpoint_dir, "weights.npz"))
    return (data["W_constrained"],   # (N, N)
            data["W_out"],            # (1, N)
            data["W_in"],             # (N, 1)
            data["d_signs"])          # (N,)


def grid_search_lambda(W_rate, W_out_rate, W_in,
                       n_eval_trials: int = 60,
                       seed: int = 99,
                       verbose: bool = True) -> tuple:
    """
    For each candidate 1/lambda, run n_eval_trials of the LIF network
    and record accuracy. Returns (best_lambda, results_dict).
    """
    rng = np.random.default_rng(seed)
    results = {}

    if verbose:
        print(f"\nGrid searching lambda (1/λ from {cfg.INV_LAMBDA_GRID[0]}"
              f" to {cfg.INV_LAMBDA_GRID[-1]}):")

    for inv_lam in cfg.INV_LAMBDA_GRID:
        lam = 1.0 / inv_lam
        W_spk     = lam * W_rate
        W_out_spk = lam * W_out_rate

        net = LIFNetwork(W_spk, W_out_spk, W_in)
        outputs_list, is_go_list = [], []

        for _ in range(n_eval_trials):
            inputs, targets, is_go = generate_batch(1, rng=rng)
            result = net.run_trial(inputs[0])
            outputs_list.append(result["output"][np.newaxis])
            is_go_list.append(is_go)

        outputs_all = np.concatenate(outputs_list, axis=0)
        is_go_all   = np.concatenate(is_go_list)
        targets_all = np.zeros_like(outputs_all)
        tgt_s = int(cfg.TARGET_START / cfg.DT)
        targets_all[is_go_all, tgt_s:, 0] = 1.0

        perf = evaluate_output(outputs_all, targets_all, is_go_all)
        results[inv_lam] = perf

        if verbose:
            bar = "█" * int(perf * 20) + "░" * (20 - int(perf * 20))
            print(f"  1/λ = {inv_lam:3d} | {bar} {perf*100:5.1f}%")

    best_inv_lam = max(results, key=results.get)
    best_lam     = 1.0 / best_inv_lam
    best_perf    = results[best_inv_lam]

    if verbose:
        print(f"\n✓ Best: 1/λ = {best_inv_lam} (λ = {best_lam:.4f}) "
              f"→ {best_perf*100:.1f}%\n")

    return best_lam, results


def get_lif_weights(checkpoint_dir: str = "checkpoints",
                    verbose: bool = True) -> tuple:
    """
    Full pipeline: load rate weights → grid search → return LIF weights.
    Returns: (W_spk, W_out_spk, W_in, d_signs, best_lambda)
    """
    W_rate, W_out_rate, W_in, d_signs = load_rate_weights(checkpoint_dir)
    best_lam, _ = grid_search_lambda(W_rate, W_out_rate, W_in,
                                     verbose=verbose)
    W_spk     = best_lam * W_rate
    W_out_spk = best_lam * W_out_rate
    np.savez(os.path.join(checkpoint_dir, "lif_weights.npz"),
             W_spk=W_spk, W_out_spk=W_out_spk, W_in=W_in,
             d_signs=d_signs, best_lambda=best_lam)
    return W_spk, W_out_spk, W_in, d_signs, best_lam


if __name__ == "__main__":
    W_spk, W_out_spk, W_in, d_signs, lam = get_lif_weights()
    print(f"LIF weights saved. λ = {lam:.5f}")
