# train_rate.py
"""
BPTT training of the rate RNN on the Go-NoGo task.
Saves model weights when training criterion is met.
Uses PyTorch instead of TensorFlow (same math).
"""
import numpy as np
import torch
import os
import json
import config as cfg
from task import generate_batch, evaluate_output
from rate_network import RateRNN


def rmse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((output - target) ** 2))


def train(save_dir: str = "checkpoints", seed: int = 42):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = RateRNN(seed=seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best_perf = 0.0
    rng_eval  = np.random.default_rng(seed + 1)

    print(f"Training | N={cfg.N} | batch={cfg.BATCH_SIZE} | max_steps={cfg.N_TRAIN}")

    for step in range(cfg.N_TRAIN):

        inputs, targets, _ = generate_batch(cfg.BATCH_SIZE, rng=rng)
        inputs_t  = torch.tensor(inputs)
        targets_t = torch.tensor(targets)

        optimizer.zero_grad()
        outputs, _ = model(inputs_t, training=True)
        loss = rmse_loss(outputs, targets_t)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % cfg.EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eval_inputs, eval_targets, eval_is_go = generate_batch(
                    200, rng=rng_eval
                )
                eval_outputs, _ = model(torch.tensor(eval_inputs), training=False)
            perf = evaluate_output(
                eval_outputs.numpy(), eval_targets, eval_is_go
            )
            print(f"  Step {step:4d} | loss={loss.item():.4f} | "
                  f"accuracy={perf*100:.1f}%")
            model.train()

            if perf > best_perf:
                best_perf = perf
                _save(model, save_dir, step, perf)

            if perf >= cfg.PERF_THRESH:
                print(f"\n✓ Target reached ({perf*100:.1f}%) at step {step}")
                break

    print(f"\nBest performance: {best_perf*100:.1f}%")
    if best_perf < cfg.PERF_THRESH:
        print("Warning: did not reach 95%")
    return model, best_perf


def _save(model, save_dir, step, perf):
    W, W_out, W_in, d_signs = model.get_numpy_weights()
    np.savez(
        os.path.join(save_dir, "weights.npz"),
        W_constrained=W,
        W_out=W_out,
        W_in=W_in,
        d_signs=d_signs,
    )
    meta = {"step": int(step), "accuracy": float(perf),
            "N": cfg.N, "TAU_D": cfg.TAU_D}
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    train()
