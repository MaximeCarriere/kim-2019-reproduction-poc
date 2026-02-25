# task.py
"""
Go-NoGo task generator.

Go trial:  50 ms input pulse → network should produce output → +1 (sustained)
NoGo trial: no input        → network should maintain output → 0
"""
import numpy as np
import config as cfg


def generate_batch(batch_size: int, go_fraction: float = 0.5,
                   rng: np.random.Generator = None):
    """
    Returns:
        inputs:  (batch_size, T_STEPS, 1)  — input signal to network
        targets: (batch_size, T_STEPS, 1)  — target output signal
        is_go:   (batch_size,)             — boolean, True = Go trial
    """
    if rng is None:
        rng = np.random.default_rng()

    T = cfg.T_STEPS
    inputs  = np.zeros((batch_size, T, 1), dtype=np.float32)
    targets = np.zeros((batch_size, T, 1), dtype=np.float32)

    pulse_s = int(cfg.PULSE_START  / cfg.DT)
    pulse_e = int(cfg.PULSE_END    / cfg.DT)
    tgt_s   = int(cfg.TARGET_START / cfg.DT)

    is_go = rng.random(batch_size) < go_fraction

    for b in range(batch_size):
        if is_go[b]:
            inputs[b, pulse_s:pulse_e, 0] = cfg.INPUT_AMP
            targets[b, tgt_s:, 0] = 1.0

    return inputs, targets, is_go


def evaluate_output(output: np.ndarray, targets: np.ndarray,
                    is_go: np.ndarray) -> float:
    """
    Accuracy: for each trial, check final 100ms window.
    Go trial correct   → mean output in window > 0.5
    NoGo trial correct → mean output in window < 0.5
    """
    window_steps = int(100 / cfg.DT)   # last 100 ms
    mean_final = output[:, -window_steps:, 0].mean(axis=1)

    go_correct   = (is_go)  & (mean_final > 0.5)
    nogo_correct = (~is_go) & (mean_final < 0.5)
    return float((go_correct | nogo_correct).mean())
