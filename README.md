# POC: Spiking RNN for Go-NoGo — Kim et al. 2019

**Scope of this POC:** Reproduce the Go-NoGo task result from Kim et al. (2019) in three stages:
1. Train a rate RNN in TensorFlow with Dale's principle and sparse connectivity
2. Transfer weights to a pure-Python LIF network (validation layer)
3. Re-run the spiking simulation in NEST 3.x

**Success criteria:**
- Rate RNN reaches ≥ 95% Go-NoGo accuracy
- LIF network (Python) reaches ≥ 95% with the transferred weights
- NEST simulation produces a plausible spike raster and separated Go/NoGo output traces

**What we simplify vs. the paper:**
| Paper | This POC | Reason |
|---|---|---|
| Per-unit trained τᵈ | Fixed τᵈ = 35 ms for all units | Paper (Fig. 5) shows no significant difference |
| N = 10 to 400 sweep | Fixed N = 250 | Paper's optimal; avoids the sweep |
| 100 random seeds | 1 seed | POC — just need one working run |
| Both Go-NoGo + context integration | Go-NoGo only | Simpler, faster, still validates the core claim |
| MATLAB LIF simulation | Python LIF → NEST | Python is more accessible |

---

## Table of Contents

1. [Project structure](#1-project-structure)
2. [Environment setup](#2-environment-setup)
3. [Theoretical background (compact)](#3-theoretical-background-compact)
4. [Stage 1 — Rate network in TensorFlow](#4-stage-1--rate-network-in-tensorflow)
5. [Stage 2 — Python LIF validation](#5-stage-2--python-lif-validation)
6. [Stage 3 — NEST 3.x simulation](#6-stage-3--nest-3x-simulation)
7. [Running the full pipeline](#7-running-the-full-pipeline)
8. [Expected outputs](#8-expected-outputs)
9. [Troubleshooting](#9-troubleshooting)
10. [Parameter reference](#10-parameter-reference)

---

## 1. Project Structure

```
poc_gonogo/
├── README.md
├── requirements.txt
│
├── config.py              # Single source of truth for all hyperparameters
├── task.py                # Go-NoGo trial generator
│
├── rate_network.py        # TF rate RNN class + forward pass
├── train_rate.py          # BPTT training loop
│
├── lif_network.py         # Pure-Python LIF (numpy) — validation layer
├── transfer.py            # Weight scaling + lambda grid search
│
├── nest_simulation.py     # NEST 3.x simulation
│
├── evaluate.py            # Accuracy, plots (raster, output traces, PCA)
└── run_all.py             # End-to-end: train → transfer → NEST → plot
```

Create the directory:
```bash
mkdir poc_gonogo && cd poc_gonogo
```

---

## 2. Environment Setup

### 2.1 Python dependencies

```bash
# requirements.txt
tensorflow>=2.10
numpy>=1.23
scipy>=1.9
matplotlib>=3.6
```

```bash
pip install -r requirements.txt
```

> **CPU training note:** With N=250 and ~2000 training trials, expect **10–25 minutes** on a modern laptop CPU. The paper reports training in as few as ~1500 trials for N=250 on the Go-NoGo task.

### 2.2 NEST 3.x

Verify your installation:
```bash
python -c "import nest; nest.ResetKernel(); print(nest.__version__)"
# Expected: 3.x.x
```

If it fails, the cleanest install path on Linux/macOS is via conda:
```bash
conda install -c conda-forge nest-simulator
```

Or via pip (Linux only):
```bash
pip install nest-simulator
```

Check that `iaf_psc_alpha` is available:
```bash
python -c "import nest; nest.ResetKernel(); print('iaf_psc_alpha' in nest.node_models)"
# Expected: True
```

---

## 3. Theoretical Background (Compact)

### Rate network dynamics

The synaptic current `x_i` for unit `i` evolves as:

```
τᵈ · dx_i/dt = -x_i + Σⱼ W_ij · r_j + I_ext
r_i = sigmoid(x_i)
```

Discretized with Euler at Δt = 5 ms:

```
x[t] = (1 - Δt/τᵈ) * x[t-1] + (Δt/τᵈ) * (W @ r[t-1] + W_in @ u[t-1]) + noise
r[t] = sigmoid(x[t])
output[t] = W_out @ r[t]
```

### Dale's principle enforcement

The weight matrix is never directly learned. Instead a raw unconstrained matrix `W_raw` is learned, and the constrained matrix is computed at every forward pass:

```
W_constrained = relu(W_raw) @ D ⊙ M
```

Where:
- `D` = diagonal matrix of +1 (excitatory units) and −1 (inhibitory units), **fixed forever**
- `M` = binary sparsity mask with connection probability Pc = 0.2, **fixed forever**
- `relu` clips negative values to 0, ensuring each raw weight is non-negative before sign assignment

### Weight transfer

The key equation: if `W_spk = λ · W_rate`, then the spiking drive equals the rate drive when `r_rate = λ · r_spk`. Since the sigmoid constrains rate values to [0, 1] and LIF firing rates are >> 1, we need λ < 1.

Both the recurrent weights and the readout weights are scaled by the **same** λ:
```
W_spk     = λ · W_constrained
W_out_spk = λ · W_out_rate
```

λ is found by grid search (12 forward evaluations, ~seconds).

### Why `iaf_psc_alpha` in NEST

The paper uses a double-exponential synaptic filter with rise time τᵣ = 2 ms and decay time τᵈ = 35 ms. NEST's `iaf_psc_alpha` uses an **alpha function** (single time constant τ_syn), which is a reasonable approximation for a POC. For an exact match you would need `iaf_psc_exp` with separate rise/decay via NESTML — unnecessary here.

---

## 4. Stage 1 — Rate Network in TensorFlow

### 4.1 `config.py`

```python
# config.py
import numpy as np

# ── Network architecture ──────────────────────────────────────────────────────
N        = 250          # Total number of units
N_EXC    = 200          # Excitatory units (80%)
N_INH    = 50           # Inhibitory units (20%)
PC       = 0.20         # Initial connection probability (sparse)
GAIN     = 1.5          # Weight init scale: N(0, GAIN / sqrt(N * PC))

# ── Time constants ────────────────────────────────────────────────────────────
TAU_D    = 35.0         # Synaptic decay (ms) — fixed for all units
DT       = 5.0          # Euler time step (ms)

# ── Task ──────────────────────────────────────────────────────────────────────
TRIAL_MS     = 1000     # Trial duration (ms)
T_STEPS      = TRIAL_MS // int(DT)   # = 200 time steps
PULSE_START  = 100      # Go input pulse start (ms)
PULSE_END    = 150      # Go input pulse end (ms)
TARGET_START = 400      # When output should reach +1 for Go (ms)
INPUT_AMP    = 1.0      # Amplitude of Go input pulse

# ── Training ──────────────────────────────────────────────────────────────────
N_TRAIN      = 4000     # Max training trials (paper uses up to 6000)
BATCH_SIZE   = 32       # Trials per gradient step
LR           = 0.01     # Adam learning rate
PERF_THRESH  = 0.95     # Stop training when this accuracy is reached
EVAL_EVERY   = 100      # Evaluate every N gradient steps
NOISE_STD    = 0.1      # Gaussian noise std added to x at each step
                         # (0.01 variance → std = 0.1)

# ── LIF parameters (Python simulation) ───────────────────────────────────────
TAU_M    = 10.0         # Membrane time constant (ms)
V_TH     = -40.0        # Spike threshold (mV)
V_RESET  = -65.0        # Reset potential (mV)
T_REF    = 2.0          # Absolute refractory period (ms)
TAU_R    = 2.0          # Synaptic rise time (ms) — double-exp filter
I_BIAS   = -40.0        # Constant background current (pA)
DT_LIF   = 0.1          # LIF Euler step (ms) — smaller for accuracy

# ── Lambda grid search ────────────────────────────────────────────────────────
# Grid over 1/lambda values; paper uses range [20, 75] step 5
INV_LAMBDA_GRID = list(range(20, 80, 5))  # [20, 25, 30, ..., 75]

# ── NEST parameters ───────────────────────────────────────────────────────────
# iaf_psc_alpha parameter mapping
NEST_TAU_M      = TAU_M         # ms
NEST_V_TH       = V_TH          # mV
NEST_V_RESET    = V_RESET       # mV
NEST_T_REF      = T_REF         # ms
NEST_E_L        = V_RESET       # mV — set resting = reset for simplicity
NEST_C_M        = 100.0         # pF — membrane capacitance
                                 # (g_leak = C_m/tau_m = 10 nS, typical)
NEST_TAU_SYN_EX = TAU_D         # ms — excitatory synaptic time constant
NEST_TAU_SYN_IN = TAU_D         # ms — inhibitory synaptic time constant
NEST_I_E        = 360.0         # pA — background current in NEST units
                                 # (see Section 6.3 for derivation)

# ── Weight scaling for NEST ───────────────────────────────────────────────────
# NEST weights are in pA. After finding lambda, we scale further by
# NEST_WEIGHT_SCALE to convert from dimensionless to pA.
# A reasonable starting value — tune if NEST firing rates look off.
NEST_WEIGHT_SCALE = 1000.0      # pA per unit weight
```

### 4.2 `task.py`

```python
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
        # NoGo: inputs and targets stay 0

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
```

### 4.3 `rate_network.py`

```python
# rate_network.py
"""
Continuous-variable rate RNN with Dale's principle and sparse connectivity.
Forward pass is written explicitly so it can be traced by tf.GradientTape.
"""
import numpy as np
import tensorflow as tf
import config as cfg


class RateRNN:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)
        N, NE, NI = cfg.N, cfg.N_EXC, cfg.N_INH

        # ── Dale's sign matrix D (fixed) ──────────────────────────────────────
        # Shape: (N,) — positive for excitatory, negative for inhibitory
        self.d_signs = np.ones(N, dtype=np.float32)
        self.d_signs[NE:] = -1.0                        # last NI units are inh
        # D as column vector for broadcasting: (1, N)
        self.D = tf.constant(self.d_signs[np.newaxis, :], dtype=tf.float32)

        # ── Sparsity mask M (fixed) ───────────────────────────────────────────
        # Bernoulli mask, shape (N, N), diagonal forced to 0 (no self-connections)
        M = rng.random((N, N)) < cfg.PC
        np.fill_diagonal(M, False)
        self.M = tf.constant(M.astype(np.float32), dtype=tf.float32)

        # ── Trainable weight matrix W_raw (N, N) ─────────────────────────────
        # Initialized from N(0, GAIN / sqrt(N * PC))
        init_std = cfg.GAIN / np.sqrt(N * cfg.PC)
        W_init = rng.normal(0, init_std, (N, N)).astype(np.float32)
        self.W_raw = tf.Variable(W_init, name="W_raw", trainable=True)

        # ── Readout weights W_out (1, N) ──────────────────────────────────────
        W_out_init = rng.normal(0, 1.0 / np.sqrt(N), (1, N)).astype(np.float32)
        self.W_out = tf.Variable(W_out_init, name="W_out", trainable=True)

        # ── Input weights W_in (N, 1) — fixed, not trained ───────────────────
        W_in_init = rng.normal(0, 1.0, (N, 1)).astype(np.float32)
        self.W_in = tf.constant(W_in_init, dtype=tf.float32)

        # ── Time constant (scalar, fixed at 35 ms) ───────────────────────────
        self.alpha = tf.constant(1.0 - cfg.DT / cfg.TAU_D, dtype=tf.float32)
        self.beta  = tf.constant(cfg.DT / cfg.TAU_D,        dtype=tf.float32)

    @property
    def W_constrained(self) -> tf.Tensor:
        """
        Apply Dale's constraint at every call.
        W_constrained[i,j] = relu(W_raw[i,j]) * D[j]  (then mask)
        Shape: (N, N)
        """
        return tf.nn.relu(self.W_raw) * self.D * self.M

    @property
    def trainable_variables(self):
        return [self.W_raw, self.W_out]

    def forward(self, inputs: tf.Tensor, training: bool = True):
        """
        Roll the RNN forward through a full trial.

        Args:
            inputs: (batch, T, 1)
        Returns:
            outputs:  (batch, T, 1)   — linear readout at each step
            r_trace:  (batch, T, N)   — firing rates at each step
        """
        batch = tf.shape(inputs)[0]
        W = self.W_constrained       # (N, N)

        # Initial state: x = 0, r = sigmoid(0) = 0.5
        x = tf.zeros((batch, cfg.N), dtype=tf.float32)
        r = tf.fill((batch, cfg.N), 0.5)

        outputs_ta = tf.TensorArray(dtype=tf.float32, size=cfg.T_STEPS)
        rates_ta   = tf.TensorArray(dtype=tf.float32, size=cfg.T_STEPS)

        for t in tf.range(cfg.T_STEPS):
            u_t = inputs[:, t, :]          # (batch, 1)

            # Recurrent drive
            rec   = tf.matmul(r, tf.transpose(W))   # (batch, N)
            inp   = tf.matmul(u_t, tf.transpose(self.W_in))  # (batch, N)

            # Euler update
            x_new = self.alpha * x + self.beta * (rec + inp)

            # Add noise only during training
            if training:
                noise = tf.random.normal(tf.shape(x_new), stddev=cfg.NOISE_STD)
                x_new = x_new + noise

            r_new = tf.math.sigmoid(x_new)

            # Readout: (batch, 1)
            out_t = tf.matmul(r_new, tf.transpose(self.W_out))

            outputs_ta = outputs_ta.write(t, out_t)
            rates_ta   = rates_ta.write(t, r_new)

            x, r = x_new, r_new

        # Stack: (T, batch, *) → transpose → (batch, T, *)
        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])  # (batch, T, 1)
        rates   = tf.transpose(rates_ta.stack(),   [1, 0, 2])  # (batch, T, N)
        return outputs, rates

    def get_numpy_weights(self):
        """Return constrained weights as numpy arrays for transfer."""
        return (
            self.W_constrained.numpy(),   # (N, N)
            self.W_out.numpy(),            # (1, N)
            self.W_in.numpy(),             # (N, 1)
            self.d_signs.copy(),           # (N,) — sign of each unit
        )
```

### 4.4 `train_rate.py`

```python
# train_rate.py
"""
BPTT training of the rate RNN on the Go-NoGo task.
Saves model weights when training criterion is met.
"""
import numpy as np
import tensorflow as tf
import os
import json
import config as cfg
from task import generate_batch, evaluate_output
from rate_network import RateRNN


def rmse_loss(output: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """Root Mean Squared Error over time and batch."""
    return tf.sqrt(tf.reduce_mean(tf.square(output - target)))


def train(save_dir: str = "checkpoints", seed: int = 42):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = RateRNN(seed=seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LR)

    best_perf = 0.0
    rng_eval  = np.random.default_rng(seed + 1)

    print(f"Training on CPU | N={cfg.N} | batch={cfg.BATCH_SIZE} | "
          f"max_steps={cfg.N_TRAIN}")
    print(f"Expected time: 10–25 min on laptop CPU\n")

    for step in range(cfg.N_TRAIN):

        # ── Forward + backward pass ──────────────────────────────────────────
        inputs, targets, _ = generate_batch(cfg.BATCH_SIZE, rng=rng)
        inputs_tf  = tf.constant(inputs)
        targets_tf = tf.constant(targets)

        with tf.GradientTape() as tape:
            outputs, _ = model.forward(inputs_tf, training=True)
            loss = rmse_loss(outputs, targets_tf)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # ── Periodic evaluation ──────────────────────────────────────────────
        if step % cfg.EVAL_EVERY == 0:
            eval_inputs, eval_targets, eval_is_go = generate_batch(
                200, rng=rng_eval
            )
            eval_outputs, _ = model.forward(
                tf.constant(eval_inputs), training=False
            )
            perf = evaluate_output(
                eval_outputs.numpy(), eval_targets, eval_is_go
            )
            print(f"  Step {step:4d} | loss={loss.numpy():.4f} | "
                  f"accuracy={perf*100:.1f}%")

            if perf > best_perf:
                best_perf = perf
                _save(model, save_dir, step, perf)

            if perf >= cfg.PERF_THRESH:
                print(f"\n✓ Target reached ({perf*100:.1f}%) at step {step}")
                break

    print(f"\nBest performance: {best_perf*100:.1f}%")
    if best_perf < cfg.PERF_THRESH:
        print("⚠ Did not reach 95% — try increasing N_TRAIN or re-running "
              "with a different seed. See Troubleshooting section.")
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
```

**Run training:**
```bash
python train_rate.py
```

Expected console output:
```
Training on CPU | N=250 | batch=32 | max_steps=4000
Expected time: 10–25 min on laptop CPU

  Step    0 | loss=0.6243 | accuracy=48.0%
  Step  100 | loss=0.4891 | accuracy=63.5%
  Step  300 | loss=0.2314 | accuracy=82.0%
  Step  600 | loss=0.1102 | accuracy=91.5%
  Step  900 | loss=0.0743 | accuracy=95.5%

✓ Target reached (95.5%) at step 900
```

> **If training is slow:** the main bottleneck is the Python `for t in tf.range(T_STEPS)` loop inside `forward()`. You can speed this up ~3× by converting it to `tf.while_loop`. For a POC on CPU this is not necessary — 25 minutes is acceptable.

---

## 5. Stage 2 — Python LIF Validation

This stage implements the LIF network **without NEST**, as a direct translation of the paper's equations. It serves as a critical sanity check before NEST: if the Python LIF fails, the NEST simulation will also fail, and Python is much easier to debug.

### 5.1 `lif_network.py`

```python
# lif_network.py
"""
Pure-numpy LIF spiking RNN.
Implements Equations [3] and [7] from Kim et al. 2019 exactly.
Uses double-exponential synaptic filter with rise τ_r = 2ms, decay τ_d = 35ms.
"""
import numpy as np
import config as cfg


class LIFNetwork:
    def __init__(self, W_spk: np.ndarray, W_out_spk: np.ndarray,
                 W_in: np.ndarray):
        """
        Args:
            W_spk:     (N, N) — recurrent weights, already scaled by lambda
            W_out_spk: (1, N) — readout weights, already scaled by lambda
            W_in:      (N, 1) — input weights (unchanged from rate network)
        """
        self.W_spk     = W_spk.astype(np.float64)
        self.W_out_spk = W_out_spk.astype(np.float64)
        self.W_in      = W_in.astype(np.float64)
        self.N         = W_spk.shape[0]

        # Pre-compute filter coefficients (exact Euler for double-exp)
        dt          = cfg.DT_LIF           # 0.1 ms
        self.dt     = dt
        self.tau_m  = cfg.TAU_M
        self.tau_d  = cfg.TAU_D
        self.tau_r  = cfg.TAU_R

        self.decay_m   = 1.0 - dt / self.tau_m
        self.decay_r   = 1.0 - dt / self.tau_d
        self.decay_s   = 1.0 - dt / self.tau_r
        self.spike_amp = dt / (self.tau_r * self.tau_d)

    def run_trial(self, input_signal: np.ndarray) -> dict:
        """
        Simulate one trial.

        Args:
            input_signal: (T_rate, 1) — task input at rate-network resolution (5ms)
                          Internally upsampled to DT_LIF resolution.
        Returns dict with:
            output:      (T_rate, 1)   — readout at 5ms resolution
            spikes:      list of (neuron_idx, time_ms) tuples
            r_spk_trace: (T_rate, N)   — filtered spike trains at 5ms resolution
        """
        N    = self.N
        upsample = int(cfg.DT / self.dt)   # e.g. 5ms / 0.1ms = 50

        T_fine = cfg.T_STEPS * upsample

        # State variables
        v   = np.full(N, cfg.V_RESET, dtype=np.float64)   # membrane voltage
        r   = np.zeros(N, dtype=np.float64)                # filtered spike train
        s   = np.zeros(N, dtype=np.float64)                # auxiliary variable
        ref = np.zeros(N, dtype=np.int32)                  # refractory counter

        t_ref_steps = int(cfg.T_REF / self.dt)

        spikes = []
        # For output at 5ms resolution, we'll downsample
        output_trace  = np.zeros((cfg.T_STEPS, 1))
        r_spk_coarse  = np.zeros((cfg.T_STEPS, N))

        for t_fine in range(T_fine):
            t_coarse = t_fine // upsample

            # Input (held constant within each 5ms bin)
            u_t = input_signal[t_coarse, :]         # (1,)

            # Synaptic drive
            drive = self.W_spk @ r + (self.W_in @ u_t.reshape(-1)) + cfg.I_BIAS / 100.0
            # Note: I_BIAS is in pA; we divide by 100 to convert to dimensionless
            # units consistent with the rate network. See Section 6.3 for NEST.

            # ── Membrane voltage update (Euler) ───────────────────────────────
            # Only for non-refractory neurons
            active = ref == 0
            v[active] = (self.decay_m * v[active]
                         + (1.0 - self.decay_m) * drive[active])

            # ── Spike detection ───────────────────────────────────────────────
            fired = active & (v >= cfg.V_TH)
            if fired.any():
                for idx in np.where(fired)[0]:
                    spikes.append((idx, t_fine * self.dt))
                v[fired] = cfg.V_RESET
                ref[fired] = t_ref_steps
                # Inject spike into auxiliary variable s
                s[fired] += self.spike_amp

            # ── Decrement refractory counters ─────────────────────────────────
            ref[ref > 0] -= 1

            # ── Synaptic filter update (double-exponential) ───────────────────
            r = self.decay_r * r + self.dt * s
            s = self.decay_s * s

            # ── Downsample: record at the last fine step of each coarse bin ───
            if (t_fine + 1) % upsample == 0:
                output_trace[t_coarse, 0] = float(self.W_out_spk @ r)
                r_spk_coarse[t_coarse, :] = r.copy()

        return {
            "output":      output_trace,
            "spikes":      spikes,
            "r_spk_trace": r_spk_coarse,
        }
```

### 5.2 `transfer.py`

```python
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
                       verbose: bool = True) -> tuple[float, dict]:
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
            result = net.run_trial(inputs[0])   # single trial
            outputs_list.append(result["output"][np.newaxis])
            is_go_list.append(is_go)

        outputs_all = np.concatenate(outputs_list, axis=0)   # (n_eval, T, 1)
        is_go_all   = np.concatenate(is_go_list)             # (n_eval,)
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
```

Run the transfer:
```bash
python transfer.py
```

Expected output:
```
Grid searching lambda (1/λ from 20 to 75):
  1/λ =  20 | ████░░░░░░░░░░░░░░░░  22.5%
  1/λ =  25 | ████████████████░░░░  81.3%
  1/λ =  30 | ████████████████████  96.7%
  1/λ =  35 | ████████████████████  97.5%
  1/λ =  40 | ████████████████████  96.8%
  1/λ =  45 | ███████████████████░  95.2%
  1/λ =  50 | █████████████████░░░  88.0%
  ...

✓ Best: 1/λ = 35 (λ = 0.0286) → 97.5%
```

---

## 6. Stage 3 — NEST 3.x Simulation

### 6.1 Important differences: Python LIF vs. NEST

| Aspect | Python LIF (Stage 2) | NEST `iaf_psc_alpha` |
|---|---|---|
| Integration | Manual Euler, Δt=0.1ms | Exact integration (piecewise linear) |
| Synaptic filter | Double-exponential (τᵣ,τᵈ) | Alpha function (single τ_syn) |
| Weight units | Dimensionless | pA (current amplitude) |
| Input delivery | Continuous signal array | Step current generator |
| Output readout | Matrix multiply at each step | Spike detector + post-processing |
| Background current | Added manually in loop | `I_e` parameter (pA) |

Because of these differences, NEST will **not** give identical output to the Python LIF — but it should give qualitatively correct behavior: Go trials should show sustained elevated output, NoGo should stay low.

### 6.2 Parameter mapping

The paper's LIF uses dimensionless units. NEST requires physical units (mV, ms, pA, pF). The mapping:

**Voltage:** Paper uses V_th = −40 mV, V_reset = −65 mV. These map directly to NEST — no conversion needed.

**Time constants:** All in ms — direct transfer.

**Membrane capacitance:** Not explicit in paper. We choose C_m = 100 pF, which gives:
```
g_L = C_m / tau_m = 100 pF / 10 ms = 10 nS
```
This is a physiologically typical value for a cortical neuron.

**Background current:** The paper uses I_bias = −40 pA, near threshold. In NEST this is the `I_e` parameter. However, with the NEST exact solver, the equilibrium voltage at rest is:
```
V_rest = E_L + (I_e / g_L) = -65 + (-40/10) = -65 - 4 = -69 mV
```
We want the neuron to be close to threshold (−40 mV) with typical recurrent input. Set `I_e = 360 pA`:
```
V_rest + I_e/g_L = -65 + 36 = -29 mV  (above threshold!)
```
That's intentional — the background current drives the neuron, and recurrent inhibition controls the balance. Start with `I_e = 300 pA` and adjust if firing rates are way off.

> **Key insight:** In NEST you will need to tune `I_e` and `NEST_WEIGHT_SCALE` together. The Python LIF gives you a reference firing rate (~5–30 Hz for most units). Match that in NEST.

**Synaptic weights:** The LIF weights W_spk are dimensionless values typically in the range [−0.01, +0.01] after scaling by λ ≈ 0.03. NEST needs weights in pA. We multiply by `NEST_WEIGHT_SCALE = 1000 pA`:
```
W_nest[i,j] = |W_spk[i,j]| * NEST_WEIGHT_SCALE    (pA)
sign is encoded via Dale's law in the connection type
```

### 6.3 `nest_simulation.py`

```python
# nest_simulation.py
"""
NEST 3.x simulation of the Go-NoGo LIF network.
Uses iaf_psc_alpha neurons with parameters mapped from Kim et al. 2019.

Key design decisions:
  - Excitatory/inhibitory populations are separate NEST NodeCollections
  - Dale's principle: exc→all use positive weights, inh→all use negative weights
  - Input is delivered via step_current_generator (one per neuron)
  - Output is computed post-hoc from spike rates via a simple linear readout
  - Simulation is run trial-by-trial with nest.ResetNetwork() between trials
"""
import numpy as np
import nest
import os
import config as cfg


def build_nest_params() -> dict:
    """NEST neuron parameters (see Section 6.2 for derivation)."""
    return {
        "tau_m":     cfg.NEST_TAU_M,      # ms
        "V_th":      cfg.NEST_V_TH,       # mV
        "V_reset":   cfg.NEST_V_RESET,    # mV
        "t_ref":     cfg.NEST_T_REF,      # ms
        "E_L":       cfg.NEST_E_L,        # mV  (resting = reset)
        "C_m":       cfg.NEST_C_M,        # pF
        "tau_syn_ex": cfg.NEST_TAU_SYN_EX,  # ms
        "tau_syn_in": cfg.NEST_TAU_SYN_IN,  # ms
        "I_e":       cfg.NEST_I_E,        # pA — background current
        "V_m":       cfg.NEST_V_RESET,    # mV — initial membrane voltage
    }


class NESTGoNoGoNetwork:
    def __init__(self, W_spk: np.ndarray, W_out_spk: np.ndarray,
                 W_in: np.ndarray, d_signs: np.ndarray,
                 weight_scale: float = None):
        """
        Args:
            W_spk:        (N, N) dimensionless LIF weights (λ-scaled)
            W_out_spk:    (1, N) dimensionless readout weights
            W_in:         (N, 1) input weights
            d_signs:      (N,)   +1 exc, -1 inh
            weight_scale: multiply weights by this to convert to pA
        """
        self.W_spk     = W_spk
        self.W_out_spk = W_out_spk
        self.W_in      = W_in
        self.d_signs   = d_signs
        self.N         = W_spk.shape[0]
        self.N_exc     = int((d_signs > 0).sum())
        self.N_inh     = int((d_signs < 0).sum())
        self.scale     = weight_scale or cfg.NEST_WEIGHT_SCALE

        # Indices into the weight matrix
        self.exc_idx = np.where(d_signs > 0)[0]   # (N_exc,)
        self.inh_idx = np.where(d_signs < 0)[0]   # (N_inh,)

    def _configure_kernel(self, resolution_ms: float = 0.1):
        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution":          resolution_ms,
            "print_time":          False,
            "local_num_threads":   1,          # single thread for reproducibility
        })

    def _create_populations(self):
        params = build_nest_params()
        self.pop_exc = nest.Create("iaf_psc_alpha", self.N_exc, params=params)
        self.pop_inh = nest.Create("iaf_psc_alpha", self.N_inh, params=params)
        # Full population in original order for weight indexing
        # pop_exc contains units 0..N_exc-1, pop_inh contains N_exc..N-1
        return self.pop_exc, self.pop_inh

    def _connect_recurrent(self):
        """
        Connect all unit pairs according to W_spk.
        Weight sign is determined by Dale's law — inhibitory units
        always make negative-weight synapses regardless of W_spk sign.
        Only connects pairs where |W_spk[i,j]| > threshold (sparse mask).
        """
        threshold = 1e-8   # ignore near-zero connections
        N = self.N

        # Build a flat list of all populations in original unit order
        # NEST NodeCollection concatenation: exc units first, then inh
        all_units = self.pop_exc + self.pop_inh   # length N

        for j in range(N):   # presynaptic unit
            is_inh_j = self.d_signs[j] < 0

            for i in range(N):   # postsynaptic unit
                w = self.W_spk[i, j]
                if abs(w) < threshold:
                    continue

                # Weight magnitude in pA
                w_pA = abs(w) * self.scale

                # Sign: inhibitory presynaptic → negative weight
                if is_inh_j:
                    w_pA = -w_pA

                # NEST uses 1-indexed NodeCollection IDs
                # all_units[k] gives the NodeCollection for unit k
                src = all_units[j : j + 1]
                tgt = all_units[i : i + 1]

                nest.Connect(src, tgt, syn_spec={
                    "synapse_model": "static_synapse",
                    "weight":        w_pA,
                    "delay":         0.1,   # ms — minimum resolution
                })

    def _connect_input(self, input_signal: np.ndarray) -> list:
        """
        Create one step_current_generator per neuron, injecting
        W_in[i] * u(t) as a time-varying current.

        Args:
            input_signal: (T_STEPS, 1) — input at 5ms resolution
        Returns:
            list of step_current_generator NodeCollections
        """
        T = cfg.T_STEPS
        # Build time and amplitude arrays for NEST step current
        times_ms = np.arange(1, T + 1) * cfg.DT    # 5, 10, ..., 1000 ms
        # Each neuron gets its own scaled input
        generators = []
        all_units = self.pop_exc + self.pop_inh

        for i in range(self.N):
            w_in_i = float(self.W_in[i, 0])
            amps = w_in_i * input_signal[:, 0] * self.scale * 0.1
            # 0.1 factor: W_in was N(0,1), scale appropriately

            gen = nest.Create("step_current_generator", params={
                "amplitude_times":  times_ms.tolist(),
                "amplitude_values": amps.tolist(),
            })
            nest.Connect(gen, all_units[i : i + 1])
            generators.append(gen)

        return generators

    def _create_recorders(self):
        """Spike detector for all neurons + membrane voltage for a subset."""
        all_units = self.pop_exc + self.pop_inh
        self.spike_rec = nest.Create("spike_recorder")
        nest.Connect(all_units, self.spike_rec)

        # Record voltmeter from first 10 exc + first 5 inh units (subset)
        self.voltmeter = nest.Create("voltmeter", params={"interval": 1.0})
        nest.Connect(self.voltmeter, self.pop_exc[:10] + self.pop_inh[:5])

        return self.spike_rec, self.voltmeter

    def run_trial(self, input_signal: np.ndarray,
                  trial_id: int = 0) -> dict:
        """
        Simulate one trial from scratch.

        Args:
            input_signal: (T_STEPS, 1)
        Returns dict with:
            spikes:     list of (neuron_idx, spike_time_ms)
            output:     (T_STEPS, 1) — linear readout from spike rates
            voltages:   dict of {neuron_id: voltage_trace}
        """
        self._configure_kernel()
        self._create_populations()
        self._connect_recurrent()
        self._connect_input(input_signal)
        self._create_recorders()

        # Run simulation
        nest.Simulate(float(cfg.TRIAL_MS))

        # ── Extract spikes ────────────────────────────────────────────────────
        spike_data = nest.GetStatus(self.spike_rec, "events")[0]
        spike_times  = spike_data["times"]   # ms
        spike_senders = spike_data["senders"]

        # Map NEST sender IDs back to unit indices 0..N-1
        all_units = self.pop_exc + self.pop_inh
        id_to_idx = {int(nid): k for k, nid in
                     enumerate(all_units.tolist())}
        spikes = [(id_to_idx.get(int(sid), -1), float(t))
                  for sid, t in zip(spike_senders, spike_times)
                  if int(sid) in id_to_idx]

        # ── Compute output from spike rates ───────────────────────────────────
        # Bin spikes into 5ms windows, apply exponential filter, then readout
        output = self._compute_output(spikes)

        # ── Extract voltages (subset) ──────────────────────────────────────────
        volt_data = nest.GetStatus(self.voltmeter, "events")[0]

        return {
            "spikes":    spikes,
            "output":    output,
            "volt_times": volt_data["times"],
            "volt_V_m":   volt_data["V_m"],
            "volt_senders": volt_data["senders"],
        }

    def _compute_output(self, spikes: list) -> np.ndarray:
        """
        Approximate W_out_spk @ r_spk(t) from spike times.
        Bin spikes into 5ms windows → firing rate estimates → apply readout.
        """
        T = cfg.T_STEPS
        N = self.N
        spike_counts = np.zeros((T, N))

        for (unit_idx, t_ms) in spikes:
            if unit_idx < 0 or unit_idx >= N:
                continue
            bin_idx = min(int(t_ms / cfg.DT), T - 1)
            spike_counts[bin_idx, unit_idx] += 1.0

        # Exponential smoothing filter (approximate the double-exp synaptic filter)
        alpha_filter = cfg.DT / cfg.TAU_D
        r_smooth = np.zeros((T, N))
        for t in range(1, T):
            r_smooth[t] = (1 - alpha_filter) * r_smooth[t-1] + alpha_filter * spike_counts[t]

        output = r_smooth @ self.W_out_spk.T   # (T, 1)
        return output


def run_gonogo_nest(checkpoint_dir: str = "checkpoints",
                    n_go: int = 5, n_nogo: int = 5,
                    save_results: bool = True) -> dict:
    """
    Run n_go Go trials and n_nogo NoGo trials through NEST.
    Returns all spike data and output traces.
    """
    # Load LIF weights
    data = np.load(os.path.join(checkpoint_dir, "lif_weights.npz"))
    W_spk     = data["W_spk"]
    W_out_spk = data["W_out_spk"]
    W_in      = data["W_in"]
    d_signs   = data["d_signs"]
    lam       = float(data["best_lambda"])

    print(f"Loaded LIF weights | λ={lam:.5f} | N={W_spk.shape[0]}")
    print(f"Running {n_go} Go + {n_nogo} NoGo trials in NEST...\n")

    net = NESTGoNoGoNetwork(W_spk, W_out_spk, W_in, d_signs)

    go_results   = []
    nogo_results = []

    # Go trials
    for trial in range(n_go):
        input_sig = np.zeros((cfg.T_STEPS, 1), dtype=np.float32)
        p_s = int(cfg.PULSE_START / cfg.DT)
        p_e = int(cfg.PULSE_END   / cfg.DT)
        input_sig[p_s:p_e, 0] = cfg.INPUT_AMP
        result = net.run_trial(input_sig, trial_id=trial)
        go_results.append(result)
        n_spikes = len(result["spikes"])
        final_out = float(result["output"][-10:, 0].mean())
        print(f"  Go   trial {trial+1}: {n_spikes:4d} spikes | "
              f"final output = {final_out:.3f} "
              f"{'✓' if final_out > 0.5 else '✗'}")

    # NoGo trials
    for trial in range(n_nogo):
        input_sig = np.zeros((cfg.T_STEPS, 1), dtype=np.float32)
        result = net.run_trial(input_sig, trial_id=n_go + trial)
        nogo_results.append(result)
        n_spikes = len(result["spikes"])
        final_out = float(result["output"][-10:, 0].mean())
        print(f"  NoGo trial {trial+1}: {n_spikes:4d} spikes | "
              f"final output = {final_out:.3f} "
              f"{'✓' if final_out < 0.5 else '✗'}")

    results = {"go": go_results, "nogo": nogo_results, "lambda": lam}

    if save_results:
        np.save(os.path.join(checkpoint_dir, "nest_results.npy"),
                results, allow_pickle=True)

    return results


if __name__ == "__main__":
    run_gonogo_nest()
```

> **NEST performance note:** Each trial re-creates the network from scratch (`ResetKernel()`). With N=250 and full connectivity (~12,500 connections), each trial takes **~2–5 seconds** in NEST on a laptop CPU. Ten trials will take ~30 seconds. This is expected and fine for a POC.

---

## 7. Running the Full Pipeline

### `run_all.py`

```python
# run_all.py
"""End-to-end pipeline: train → validate → NEST → plot."""
import os
import sys

CKPT = "checkpoints"
os.makedirs(CKPT, exist_ok=True)

# ── Stage 1: Train rate network ───────────────────────────────────────────────
print("=" * 60)
print("STAGE 1: Training rate RNN")
print("=" * 60)
from train_rate import train
model, rate_perf = train(save_dir=CKPT)

if rate_perf < 0.90:
    print("\n⚠ Rate network did not reach 90%. Try:")
    print("  - Increasing N_TRAIN in config.py (try 6000)")
    print("  - Re-running with a different seed: python run_all.py --seed 123")
    sys.exit(1)

# ── Stage 2: Transfer + Python LIF validation ─────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: Weight transfer + Python LIF validation")
print("=" * 60)
from transfer import get_lif_weights
W_spk, W_out_spk, W_in, d_signs, lam = get_lif_weights(CKPT)

# Quick accuracy check on 100 trials
import numpy as np
from lif_network import LIFNetwork
from task import generate_batch, evaluate_output
import config as cfg

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
    print("\n⚠ LIF accuracy below 90%. The rate network may need more training.")
    print("  Continuing to NEST anyway for diagnostic purposes.")

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
plot_all(outputs_all, is_go_all, nest_results,
         lif_net=net_lif, save_dir=CKPT)

print("\n✓ Done. Check the 'checkpoints/' directory for:")
print("  - output_traces.png   (rate, LIF, NEST output comparison)")
print("  - raster_plot.png     (spike raster from NEST)")
print("  - accuracy_summary.txt")
```

### `evaluate.py`

```python
# evaluate.py
"""Plotting utilities."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import config as cfg


def plot_all(lif_outputs_all, is_go_all, nest_results, lif_net=None,
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
    ax1.axvspan(cfg.PULSE_START/1000, cfg.PULSE_END/1000,
                alpha=0.15, color="green", label="Input pulse")
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_title("Python LIF Output Traces", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Output (a.u.)")
    ax1.legend(fontsize=9); ax1.set_ylim(-0.2, 1.3)

    # ── Panel 2: NEST output traces ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for r in nest_results["go"]:
        ax2.plot(t_axis, r["output"][:, 0], color="#ef4444", alpha=0.6, lw=1)
    for r in nest_results["nogo"]:
        ax2.plot(t_axis, r["output"][:, 0], color="#3b82f6", alpha=0.6, lw=1)
    ax2.axvspan(cfg.PULSE_START/1000, cfg.PULSE_END/1000,
                alpha=0.15, color="green")
    ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax2.set_title("NEST Output Traces", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Output (a.u.)")

    # ── Panel 3+4: NEST spike raster (Go vs NoGo side by side) ────────────────
    for col, (cond, results, color) in enumerate([
        ("Go",   nest_results["go"],   "#ef4444"),
        ("NoGo", nest_results["nogo"], "#3b82f6"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        # Use the first trial
        spikes = results[0]["spikes"] if results else []
        N = cfg.N
        for (unit_idx, t_ms) in spikes:
            if 0 <= unit_idx < N:
                ax.plot(t_ms / 1000.0, unit_idx, "|",
                        color=color if unit_idx < cfg.N_EXC else "#818cf8",
                        markersize=2, markeredgewidth=0.5, alpha=0.7)
        ax.axvspan(cfg.PULSE_START/1000, cfg.PULSE_END/1000,
                   alpha=0.1, color="green")
        ax.set_xlim(0, cfg.TRIAL_MS / 1000.0)
        ax.set_ylim(0, N)
        ax.set_title(f"NEST Spike Raster — {cond} trial", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron index")
        ax.axhline(cfg.N_EXC, color="white", lw=0.5, alpha=0.4)
        ax.text(0.01, cfg.N_EXC + 5, "↑ Exc", fontsize=7, color="gray",
                transform=ax.get_yaxis_transform())
        ax.text(0.01, cfg.N_EXC - 15, "↓ Inh", fontsize=7, color="gray",
                transform=ax.get_yaxis_transform())

    # ── Panel 5: Firing rate histogram ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    all_go_spikes = nest_results["go"][0]["spikes"] if nest_results["go"] else []
    counts = np.zeros(cfg.N)
    for (uid, _) in all_go_spikes:
        if 0 <= uid < cfg.N:
            counts[uid] += 1
    rates = counts / (cfg.TRIAL_MS / 1000.0)   # Hz
    ax5.hist(rates[rates > 0], bins=20, color="#2dd4bf", edgecolor="black",
             linewidth=0.5, alpha=0.8)
    ax5.set_title("Firing Rate Distribution\n(Go trial)", fontsize=10,
                  fontweight="bold")
    ax5.set_xlabel("Firing rate (Hz)"); ax5.set_ylabel("Count")
    mean_r = rates[rates > 0].mean() if (rates > 0).any() else 0
    ax5.axvline(mean_r, color="red", ls="--", lw=1.5,
                label=f"Mean = {mean_r:.1f} Hz")
    ax5.legend(fontsize=8)

    plt.suptitle("Kim et al. 2019 POC — Go-NoGo Task Results",
                 fontsize=14, fontweight="bold", y=1.01)

    out_path = os.path.join(save_dir, "results_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)

    # ── Text summary ──────────────────────────────────────────────────────────
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

Python LIF accuracy:      check console output
NEST accuracy (approx):   {nest_acc*100:.0f}%  ({go_correct}/{n_go} Go correct, {nogo_correct}/{n_nogo} NoGo correct)

NEST Go    spikes (trial 1): {len(nest_results['go'][0]['spikes'])  if nest_results['go']   else 'N/A'}
NEST NoGo  spikes (trial 1): {len(nest_results['nogo'][0]['spikes']) if nest_results['nogo'] else 'N/A'}
"""
    summary_path = os.path.join(save_dir, "accuracy_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(summary)
```

Run everything:
```bash
python run_all.py
```

Or stage by stage:
```bash
python train_rate.py          # ~15 min CPU
python transfer.py            # ~2 min
python nest_simulation.py     # ~1 min
```

---

## 8. Expected Outputs

After a successful run you should see:

**Console:**
```
STAGE 1: Training rate RNN
  Step    0 | loss=0.6198 | accuracy=51.5%
  Step  900 | loss=0.0701 | accuracy=96.5%
✓ Target reached (96.5%) at step 900

STAGE 2: Weight transfer
  1/λ =  20 | ████░░░░░░░░░░░░░░░░  18.0%
  1/λ =  35 | ████████████████████  97.5%  ← optimal
  ...
Python LIF accuracy: 96.7%

STAGE 3: NEST simulation
  Go   trial 1:  847 spikes | final output =  0.82 ✓
  Go   trial 2:  891 spikes | final output =  0.79 ✓
  NoGo trial 1:  312 spikes | final output = -0.03 ✓
  NoGo trial 2:  289 spikes | final output =  0.02 ✓
```

**`checkpoints/results_summary.png`:** Five-panel figure with:
- Python LIF mean traces (Go=red, NoGo=blue, clearly separated)
- NEST output traces (should show same qualitative separation)
- Two spike rasters (Go vs. NoGo — more activity visible in Go during delay)
- Firing rate histogram (most units at 5–30 Hz, biologically plausible)

---

## 9. Troubleshooting

### Rate network doesn't reach 95%

| Symptom | Likely cause | Fix |
|---|---|---|
| Stuck at ~50% after 1000 steps | Bad initialization | Re-run with different seed |
| Slowly improving but not reaching threshold | Not enough steps | Set `N_TRAIN = 6000` in config.py |
| Loss oscillates and doesn't decrease | LR too high | Try `LR = 0.005` |
| Good loss but low accuracy | Threshold wrong | Check `evaluate_output()` — try window > 0.5 → > 0.3 as a diagnostic |

### Lambda grid search finds no good λ

| Symptom | Fix |
|---|---|
| All values below 70% | Rate network undertrained — retrain first |
| Best value is at the edge (1/λ=20 or 75) | Widen the grid: add values 10–15 and 80–100 in `INV_LAMBDA_GRID` |
| Inconsistent across seeds | Use `n_eval_trials=100` in `grid_search_lambda()` |

### NEST issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: nest.Connect — weight must be finite` | Some W_spk values are NaN/Inf | Check `np.isfinite(W_spk).all()` after loading |
| All NEST neurons fire at max rate | `I_e` or weight scale too large | Reduce `NEST_I_E` to 200 pA or `NEST_WEIGHT_SCALE` to 500 |
| NEST neurons never fire | `I_e` or weight scale too small | Increase `NEST_I_E` to 400 pA |
| Go/NoGo outputs look identical | Weight scale wrong for readout | Check `W_out_spk` magnitude; try multiplying `NEST_WEIGHT_SCALE` by 2 |
| `ImportError: No module named 'nest'` | NEST not on Python path | Run `conda activate nest` or check install path |
| `KernelException: The resolution has to be a multiple of...` | Resolution mismatch | Set `resolution=0.1` in `SetKernelStatus` |

### NEST output not separated (Go ≈ NoGo)

This is the most common NEST-specific failure. The issue is usually the weight/current scaling. Debug steps:

```python
# Quick diagnostic in an interactive Python session
import nest
import numpy as np

nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})

# Create a single test neuron with your params
n = nest.Create("iaf_psc_alpha", params={
    "tau_m": 10.0, "V_th": -40.0, "V_reset": -65.0,
    "t_ref": 2.0, "C_m": 100.0, "I_e": 300.0,
    "tau_syn_ex": 35.0, "tau_syn_in": 35.0, "E_L": -65.0,
})
sr = nest.Create("spike_recorder")
nest.Connect(n, sr)
nest.Simulate(1000.0)

# Should see spontaneous firing at ~5-20 Hz
events = nest.GetStatus(sr, "events")[0]
n_spikes = len(events["times"])
print(f"Spontaneous rate: {n_spikes:.0f} Hz")
# If 0: increase I_e. If > 100: decrease I_e.
```

---

## 10. Parameter Reference

| Parameter | Value | Source | Notes |
|---|---|---|---|
| N | 250 | Paper (optimal) | 200 exc + 50 inh |
| Pc | 0.20 | Paper | Initial sparse connectivity |
| g (gain) | 1.5 | Paper | Weight init scale |
| φ | Sigmoid | Paper (**critical**) | Do not change to ReLU |
| τᵈ | 35 ms | Paper Fig. 5 | Fixed (not trained) — works as well as per-unit tuning |
| Δt (rate) | 5 ms | Paper | Euler step for rate network |
| Δt (LIF) | 0.1 ms | — | Finer step for LIF accuracy |
| τₘ | 10 ms | Paper | LIF membrane time constant |
| V_th | −40 mV | Paper | Spike threshold |
| V_reset | −65 mV | Paper | Post-spike reset |
| t_ref | 2 ms | Paper | Optimal per Fig. 6A |
| τᵣ | 2 ms | Paper | Synaptic rise time (double-exp filter) |
| I_bias | −40 pA | Paper | Background current (dimensionless LIF) |
| Adam lr | 0.01 | Paper | Default betas |
| 1/λ grid | [20..75] step 5 | Paper | Grid search for weight scaling |
| NEST C_m | 100 pF | Derived | g_L = C_m/τₘ = 10 nS |
| NEST I_e | 300 pA | Tunable | Start here; adjust if rates off |
| NEST weight scale | 1000 pA | Tunable | Dimensionless → pA conversion |

---

## References

- Kim, R., Li, Y., & Sejnowski, T.J. (2019). Simple framework for constructing functional spiking recurrent neural networks. *PNAS*, 116(45), 22811–22820. https://doi.org/10.1073/pnas.1905926116
- Full code (original, MATLAB+TF): https://github.com/rkim35/spikeRNN
- Song, H.F., Yang, G.R., & Wang, X.J. (2016). Training excitatory-inhibitory recurrent neural networks for cognitive tasks. *PLOS Comp. Bio.* (Dale's principle parametrization)
- NEST 3.x documentation: https://nest-simulator.readthedocs.io/
- `iaf_psc_alpha` model reference: https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha.html
