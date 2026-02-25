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

# ── LIF parameters (Python simulation) ───────────────────────────────────────
TAU_M    = 10.0         # Membrane time constant (ms)
V_TH     = -40.0        # Spike threshold (mV)
V_RESET  = -65.0        # Reset potential (mV)
T_REF    = 2.0          # Absolute refractory period (ms)
TAU_R    = 2.0          # Synaptic rise time (ms) — double-exp filter
I_BIAS   = -40.0        # Constant background current (pA)
DT_LIF   = 0.1          # LIF Euler step (ms) — smaller for accuracy

# ── Lambda grid search ────────────────────────────────────────────────────────
INV_LAMBDA_GRID = list(range(1, 21))  # [1, 2, ..., 20]  r_spk ≈ r_rate/λ in [0,1] range

# ── NEST parameters ───────────────────────────────────────────────────────────
NEST_TAU_M      = TAU_M         # ms
NEST_V_TH       = V_TH          # mV
NEST_V_RESET    = V_RESET       # mV
NEST_T_REF      = T_REF         # ms
NEST_E_L        = V_RESET       # mV
NEST_C_M        = 100.0         # pF
NEST_TAU_SYN_EX = TAU_D         # ms
NEST_TAU_SYN_IN = TAU_D         # ms
NEST_I_E        = 250.0         # pA — threshold current: V_eq = E_L + (τ_m/C_m)*I_e = -40 = V_th
                                 # ≡ Python-LIF condition: resting drive = v_th (fires only on net exc)
NEST_WEIGHT_SCALE = 50.0        # pA per unit weight (tuned so Go ≈ 20–80 Hz, NoGo ≈ silent)
