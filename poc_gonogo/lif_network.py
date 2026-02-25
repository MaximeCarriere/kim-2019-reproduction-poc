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

        dt          = cfg.DT_LIF           # 0.1 ms
        self.dt     = dt
        self.tau_m  = cfg.TAU_M
        self.tau_d  = cfg.TAU_D
        self.tau_r  = cfg.TAU_R

        self.decay_m   = 1.0 - dt / self.tau_m
        self.decay_r   = 1.0 - dt / self.tau_d
        self.decay_s   = 1.0 - dt / self.tau_r
        # Normalisation: spike_amp = 1/τ_r gives r_ss = R·τ_d for Poisson
        # at rate R (spk/ms), putting r_spk in the same [0–1+] range as
        # the rate-network's sigmoid output.  Each spike contributes τ_d
        # to ∫r dt, so r_spk ≈ r_rate/λ when firing at rate R·τ_d=r_rate/λ.
        self.spike_amp = 1.0 / self.tau_r

        # Dimensionless threshold = i_bias (fire only when net drive > 0).
        # Reset well below threshold so the refractory recovery is fast.
        i_bias = cfg.I_BIAS / 100.0       # -0.4
        self.v_th    = i_bias             # -0.4  (threshold = resting drive)
        self.v_reset = i_bias - 1.0      # -1.4

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
        upsample = int(cfg.DT / self.dt)   # 5ms / 0.1ms = 50
        T_fine = cfg.T_STEPS * upsample

        # State variables (use dimensionless reset)
        v   = np.full(N, self.v_reset, dtype=np.float64)
        r   = np.zeros(N, dtype=np.float64)
        s   = np.zeros(N, dtype=np.float64)
        ref = np.zeros(N, dtype=np.int32)

        t_ref_steps = int(cfg.T_REF / self.dt)

        spikes = []
        output_trace  = np.zeros((cfg.T_STEPS, 1))
        r_spk_coarse  = np.zeros((cfg.T_STEPS, N))

        # Bias scaled to the same dimensionless units as the drive
        i_bias = cfg.I_BIAS / 100.0  # -0.4  keeps neurons subthreshold at rest

        for t_fine in range(T_fine):
            t_coarse = t_fine // upsample
            u_t = input_signal[t_coarse, :]   # (1,)

            drive = self.W_spk @ r + (self.W_in[:, 0] * u_t[0]) + i_bias

            # Membrane voltage update — only non-refractory neurons
            active = ref == 0
            v[active] = self.decay_m * v[active] + (1.0 - self.decay_m) * drive[active]

            # Spike detection
            fired = active & (v >= self.v_th)
            if fired.any():
                for idx in np.where(fired)[0]:
                    spikes.append((idx, t_fine * self.dt))
                v[fired] = self.v_reset
                ref[fired] = t_ref_steps
                s[fired] += self.spike_amp

            ref[ref > 0] -= 1

            # Double-exponential synaptic filter
            r = self.decay_r * r + self.dt * s
            s = self.decay_s * s

            # Downsample: record at last fine step of each coarse bin
            if (t_fine + 1) % upsample == 0:
                output_trace[t_coarse, 0] = float(self.W_out_spk @ r)
                r_spk_coarse[t_coarse, :] = r.copy()

        return {
            "output":      output_trace,
            "spikes":      spikes,
            "r_spk_trace": r_spk_coarse,
        }
