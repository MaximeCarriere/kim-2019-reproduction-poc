# nest_simulation.py
"""
NEST 3.x simulation of the Go-NoGo LIF network.
Uses iaf_psc_alpha neurons with parameters mapped from Kim et al. 2019.
"""
import numpy as np
import nest
import os
import config as cfg


def build_nest_params() -> dict:
    return {
        "tau_m":      cfg.NEST_TAU_M,
        "V_th":       cfg.NEST_V_TH,
        "V_reset":    cfg.NEST_V_RESET,
        "t_ref":      cfg.NEST_T_REF,
        "E_L":        cfg.NEST_E_L,
        "C_m":        cfg.NEST_C_M,
        "tau_syn_ex": cfg.NEST_TAU_SYN_EX,
        "tau_syn_in": cfg.NEST_TAU_SYN_IN,
        "I_e":        cfg.NEST_I_E,
        "V_m":        cfg.NEST_V_RESET,
    }


class NESTGoNoGoNetwork:
    def __init__(self, W_spk: np.ndarray, W_out_spk: np.ndarray,
                 W_in: np.ndarray, d_signs: np.ndarray,
                 weight_scale: float = None):
        self.W_spk     = W_spk
        self.W_out_spk = W_out_spk
        self.W_in      = W_in
        self.d_signs   = d_signs
        self.N         = W_spk.shape[0]
        self.N_exc     = int((d_signs > 0).sum())
        self.N_inh     = int((d_signs < 0).sum())
        self.scale     = weight_scale or cfg.NEST_WEIGHT_SCALE

        self.exc_idx = np.where(d_signs > 0)[0]
        self.inh_idx = np.where(d_signs < 0)[0]

    def _configure_kernel(self, resolution_ms: float = 0.1):
        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution":        resolution_ms,
            "print_time":        False,
            "local_num_threads": 1,
        })

    def _create_populations(self):
        params = build_nest_params()
        self.pop_exc = nest.Create("iaf_psc_alpha", self.N_exc, params=params)
        self.pop_inh = nest.Create("iaf_psc_alpha", self.N_inh, params=params)

    def _connect_recurrent(self):
        """
        Connect all unit pairs according to W_spk using Dale's law.
        Build connection lists per (pre, syn_type) for efficiency.
        """
        threshold = 1e-8
        N = self.N
        all_units = self.pop_exc + self.pop_inh  # length N, units in original order

        # Collect connections as lists for batch Connect calls
        # Format: {(src_idx, is_inh): [(tgt_idx, weight_pA), ...]}
        exc_src_tgts = {}  # src_idx -> list of (tgt_idx, w_pA)
        inh_src_tgts = {}

        for j in range(N):
            is_inh_j = self.d_signs[j] < 0
            for i in range(N):
                w = self.W_spk[i, j]
                if abs(w) < threshold:
                    continue
                w_pA = abs(w) * self.scale
                if is_inh_j:
                    w_pA = -w_pA
                    inh_src_tgts.setdefault(j, []).append((i, w_pA))
                else:
                    exc_src_tgts.setdefault(j, []).append((i, w_pA))

        # Connect
        for j, connections in exc_src_tgts.items():
            src = all_units[j: j + 1]
            for i, w_pA in connections:
                tgt = all_units[i: i + 1]
                nest.Connect(src, tgt, syn_spec={
                    "synapse_model": "static_synapse",
                    "weight": float(w_pA),
                    "delay": 0.1,
                })

        for j, connections in inh_src_tgts.items():
            src = all_units[j: j + 1]
            for i, w_pA in connections:
                tgt = all_units[i: i + 1]
                nest.Connect(src, tgt, syn_spec={
                    "synapse_model": "static_synapse",
                    "weight": float(w_pA),
                    "delay": 0.1,
                })

    def _connect_input(self, input_signal: np.ndarray):
        """
        Create one step_current_generator per neuron.
        input_signal: (T_STEPS, 1)
        """
        T = cfg.T_STEPS
        # Times: the step current changes at these times (ms)
        # step_current_generator changes amplitude at the specified time
        times_ms = (np.arange(T) * cfg.DT + cfg.DT).tolist()  # [5, 10, ..., 1000]
        all_units = self.pop_exc + self.pop_inh

        for i in range(self.N):
            w_in_i = float(self.W_in[i, 0])
            # Scale input similarly to recurrent weights
            amps = (w_in_i * input_signal[:, 0] * self.scale * 0.1).tolist()

            gen = nest.Create("step_current_generator", params={
                "amplitude_times":  times_ms,
                "amplitude_values": amps,
            })
            nest.Connect(gen, all_units[i: i + 1])

    def _create_recorders(self):
        all_units = self.pop_exc + self.pop_inh
        self.spike_rec = nest.Create("spike_recorder")
        nest.Connect(all_units, self.spike_rec)

        self.voltmeter = nest.Create("voltmeter", params={"interval": 1.0})
        n_vm = min(10, self.N_exc)
        nest.Connect(self.voltmeter, self.pop_exc[:n_vm])

    def run_trial(self, input_signal: np.ndarray) -> dict:
        """
        Simulate one trial from scratch.
        input_signal: (T_STEPS, 1)
        """
        self._configure_kernel()
        self._create_populations()
        self._connect_recurrent()
        self._connect_input(input_signal)
        self._create_recorders()

        nest.Simulate(float(cfg.TRIAL_MS))

        # Extract spikes
        spike_data   = nest.GetStatus(self.spike_rec, "events")[0]
        spike_times  = spike_data["times"]
        spike_senders = spike_data["senders"]

        all_units = self.pop_exc + self.pop_inh
        id_to_idx = {int(nid): k for k, nid in enumerate(all_units.tolist())}
        spikes = [(id_to_idx[int(sid)], float(t))
                  for sid, t in zip(spike_senders, spike_times)
                  if int(sid) in id_to_idx]

        output = self._compute_output(spikes)

        volt_data = nest.GetStatus(self.voltmeter, "events")[0]

        return {
            "spikes":       spikes,
            "output":       output,
            "volt_times":   volt_data["times"],
            "volt_V_m":     volt_data["V_m"],
            "volt_senders": volt_data["senders"],
        }

    def _compute_output(self, spikes: list) -> np.ndarray:
        T = cfg.T_STEPS
        N = self.N
        spike_counts = np.zeros((T, N))

        for (unit_idx, t_ms) in spikes:
            if 0 <= unit_idx < N:
                bin_idx = min(int(t_ms / cfg.DT), T - 1)
                spike_counts[bin_idx, unit_idx] += 1.0

        # Match Python LIF normalisation: r_ss = R·τ_d (not R·DT).
        # Using r[t] = (1-α)·r[t-1] + spike_counts[t] gives
        # r_ss = spike_counts / α = R·DT / (DT/τ_d) = R·τ_d  ✓
        alpha_filter = cfg.DT / cfg.TAU_D
        r_smooth = np.zeros((T, N))
        for t in range(1, T):
            r_smooth[t] = ((1 - alpha_filter) * r_smooth[t - 1]
                           + spike_counts[t])

        output = r_smooth @ self.W_out_spk.T   # (T, 1)
        return output


def run_gonogo_nest(checkpoint_dir: str = "checkpoints",
                    n_go: int = 5, n_nogo: int = 5,
                    save_results: bool = True) -> dict:
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

    for trial in range(n_go):
        input_sig = np.zeros((cfg.T_STEPS, 1), dtype=np.float32)
        p_s = int(cfg.PULSE_START / cfg.DT)
        p_e = int(cfg.PULSE_END   / cfg.DT)
        input_sig[p_s:p_e, 0] = cfg.INPUT_AMP
        result = net.run_trial(input_sig)
        go_results.append(result)
        n_spikes  = len(result["spikes"])
        final_out = float(result["output"][-10:, 0].mean())
        print(f"  Go   trial {trial+1}: {n_spikes:4d} spikes | "
              f"output={final_out:.3f} {'✓' if final_out > 0.5 else '✗'}")

    for trial in range(n_nogo):
        input_sig = np.zeros((cfg.T_STEPS, 1), dtype=np.float32)
        result = net.run_trial(input_sig)
        nogo_results.append(result)
        n_spikes  = len(result["spikes"])
        final_out = float(result["output"][-10:, 0].mean())
        print(f"  NoGo trial {trial+1}: {n_spikes:4d} spikes | "
              f"output={final_out:.3f} {'✓' if final_out < 0.5 else '✗'}")

    results = {"go": go_results, "nogo": nogo_results, "lambda": lam}

    if save_results:
        np.save(os.path.join(checkpoint_dir, "nest_results.npy"),
                results, allow_pickle=True)

    return results


if __name__ == "__main__":
    run_gonogo_nest()
