# rate_network.py
"""
Continuous-variable rate RNN with Dale's principle and sparse connectivity.
Implemented in PyTorch (README used TF — same math, same architecture).
Forward pass is written explicitly so it can be traced by autograd.
"""
import numpy as np
import torch
import torch.nn as nn
import config as cfg


class RateRNN(nn.Module):
    def __init__(self, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        N, NE, NI = cfg.N, cfg.N_EXC, cfg.N_INH

        # ── Dale's sign vector (fixed) ────────────────────────────────────────
        d_signs = np.ones(N, dtype=np.float32)
        d_signs[NE:] = -1.0
        self.register_buffer('D', torch.tensor(d_signs[np.newaxis, :]))  # (1, N)
        self.d_signs = d_signs

        # ── Sparsity mask M (fixed) ───────────────────────────────────────────
        M = rng.random((N, N)) < cfg.PC
        np.fill_diagonal(M, False)
        self.register_buffer('M', torch.tensor(M.astype(np.float32)))    # (N, N)

        # ── Trainable recurrent weights W_raw (N, N) ─────────────────────────
        init_std = cfg.GAIN / np.sqrt(N * cfg.PC)
        W_init = rng.normal(0, init_std, (N, N)).astype(np.float32)
        self.W_raw = nn.Parameter(torch.tensor(W_init))

        # ── Trainable readout weights W_out (1, N) ────────────────────────────
        W_out_init = rng.normal(0, 1.0 / np.sqrt(N), (1, N)).astype(np.float32)
        self.W_out = nn.Parameter(torch.tensor(W_out_init))

        # ── Fixed input weights W_in (N, 1) ──────────────────────────────────
        W_in_init = rng.normal(0, 1.0, (N, 1)).astype(np.float32)
        self.register_buffer('W_in', torch.tensor(W_in_init))

        # ── Time constants ────────────────────────────────────────────────────
        self.alpha = 1.0 - cfg.DT / cfg.TAU_D
        self.beta  = cfg.DT / cfg.TAU_D

    @property
    def W_constrained(self) -> torch.Tensor:
        """
        Dale's constraint: W_constrained[i,j] = relu(W_raw[i,j]) * D[j] * M[i,j]
        Shape: (N, N)
        """
        return torch.relu(self.W_raw) * self.D * self.M

    def forward(self, inputs: torch.Tensor, training: bool = True):
        """
        Roll the RNN forward through a full trial.

        Args:
            inputs: (batch, T, 1)
        Returns:
            outputs:  (batch, T, 1)   — linear readout at each step
            r_trace:  (batch, T, N)   — firing rates at each step
        """
        batch = inputs.shape[0]
        W = self.W_constrained       # (N, N)

        # Initial state: x = 0, r = sigmoid(0) = 0.5
        x = torch.zeros(batch, cfg.N, device=inputs.device)
        r = torch.full((batch, cfg.N), 0.5, device=inputs.device)

        outputs_list = []
        rates_list   = []

        for t in range(cfg.T_STEPS):
            u_t = inputs[:, t, :]          # (batch, 1)

            # Recurrent + input drive
            rec = r @ W.t()                          # (batch, N)
            inp = u_t @ self.W_in.t()                # (batch, N)

            x_new = self.alpha * x + self.beta * (rec + inp)

            if training:
                x_new = x_new + torch.randn_like(x_new) * cfg.NOISE_STD

            r_new = torch.sigmoid(x_new)

            out_t = r_new @ self.W_out.t()           # (batch, 1)

            outputs_list.append(out_t.unsqueeze(1))  # (batch, 1, 1)
            rates_list.append(r_new.unsqueeze(1))    # (batch, 1, N)

            x, r = x_new, r_new

        outputs = torch.cat(outputs_list, dim=1)     # (batch, T, 1)
        rates   = torch.cat(rates_list,   dim=1)     # (batch, T, N)
        return outputs, rates

    def get_numpy_weights(self):
        """Return constrained weights as numpy arrays for transfer."""
        with torch.no_grad():
            return (
                self.W_constrained.cpu().numpy(),   # (N, N)
                self.W_out.cpu().numpy(),            # (1, N)
                self.W_in.cpu().numpy(),             # (N, 1)
                self.d_signs.copy(),                 # (N,)
            )
