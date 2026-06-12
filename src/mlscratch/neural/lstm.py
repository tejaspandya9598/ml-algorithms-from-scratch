"""
LSTM from scratch (HW4): a hand-written LSTM cell and sequence model, with a PyTorch
LSTM kept alongside as the reference implementation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LSTMCell:
    """Single LSTM cell with forget, input, output gates."""

    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier-style initialization so weights don't blow up over time.
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        # [x_t, h_{t-1}]
        concat_dim = input_dim + hidden_dim
        self.Wf = np.random.randn(concat_dim, hidden_dim) * scale
        self.bf = np.zeros((1, hidden_dim))

        self.Wi = np.random.randn(concat_dim, hidden_dim) * scale
        self.bi = np.zeros((1, hidden_dim))

        self.Wg = np.random.randn(concat_dim, hidden_dim) * scale
        self.bg = np.zeros((1, hidden_dim))

        self.Wo = np.random.randn(concat_dim, hidden_dim) * scale
        self.bo = np.zeros((1, hidden_dim))

    def sigmoid(self, x):
        # Clip to prevent np.exp() overflow which causes NaNs in deep time series
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        """One time step forward."""
        concat = np.concatenate([x, h_prev], axis=1)

        # The 4 core gates. Sigmoid (0 to 1) for gating, Tanh (-1 to 1) for values.
        f = self.sigmoid(
            concat @ self.Wf + self.bf
        )  # Forget: what past memory to erase
        i = self.sigmoid(
            concat @ self.Wi + self.bi
        )  # Input: how much new info to let in
        g = np.tanh(concat @ self.Wg + self.bg)  # Candidate: the actual new info
        o = self.sigmoid(
            concat @ self.Wo + self.bo
        )  # Output: how much memory to reveal

        # update long-term memory (c)
        c = f * c_prev + i * g

        # Calculate short-term memory / output (h)
        h = o * np.tanh(c)

        cache = (x, h_prev, c_prev, concat, f, i, g, o, c)
        return h, c, cache

    def backward(self, dh_next, dc_next, cache):
        """Backward through one time step."""
        x, h_prev, c_prev, concat, f, i, g, o, c = cache

        # Gradient flows into the cell state from BOTH the next cell state (dc_next)
        # and the current hidden state (dh_next)
        tanh_c = np.tanh(c)
        dc = dc_next + dh_next * o * (1 - tanh_c**2)

        # Route gradients through the specific gates
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        do = dh_next * tanh_c

        # Apply derivatives of the activation functions (sigmoid' = s*(1-s), tanh' = 1-t^2)
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dg_raw = dg * (1 - g**2)
        do_raw = do * o * (1 - o)

        # Compute weight gradients for this specific time step
        self.dWf = concat.T @ df_raw
        self.dbf = np.sum(df_raw, axis=0, keepdims=True)
        self.dWi = concat.T @ di_raw
        self.dbi = np.sum(di_raw, axis=0, keepdims=True)
        self.dWg = concat.T @ dg_raw
        self.dbg = np.sum(dg_raw, axis=0, keepdims=True)
        self.dWo = concat.T @ do_raw
        self.dbo = np.sum(do_raw, axis=0, keepdims=True)

        # Accumulate the gradient to pass back to the previous layer/time step
        d_concat = (
            df_raw @ self.Wf.T
            + di_raw @ self.Wi.T
            + dg_raw @ self.Wg.T
            + do_raw @ self.Wo.T
        )

        # Split the concatenated gradient back into dx (for lower layers) and dh (for previous time step)
        dx = d_concat[:, : self.input_dim]
        dh_prev = d_concat[:, self.input_dim :]
        dc_prev = dc * f

        return dx, dh_prev, dc_prev


class LSTMModel:
    """LSTM for sequence-to-one regression."""

    def __init__(self, input_dim, hidden_dim, lr=0.001, clip_val=5.0):
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_val = clip_val

        # Final linear layer to project the last hidden state to a single continuous prediction
        self.Wy = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.by = np.zeros((1, 1))

    def forward_seq(self, X_seq):
        """Forward through a full sequence. X_seq shape: (batch, seq_len, features)"""
        batch_size, seq_len, _ = X_seq.shape
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        self.caches = []

        for t in range(seq_len):
            x_t = X_seq[:, t, :]
            h, c, cache = self.cell.forward(x_t, h, c)
            self.caches.append(cache)

        y_pred = h @ self.Wy + self.by
        self.last_h = h
        return y_pred

    def backward_seq(self, y_true, y_pred, X_seq):
        """BPTT with gradient clipping."""
        batch_size = y_true.shape[0]
        seq_len = X_seq.shape[1]

        # Standard MSE derivative at the output node
        dy = 2 * (y_pred - y_true) / batch_size
        self.dWy = self.last_h.T @ dy
        self.dby = np.sum(dy, axis=0, keepdims=True)

        dh = dy @ self.Wy.T
        dc = np.zeros_like(dh)

        # Initialize containers to sum the gradients across all time steps
        dWf_total = np.zeros_like(self.cell.Wf)
        dbf_total = np.zeros_like(self.cell.bf)
        dWi_total = np.zeros_like(self.cell.Wi)
        dbi_total = np.zeros_like(self.cell.bi)
        dWg_total = np.zeros_like(self.cell.Wg)
        dbg_total = np.zeros_like(self.cell.bg)
        dWo_total = np.zeros_like(self.cell.Wo)
        dbo_total = np.zeros_like(self.cell.bo)

        # Backpropagation Through Time (BPTT): step backwards from t=N to t=0
        for t in reversed(range(seq_len)):
            dx, dh, dc = self.cell.backward(dh, dc, self.caches[t])
            dWf_total += self.cell.dWf
            dbf_total += self.cell.dbf
            dWi_total += self.cell.dWi
            dbi_total += self.cell.dbi
            dWg_total += self.cell.dWg
            dbg_total += self.cell.dbg
            dWo_total += self.cell.dWo
            dbo_total += self.cell.dbo

        # Gradient Clipping: If the L2 norm of all gradients
        # combined exceeds a threshold, scale them all down proportionally.
        all_grads = [
            dWf_total,
            dbf_total,
            dWi_total,
            dbi_total,
            dWg_total,
            dbg_total,
            dWo_total,
            dbo_total,
            self.dWy,
            self.dby,
        ]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))

        if total_norm > self.clip_val:
            scale = self.clip_val / total_norm
            for g in all_grads:
                g *= scale

        # Apply the accumulated, clipped updates
        self.cell.Wf -= self.lr * dWf_total
        self.cell.bf -= self.lr * dbf_total
        self.cell.Wi -= self.lr * dWi_total
        self.cell.bi -= self.lr * dbi_total
        self.cell.Wg -= self.lr * dWg_total
        self.cell.bg -= self.lr * dbg_total
        self.cell.Wo -= self.lr * dWo_total
        self.cell.bo -= self.lr * dbo_total
        self.Wy -= self.lr * self.dWy
        self.by -= self.lr * self.dby

    def train(self, X_seqs, y_targets, epochs=50, batch_size=32, verbose=True):
        n = len(X_seqs)
        losses = []
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuf = X_seqs[perm]
            y_shuf = y_targets[perm]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                y_pred = self.forward_seq(Xb)
                loss = np.mean((y_pred.flatten() - yb) ** 2)
                epoch_loss += loss
                n_batches += 1

                self.backward_seq(yb.reshape(-1, 1), y_pred, Xb)

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} — MSE={avg_loss:.6f}")
        return losses

    def predict(self, X_seqs):
        return self.forward_seq(X_seqs).flatten()


class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # batch_first=True is crucial: it expects inputs as (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Final projection layer to map the hidden state to a single price prediction
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 'out' holds the hidden states for all time steps. '_' holds the final cell state.
        out, _ = self.lstm(x)
        # we only want the output from the very last time step [-1]
        return self.fc(out[:, -1, :])
