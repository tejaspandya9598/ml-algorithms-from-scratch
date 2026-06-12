"""
Autoencoder (HW4) for unsupervised representation learning on financial data.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Autoencoder:
    """
    Feedforward autoencoder for non-linear dimensionality reduction.
    """

    def __init__(self, input_dim, hidden1, latent_dim, lr=0.001):
        # The bottleneck architecture forces the network to compress the data.
        # He initialization is used throughout because of the ReLU activations.

        # Encoder: compresses input features down to the latent dimension
        self.We1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.be1 = np.zeros((1, hidden1))
        self.We2 = np.random.randn(hidden1, latent_dim) * np.sqrt(2.0 / hidden1)
        self.be2 = np.zeros((1, latent_dim))

        # Decoder: attempts to reconstruct the original input from the latent space.
        self.Wd1 = np.random.randn(latent_dim, hidden1) * np.sqrt(2.0 / latent_dim)
        self.bd1 = np.zeros((1, hidden1))
        self.Wd2 = np.random.randn(hidden1, input_dim) * np.sqrt(2.0 / hidden1)
        self.bd2 = np.zeros((1, input_dim))

        self.lr = lr

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        # Encoder pass
        self.ze1 = X @ self.We1 + self.be1
        self.ae1 = self.relu(self.ze1)
        self.ze2 = self.ae1 @ self.We2 + self.be2
        self.ae2 = self.relu(self.ze2)

        # Decoder pass
        self.zd1 = self.ae2 @ self.Wd1 + self.bd1
        self.ad1 = self.relu(self.zd1)
        self.zd2 = self.ad1 @ self.Wd2 + self.bd2

        # Linear activation at the output because we are reconstructing continuous
        # features (like normalized returns or z-scores), not probabilities.
        self.output = self.zd2

        return self.output

    def backward(self, X):
        n = X.shape[0]

        # MSE gradient: the target is the input X itself (unsupervised learning)
        # Derivative of (output - X)^2 is 2*(output - X)
        d_out = 2 * (self.output - X) / n

        # Backprop through the decoder
        dWd2 = self.ad1.T @ d_out
        dbd2 = np.sum(d_out, axis=0, keepdims=True)

        dd1 = (d_out @ self.Wd2.T) * self.relu_deriv(self.zd1)
        dWd1 = self.ae2.T @ dd1
        dbd1 = np.sum(dd1, axis=0, keepdims=True)

        # Backprop through the encoder
        de2 = (dd1 @ self.Wd1.T) * self.relu_deriv(self.ze2)
        dWe2 = self.ae1.T @ de2
        dbe2 = np.sum(de2, axis=0, keepdims=True)

        de1 = (de2 @ self.We2.T) * self.relu_deriv(self.ze1)
        dWe1 = X.T @ de1
        dbe1 = np.sum(de1, axis=0, keepdims=True)

        # Parameter updates via standard SGD
        self.We1 -= self.lr * dWe1
        self.be1 -= self.lr * dbe1
        self.We2 -= self.lr * dWe2
        self.be2 -= self.lr * dbe2
        self.Wd1 -= self.lr * dWd1
        self.bd1 -= self.lr * dbd1
        self.Wd2 -= self.lr * dWd2
        self.bd2 -= self.lr * dbd2

    def train(self, X, epochs=50, batch_size=256, verbose=True):
        n = X.shape[0]
        losses = []

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]

                recon = self.forward(Xb)
                loss = np.mean((recon - Xb) ** 2)
                epoch_loss += loss
                n_batches += 1

                self.backward(Xb)

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} — MSE={avg_loss:.6f}")

        return losses

    def reconstruction_error(self, X):
        # calculates how badly the model failed to
        # reconstruct each specific sample.
        recon = self.forward(X)
        return np.mean((recon - X) ** 2, axis=1)
