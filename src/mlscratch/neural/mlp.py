"""
Multilayer perceptron from scratch (HW4): forward pass and backprop in pure NumPy.
"""
from __future__ import annotations

import numpy as np


def onehot(y, n_classes=2):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1.0
    return oh


class MLP:
    """2-hidden-layer feedforward network with ReLU + Softmax output."""

    def __init__(self, input_dim, hidden1, hidden2, output_dim, lr=0.01):
        self.lr = lr

        # He initialization: optimal for ReLU. Scales weights by sqrt(2/fan_in)
        # to prevent variance from vanishing or exploding across deep layers.
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, output_dim))

    def relu(self, z):
        # Non-linear activation: clips negative values to 0
        return np.maximum(0, z)

    def relu_deriv(self, z):
        # Derivative is 1 if z > 0, else 0 (ignoring the undefined point at exactly 0)
        return (z > 0).astype(float)

    def softmax(self, z):
        # Shift trick for numerical stability: subtracting the max prevents np.exp()
        # from overflowing to infinity when dealing with large raw logits.
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # Standard forward pass: Z = XW + b, followed by A = activation(Z)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.probs = self.softmax(self.z3)
        return self.probs

    def cross_entropy_loss(self, y_onehot, probs):
        n = y_onehot.shape[0]
        # Add epsilon (1e-12) to prevent taking log(0), which results in NaN
        log_probs = np.log(probs + 1e-12)
        loss = -np.sum(y_onehot * log_probs) / n
        return loss

    def backward(self, X, y_onehot):
        n = X.shape[0]

        # Output gradient: The derivative of Softmax + Cross Entropy simplifies beautifully
        # to just (Predicted Probs - Actual). Divided by n to average across the batch.
        dz3 = (self.probs - y_onehot) / n
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Hidden Layer 2: apply chain rule, multiplying by derivative of local ReLU
        dz2 = (dz3 @ self.W3.T) * self.relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden Layer 1: continue propagating error backward
        dz1 = (dz2 @ self.W2.T) * self.relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Parameter updates via standard Gradient Descent
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(
        self, X, y_onehot, X_val, y_val_onehot, epochs=100, batch_size=64, verbose=True
    ):
        n = X.shape[0]
        train_losses, val_losses = [], []

        # Mini-batch Stochastic Gradient Descent (SGD)
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch to break any ordering bias
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y_onehot[perm]

            # Process data in chunks (mini-batches) to balance computation speed and memory
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                self.forward(Xb)
                self.backward(Xb, yb)

            # Record epoch-level losses for plotting/debugging
            probs_train = self.forward(X)
            t_loss = self.cross_entropy_loss(y_onehot, probs_train)

            probs_val = self.forward(X_val)
            v_loss = self.cross_entropy_loss(y_val_onehot, probs_val)

            train_losses.append(t_loss)
            val_losses.append(v_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs} — train_loss={t_loss:.4f}, val_loss={v_loss:.4f}"
                )

        return train_losses, val_losses

    def predict(self, X):
        probs = self.forward(X)
        # Returns the index of the highest probability class
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.forward(X)
