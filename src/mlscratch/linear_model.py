"""
Logistic regression from scratch (HW2): batch gradient descent on the cross-entropy
loss, with a sigmoid link.
"""
from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    """
    Binary logistic regression with mini-batch SGD, L2 regularization,
    and early stopping on validation loss.
    """

    def __init__(self, lr=0.01, n_epochs=1000, batch_size=32,
                 lambda_reg=0.01, patience=10, tol=1e-5, random_state=None):
        self.lr = lr
        # The batch shuffle used the global numpy RNG, so two runs of the same
        # model gave different weights. In a repo whose whole claim is matching
        # scikit-learn to three decimals, that has to be pinnable.
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.tol = tol
        self.weights = None
        self.bias = None

        # these will store loss curves for plotting
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def _sigmoid(z):

        # clip to avoid overflow in exp
        z = np.clip(z, -500, 500)

        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, X, y):

        """binary cross-entropy + L2 penalty"""

        h = self._sigmoid(X @ self.weights + self.bias)

        # clip predictions so log doesn't blow up
        h = np.clip(h, 1e-12, 1 - 1e-12)
        bce = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

        # Regularisation term. This used to divide by len(y) - the size of
        # whatever set it was handed - while the gradient divided by the
        # mini-batch size, so the curve being plotted was not the objective
        # being minimised, and the training and validation losses were scaled by
        # different amounts. Both sides now use the same penalty.
        l2 = (self.lambda_reg / 2.0) * np.sum(self.weights ** 2)

        return bce + l2

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        best_val_loss = np.inf
        patience_ctr  = 0
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.n_epochs):

            # reshuffle every epoch so batches aren't the same
            idx = rng.permutation(n_samples)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            # mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                mb = len(yb)

                # forward pass
                h     = self._sigmoid(Xb @ self.weights + self.bias)
                error = h - yb

                # L2 only on weights, not on bias
                dw = (1 / mb) * (Xb.T @ error) + self.lambda_reg * self.weights
                db = np.mean(error)

                # update step
                self.weights -= self.lr * dw
                self.bias    -= self.lr * db

            # track training loss each epoch
            self.train_losses.append(self._loss(X_train, y_train))

            # validation check if val set is provided
            if X_val is not None:
                val_loss = self._loss(X_val, y_val)
                self.val_losses.append(val_loss)

                # early stopping: if no improvement for `patience` epochs, stop
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    patience_ctr  = 0

                    # save best weights so far
                    self._best_w = self.weights.copy()
                    self._best_b = self.bias

                else:
                    patience_ctr += 1

                    if patience_ctr >= self.patience:

                        # roll back to best weights and stop
                        self.weights = self._best_w
                        self.bias    = self._best_b
                        print(f"  Early stopping at epoch {epoch+1}  "
                              f"(best val_loss = {best_val_loss:.6f})")

                        return self

        # if we used validation, make sure we end with best weights
        if X_val is not None and hasattr(self, '_best_w'):
            self.weights = self._best_w
            self.bias    = self._best_b

        return self

    # returns raw probabilities
    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    # threshold the probabilities to get 0/1 labels
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
