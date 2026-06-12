"""
Linear support-vector machine from scratch (HW2): hinge loss + L2 regularisation,
trained by sub-gradient descent.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class SVMClassifier:
    """
    SVM using dual formulation with linear, RBF, and polynomial kernels.
    Optimization done through scipy's SLSQP solver.
    """

    def __init__(self, C=1.0, kernel='linear', gamma=0.1, degree=3):
        self.C = C # how much to penalize misclassifications
        self.kernel = kernel
        self.gamma = gamma # bandwidth for RBF kernel
        self.degree = degree # degree for polynomial kernel
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None
        self.support_alphas = None
        self.b = 0.0

    def _kernel_func(self, x1, x2):

        """Computes the kernel matrix between x1 and x2."""

        if self.kernel == 'linear':
            return x1 @ x2.T

        elif self.kernel == 'rbf':
            if x1.ndim == 1: x1 = x1.reshape(1, -1)
            if x2.ndim == 1: x2 = x2.reshape(1, -1)

            # squared euclidean distance between all pairs
            sq_dist = (np.sum(x1**2, axis=1, keepdims=True)
                       + np.sum(x2**2, axis=1, keepdims=True).T
                       - 2 * x1 @ x2.T)

            return np.exp(-self.gamma * sq_dist)

        elif self.kernel == 'poly':
            return (x1 @ x2.T + 1) ** self.degree

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):

        """
        Solves the dual QP:
        max  sum(alpha_i) - 0.5 * alpha^T (y*y^T * K) alpha
        s.t. 0 <= alpha_i <= C, sum(alpha_i * y_i) = 0
        """

        n = X.shape[0]
        y = y.astype(float)
        K = self._kernel_func(X, X)

        # Q_ij = y_i * y_j * K_ij
        Q = np.outer(y, y) * K

        # minimize the negative of the dual objective
        def objective(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

        def gradient(alpha):
            return Q @ alpha - np.ones(n)

        # constraint: alpha dot y = 0
        constraints = [{'type': 'eq',
                        'fun': lambda a: np.dot(a, y),
                        'jac': lambda a: y}]

        # each alpha between 0 and C
        bounds = [(0, self.C)] * n
        alpha0 = np.zeros(n)

        # solve using SLSQP
        result = minimize(objective, alpha0, jac=gradient,
                          method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-8})
        self.alphas = result.x

        # support vectors are the ones with alpha > ~0
        sv_mask = self.alphas > 1e-6
        self.support_alphas  = self.alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_labels  = y[sv_mask]

        # compute bias from free SVs (alphas not pinned to C)
        free_mask = (self.alphas > 1e-6) & (self.alphas < self.C - 1e-6)

        if free_mask.sum() > 0:
            K_sv = self._kernel_func(X[free_mask], self.support_vectors)
            self.b = np.mean(
                y[free_mask] - (self.support_alphas * self.support_labels) @ K_sv.T
            )

        else:
            # fallback if all SVs are at the boundary
            K_sv = self._kernel_func(self.support_vectors, self.support_vectors)
            self.b = np.mean(
                self.support_labels - (self.support_alphas * self.support_labels) @ K_sv.T
            )

        print(f"  SVM fitted - {sv_mask.sum()} support vectors out of {n} samples")
        return self

    def decision_function(self, X):

        """f(x) = sum(alpha_i * y_i * K(x_i, x)) + b"""

        K = self._kernel_func(X, self.support_vectors)
        return (self.support_alphas * self.support_labels) @ K.T + self.b

    def predict(self, X):
        # just take the sign of the decision function
        return np.sign(self.decision_function(X)).astype(int)

    def predict_proba_approx(self, X):
        # rough probability estimate using sigmoid on decision values
        d = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
