"""
Support-vector machines from scratch (HW2), two ways.

  * `LinearSVMSubgradient` — the primal problem, 0.5||w||^2 + C * mean hinge,
    minimised by sub-gradient descent. O(n) per step, so it scales.
  * `SVMClassifier`        — the dual QP with linear / RBF / polynomial kernels,
    solved with SciPy's SLSQP. Kernels, but an n x n matrix.

The module docstring and the README both used to describe the sub-gradient
learner, and only the dual solver existed. On the same data the two now agree,
which is the point of writing both.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class LinearSVMSubgradient:
    """Primal linear SVM: min 0.5||w||^2 + C * mean(max(0, 1 - y(w.x + b))).

    The hinge is not differentiable at its kink, so this is a sub-gradient
    method, not gradient descent: at the kink any value in the subdifferential
    is a valid direction, and we take the one from the active side. The step size
    decays as lr / (1 + t/n_iters) because a constant step on a non-smooth
    objective circles the optimum rather than settling into it.

    A note on C. The hinge here is a *mean*, where scikit-learn's LinearSVC sums
    it, so this C is smaller than sklearn's by a factor of n. On breast-cancer
    (n=426) C=50 reaches 0.986, the same test accuracy as the dual solver below;
    C=1 over-regularises to 0.944.
    """

    def __init__(self, C=1.0, lr=0.01, n_iters=1000, batch_size=32, random_state=None):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.losses = []

    def _objective(self, X, y):
        margins = 1.0 - y * (X @ self.w + self.b)
        return 0.5 * self.w @ self.w + self.C * np.mean(np.maximum(0.0, margins))

    def fit(self, X, y):
        """`y` must be in {-1, +1}; {0, 1} is remapped."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if set(np.unique(y)) <= {0.0, 1.0}:
            y = 2.0 * y - 1.0

        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d)
        self.b = 0.0

        for t in range(self.n_iters):
            idx = rng.choice(n, size=min(self.batch_size, n), replace=False)
            Xb, yb = X[idx], y[idx]

            # active set: only the points inside the margin push on w
            violating = (yb * (Xb @ self.w + self.b)) < 1.0
            dw = self.w - self.C * (yb[violating] @ Xb[violating]) / len(yb)
            db = -self.C * yb[violating].sum() / len(yb)

            step = self.lr / (1.0 + t / self.n_iters)
            self.w -= step * dw
            self.b -= step * db
            self.losses.append(self._objective(X, y))
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        # >= 0 rather than sign(): sign returns 0 on the boundary, which is not a class
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(int)


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

            # squared Euclidean distance between all pairs
            sq_dist = (np.sum(x1**2, axis=1, keepdims=True)
                       + np.sum(x2**2, axis=1, keepdims=True).T
                       - 2 * x1 @ x2.T)

            # ||a||^2 + ||b||^2 - 2a.b is exact in algebra and not in floating
            # point: a point against itself came out slightly negative, and
            # exp(-gamma * negative) is a kernel value above 1.
            np.maximum(sq_dist, 0.0, out=sq_dist)

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
        # >= 0 rather than np.sign: sign returns 0 for a point exactly on the
        # boundary, and 0 is not one of the two classes.
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(int)

    def predict_proba_approx(self, X):
        # rough probability estimate using sigmoid on decision values
        d = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
