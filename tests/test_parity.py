"""The from-scratch models should track scikit-learn, and the hand-written metrics
should be correct. Small, fast checks on the breast-cancer set."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlscratch.linear_model import LogisticRegressionScratch
from mlscratch.metrics import accuracy_manual
from mlscratch.tree import DecisionTreeClassifier


def _data():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    s = StandardScaler().fit(Xtr)
    return s.transform(Xtr), s.transform(Xte), ytr, yte


def test_logreg_matches_sklearn():
    Xtr, Xte, ytr, yte = _data()
    m = LogisticRegressionScratch(lr=0.1, n_epochs=500)
    m.fit(Xtr, ytr)
    scratch_acc = accuracy_manual(yte, np.asarray(m.predict(Xte)))
    sk = SkLR(max_iter=2000).fit(Xtr, ytr)
    sklearn_acc = accuracy_manual(yte, sk.predict(Xte))
    assert scratch_acc >= sklearn_acc - 0.03   # within 3 points of the library


def test_decision_tree_is_accurate():
    Xtr, Xte, ytr, yte = _data()
    m = DecisionTreeClassifier(max_depth=6)
    m.fit(Xtr, ytr)
    assert accuracy_manual(yte, np.asarray(m.predict(Xte))) > 0.90


def test_metrics_are_correct():
    y_true, y_pred = [0, 1, 1, 0, 1], [0, 1, 0, 0, 1]
    assert abs(accuracy_manual(y_true, y_pred) - 0.8) < 1e-9   # 4 of 5 correct


def test_roc_auc_handles_ties_like_sklearn():
    """Stepping one sample at a time through a block of equal scores traces a
    staircase through the block instead of the single diagonal the threshold
    actually produces, and the answer depended on the arbitrary order the tied
    samples arrived in. A shallow tree emits only a handful of distinct scores,
    so this is the common case, not the corner case."""
    from sklearn.metrics import roc_auc_score

    from mlscratch.metrics import roc_auc_manual

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    for decimals in (1, 2, 3):
        scores = np.round(rng.random(400), decimals)
        assert abs(roc_auc_manual(y, scores) - roc_auc_score(y, scores)) < 1e-12


def test_r2_on_a_constant_target():
    from mlscratch.metrics import r2_metric

    ones = np.ones(5)
    assert r2_metric(ones, ones) == 1.0     # was nan (0/0)
    assert r2_metric(ones, np.zeros(5)) == 0.0   # was -inf


def test_adf_test_runs_and_separates_stationary_from_not():
    """`adfuller` was never imported - every call raised NameError."""
    from mlscratch.timeseries import adf_test

    rng = np.random.default_rng(0)
    assert adf_test(rng.normal(size=300), verbose=False)[2] is True
    assert adf_test(np.cumsum(rng.normal(size=300)), verbose=False)[2] is False


def test_both_svms_agree_on_breast_cancer():
    """The README and the module docstring described a hinge-loss sub-gradient
    learner and only the dual QP existed."""
    from mlscratch.metrics import accuracy_manual
    from mlscratch.svm import LinearSVMSubgradient, SVMClassifier

    Xtr, Xte, ytr, yte = _data()
    y_pm = 2 * yte - 1

    primal = LinearSVMSubgradient(C=50, lr=0.05, n_iters=4000, random_state=0).fit(Xtr, ytr)
    dual = SVMClassifier(C=1.0).fit(Xtr, 2 * ytr - 1)

    acc_primal = accuracy_manual(y_pm, primal.predict(Xte))
    acc_dual = accuracy_manual(y_pm, dual.predict(Xte))
    assert acc_primal > 0.95 and acc_dual > 0.95
    assert abs(acc_primal - acc_dual) < 0.02
    assert primal.losses[-1] < primal.losses[0]        # it actually descended


def test_svm_never_predicts_the_zero_class():
    """np.sign returns 0 on the decision boundary, and 0 is not a class."""
    from mlscratch.svm import LinearSVMSubgradient, SVMClassifier

    Xtr, Xte, ytr, _ = _data()
    for model in (LinearSVMSubgradient(C=50, n_iters=200, random_state=0).fit(Xtr, ytr),
                  SVMClassifier(C=1.0).fit(Xtr, 2 * ytr - 1)):
        preds = model.predict(np.zeros((3, Xte.shape[1])))
        assert set(np.unique(preds)) <= {-1, 1}


def test_rbf_kernel_never_exceeds_one():
    """||a||^2 + ||b||^2 - 2a.b is exact in algebra, not in floating point: a
    point against itself came out slightly negative and exp(-gamma*neg) > 1."""
    from mlscratch.svm import SVMClassifier

    rng = np.random.default_rng(1)
    X = rng.normal(scale=1e4, size=(30, 8))
    K = SVMClassifier(kernel="rbf", gamma=0.5)._kernel_func(X, X)
    assert K.max() <= 1.0 + 1e-12
    assert np.allclose(np.diag(K), 1.0)


def test_logistic_regression_is_reproducible():
    """The epoch shuffle used the global numpy RNG, so two runs of the same
    model gave different weights."""
    Xtr, _, ytr, _ = _data()
    a = LogisticRegressionScratch(lr=0.1, n_epochs=50, random_state=7).fit(Xtr, ytr)
    b = LogisticRegressionScratch(lr=0.1, n_epochs=50, random_state=7).fit(Xtr, ytr)
    assert np.allclose(a.weights, b.weights) and a.bias == b.bias


def test_knn_uses_k_neighbours():
    """simple_knn_predict called np.argmin - 1-NN under a k-NN name, with no way
    to ask for anything else."""
    from mlscratch.neighbors import simple_knn_predict

    Xtr = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    ytr = np.array([1, 0, 0, 0, 0])
    assert simple_knn_predict(Xtr, ytr, np.array([[0.1]]), k=1)[0] == 1   # nearest is the outlier
    assert simple_knn_predict(Xtr, ytr, np.array([[0.1]]), k=3)[0] == 0   # majority overrules it


def test_pca_survives_a_constant_feature():
    from mlscratch.decomposition import pca_algorithm

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=60), rng.normal(size=60), np.full(60, 3.0)])
    proj, evr = pca_algorithm(X, k=2)
    assert np.isfinite(proj).all() and np.isfinite(evr).all()
