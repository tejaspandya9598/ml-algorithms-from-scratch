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
