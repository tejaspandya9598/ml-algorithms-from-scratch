"""
Benchmark every from-scratch model against its scikit-learn counterpart.

The whole point of the homeworks was to implement the algorithms by hand and then
check they actually match a battle-tested library. This script does that on real
datasets and writes a comparison table + chart to reports/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import GradientBoostingRegressor as SkGBR
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as SkKNN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SkDT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlscratch.ensemble import BaggingKNNRegressor, RandomForestClassifier  # noqa: E402
from mlscratch.linear_model import LogisticRegressionScratch                # noqa: E402
from mlscratch.metrics import accuracy_manual, rmse_metric                  # noqa: E402
from mlscratch.neighbors import KNNRegressor                                # noqa: E402
from mlscratch.svm import SVMClassifier                                     # noqa: E402
from mlscratch.tree import DecisionTreeClassifier, RegressionTree           # noqa: E402

FIG = ROOT / "reports" / "figures"


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def classification() -> pd.DataFrame:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    s = StandardScaler().fit(Xtr)
    Xtr, Xte = s.transform(Xtr), s.transform(Xte)

    rows = []
    pairs = {
        "Logistic Regression": (LogisticRegressionScratch(lr=0.1, n_epochs=500), SkLogReg(max_iter=2000)),
        "SVM (linear)": (SVMClassifier(C=1.0, kernel="linear"), SVC(kernel="linear")),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6), SkDT(max_depth=6, random_state=0)),
        "Random Forest": (RandomForestClassifier(n_estimators=15, max_depth=8), SkRF(n_estimators=15, max_depth=8, random_state=0)),
    }
    for name, (scratch, sk) in pairs.items():
        # SVM here trains on +/-1 labels; map and back.
        if name.startswith("SVM"):
            (_, ft) = _timed(lambda: scratch.fit(Xtr, np.where(ytr == 0, -1, 1)))
            pred = np.where(scratch.predict(Xte) <= 0, 0, 1)
        else:
            (_, ft) = _timed(lambda: scratch.fit(Xtr, ytr))
            pred = np.asarray(scratch.predict(Xte))
        (_, ft_sk) = _timed(lambda: sk.fit(Xtr, ytr))
        rows.append({
            "task": "classification", "model": name,
            "scratch": accuracy_manual(yte, pred), "sklearn": accuracy_manual(yte, sk.predict(Xte)),
            "metric": "accuracy", "scratch_s": ft, "sklearn_s": ft_sk,
        })
    return pd.DataFrame(rows)


def regression() -> pd.DataFrame:
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    s = StandardScaler().fit(Xtr)
    Xtr, Xte = s.transform(Xtr), s.transform(Xte)

    rows = []
    pairs = {
        "Regression Tree": (RegressionTree(max_depth=4), None),
        "KNN Regressor": (KNNRegressor(k=10), SkKNN(n_neighbors=10)),
        "Bagging KNN": (BaggingKNNRegressor(n_estimators=15, k=10), SkGBR(random_state=0)),
    }
    sk_tree = __import__("sklearn.tree", fromlist=["DecisionTreeRegressor"]).DecisionTreeRegressor
    pairs["Regression Tree"] = (RegressionTree(max_depth=4), sk_tree(max_depth=4, random_state=0))
    for name, (scratch, sk) in pairs.items():
        (_, ft) = _timed(lambda: scratch.fit(Xtr, ytr))
        pred = np.asarray(scratch.predict(Xte))
        sk_rmse = np.nan
        if sk is not None:
            sk.fit(Xtr, ytr)
            sk_rmse = rmse_metric(yte, sk.predict(Xte))
        rows.append({
            "task": "regression", "model": name,
            "scratch": rmse_metric(yte, pred), "sklearn": sk_rmse,
            "metric": "rmse", "scratch_s": ft, "sklearn_s": np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    clf, reg = classification(), regression()
    table = pd.concat([clf, reg], ignore_index=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    table.to_csv(ROOT / "reports" / "benchmark.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Accuracy comparison chart (classification only).
    c = clf.set_index("model")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(c))
    ax.bar(x - 0.2, c["scratch"], 0.4, label="from scratch", color="#3b6ea5")
    ax.bar(x + 0.2, c["sklearn"], 0.4, label="scikit-learn", color="#2c7a4b")
    ax.set_xticks(x); ax.set_xticklabels(c.index, rotation=15)
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("test accuracy")
    ax.set_title("From scratch vs scikit-learn — classification", fontweight="bold")
    ax.legend(frameon=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "scratch_vs_sklearn.png", dpi=160); plt.close(fig)
    print(f"\nwrote -> reports/benchmark.csv, {FIG}/scratch_vs_sklearn.png")


if __name__ == "__main__":
    main()
