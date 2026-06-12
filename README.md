# ML Algorithms From Scratch

Classic machine-learning algorithms implemented **by hand in NumPy** — no
`scikit-learn`, no autograd — then benchmarked against the library versions to prove
they're correct. Grew out of four graduate ML assignments, merged into one library.

![From scratch vs scikit-learn](reports/figures/scratch_vs_sklearn.png)

## The point

Anyone can call `model.fit()`. These are written from the math up — the splitting
criteria, the gradient updates, the backprop — and then checked against
`scikit-learn` / `PyTorch` on real datasets. On the breast-cancer set the
hand-written models reproduce the library's **test accuracy to the digit**:

| Model | From scratch | scikit-learn |
|---|--:|--:|
| Logistic Regression | 0.986 | 0.986 |
| SVM (linear) | 0.986 | 0.986 |
| Decision Tree | 0.937 | 0.937 |
| Random Forest | 0.958 | 0.958 |

(Regression on the diabetes set lands within ~1 RMSE of the library trees/KNN.)
They're slower — pure NumPy vs. compiled C — but the *answers* match, which is the
whole point.

## What's implemented

| Module | From scratch | Benchmarked against |
|---|---|---|
| `decomposition` | PCA (covariance eigendecomposition) | `sklearn.decomposition.PCA` |
| `timeseries` | multiplicative decomposition, ADF test, lag features | — |
| `neighbors` | KNN classifier + regressor | `sklearn.neighbors` |
| `linear_model` | logistic regression (gradient descent) | `sklearn.linear_model` |
| `svm` | linear SVM (hinge loss / dual) | `sklearn.svm.SVC` |
| `tree` | decision & regression trees (Gini / variance) | `sklearn.tree` |
| `ensemble` | random forest, gradient boosting, bagging | `sklearn.ensemble` |
| `neural.mlp` | multilayer perceptron (NumPy backprop) | — |
| `neural.lstm` | LSTM cell + sequence model | `torch.nn.LSTM` |
| `neural.autoencoder` | autoencoder on financial data | — |

## Structure

```
ml-algorithms-from-scratch/
├── src/mlscratch/        # the from-scratch library
│   ├── metrics.py        # accuracy, P/R/F1, ROC-AUC, MSE/RMSE/MAE/R2 — also by hand
│   ├── decomposition.py  neighbors.py  linear_model.py  svm.py  tree.py  ensemble.py
│   └── neural/           # mlp.py  lstm.py  autoencoder.py
├── benchmarks/run.py     # from-scratch vs scikit-learn comparison + chart
├── notebooks/            # the original four assignments (the working record)
├── reports/              # benchmark.csv + figures
└── tests/                # parity tests vs scikit-learn
```

## Run

```bash
uv sync
uv run python benchmarks/run.py     # comparison table + reports/figures/scratch_vs_sklearn.png
uv run pytest                       # asserts the from-scratch models match scikit-learn

uv sync --extra neural              # adds torch for the LSTM / autoencoder modules
```

## Notes

- Metrics are hand-written too (`metrics.py`), so the comparison is fair — both sides
  scored by the same code.
- The neural modules (`neural/lstm.py`, `neural/autoencoder.py`) keep a PyTorch
  reference next to the from-scratch version; install the `neural` extra to use them.
- These prioritise being *readable and correct* over fast. For production, use the
  library — the value here is understanding what it's doing.

---

*Built by Tejas Pandya — NYU MSFE. Merged from four graduate ML assignments.*
