"""
Evaluation metrics written by hand — no sklearn.metrics.

Classification (accuracy, precision/recall/F1, confusion matrix, ROC-AUC) and
regression (MSE/RMSE/MAE/R^2). Both the from-scratch models and their scikit-learn
baselines are scored through these, so any comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np


def mse_metric(y_true, y_pred):
    # average of squared differences between actual and predicted
    return np.mean((y_true - y_pred) ** 2)


def rmse_metric(y_true, y_pred):
    # square root of MSE so the error is in the same units as the target
    return np.sqrt(mse_metric(y_true, y_pred))


def mae_metric(y_true, y_pred):
    # average absolute gap — less sensitive to outliers than RMSE
    return np.mean(np.abs(y_true - y_pred))


def r2_metric(y_true, y_pred):
    # how much variance the model explains (1.0 = perfect, 0 = predicting the mean)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        # A constant target has no variance to explain. Dividing by it gave -inf
        # for any error at all and nan for a perfect fit.
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot


def accuracy_manual(y_true, y_pred):
    # fraction of predictions that match the true labels
    return np.mean(np.array(y_true) == np.array(y_pred))


def confusion_matrix_manual(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # map each label to a row/column index
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    # diagonal entries = correct predictions, off-diagonal = mistakes
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    return cm, labels


def precision_recall_f1_manual(y_true, y_pred, pos_label=1):

    # count true positives: we said positive and it actually was
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)

    # count false positives: we said positive but it was actually negative
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)

    # count false negatives: it was positive but we missed it
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

    # guard against division by zero-
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return prec, rec, f1


def roc_auc_manual(y_true, y_scores):
    """Area under the ROC curve, sweeping the threshold from strict to lenient.

    Tied scores have to move the curve in one step. Advancing one sample at a
    time through a block of equal scores traces a staircase through the interior
    of the block instead of the single diagonal the threshold actually produces,
    and the answer then depends on the arbitrary order the tied samples happened
    to arrive in. On 400 samples rounded to one decimal that read 0.5189 against
    sklearn's 0.5196; the gap grows with the size of the tie blocks, which is
    exactly the case for a shallow tree or a small forest, where the model only
    emits a handful of distinct scores.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    total_pos = int(np.sum(y_true == 1))
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    # sort samples by predicted score from highest to lowest
    order = np.argsort(-y_scores, kind="mergesort")
    scores, labels = y_scores[order], y_true[order]

    tp = fp = 0
    tpr_list = [0.0]
    fpr_list = [0.0]

    # walk the distinct score levels: every tie block is a single threshold
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        block = labels[i:j]
        tp += int(np.sum(block == 1))
        fp += len(block) - int(np.sum(block == 1))
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)
        i = j

    # trapezoidal rule: width = delta(fpr), height = avg of adjacent tpr values
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc
