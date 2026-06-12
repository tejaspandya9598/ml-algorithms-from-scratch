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

    # sort samples by predicted score from highest to lowest
    # this simulates sweeping the threshold from strict to lenient
    paired = sorted(zip(y_scores, y_true), key=lambda x: -x[0])

    tp = 0
    fp = 0

    total_pos = sum(1 for _, y in paired if y == 1)
    total_neg = len(paired) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    # ROC curve starts at the origin
    tpr_list = [0.0]
    fpr_list = [0.0]

    # each step is like lowering the threshold by one notch
    for score, label in paired:
        if label == 1:
            tp += 1  # correctly captured a positive
        else:
            fp += 1  # incorrectly flagged a negative

        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    # trapezoidal rule: width = delta(fpr), height = avg of adjacent tpr values
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc
