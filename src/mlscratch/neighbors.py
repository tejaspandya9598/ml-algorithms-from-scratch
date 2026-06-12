"""
k-nearest-neighbours from scratch: a classifier (HW1) and a regressor (HW3).
"""
from __future__ import annotations

import numpy as np


def simple_knn_predict(X_train_proj, y_train, X_test_proj):
    y_pred = []
    for test_point in X_test_proj:
        # Calculating Euclidean distance from this test point to all training points
        distances = np.sqrt(np.sum((X_train_proj - test_point) ** 2, axis=1))

        # Finding the index of the closest training point
        nearest_idx = np.argmin(distances)

        # Assigning the label of the nearest neighbor
        y_pred.append(y_train[nearest_idx])
    return np.array(y_pred)


class KNNRegressor:
    """
    K-Nearest Neighbors (KNN) is like asking your closest neighbors for advice.
    To predict the price of a house, it finds the 'k' houses that are most similar
    to it (closest in distance) and simply averages their prices.
    """

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        memorizes the entire dataset so it can look up neighbors later.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def _distance(self, a, b):
        """Calculates the mathematical distance between data points."""
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X):
        """Predicts the target value for new, unseen data points."""
        # Create an empty array to hold our final predictions
        preds = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            # Calculate the distance between new point and EVERY point in our memorized training set
            dists = self._distance(self.X_train, x.reshape(1, -1))

            # Sort those distances from smallest to largest and grab the index numbers of the top 'k' closest
            nn_idx = np.argsort(dists)[: self.k]

            # Look up the actual targets of those 'k' closest neighbors and take the average
            preds[i] = np.mean(self.y_train[nn_idx])

        return preds
