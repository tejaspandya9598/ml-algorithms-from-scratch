"""
Ensembles from scratch (HW3): gradient boosting, random forests, and bagging — built
on the from-scratch trees and KNN.
"""
from __future__ import annotations

import numpy as np
from mlscratch.tree import ClassificationTree, RegressionTree
from mlscratch.neighbors import KNNRegressor


class GradientBoostingRegressor:
    """
    Gradient boosting builds a 'team' of small, shallow trees
    Instead of one giant tree trying to learn everything, we train a sequence of small trees, where each new tree specifically focuses on fixing the mistakes made by the previous ones
    """

    def __init__(
        self,
        n_estimators=50,  # number of trees
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        subsample=1.0,  # The fraction of data to randomly select for each tree
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        # Every draw below used the global np.random while the scikit-learn models
        # these are benchmarked against all take a random_state. The comparison was
        # a seeded library against an unseeded reimplementation, and the number in
        # the README was whichever draw got written down.
        self.random_state = random_state

        self.trees = []
        self.init_pred = None
        self.train_errors = []

    def fit(self, X, y):
        """Trains the sequence of trees to correct each other's errors"""
        n = len(y)
        rng = np.random.default_rng(self.random_state)

        # baseline guess: the average of all targets
        self.init_pred = np.mean(y)

        current_pred = np.full(n, self.init_pred)

        self.trees = []
        self.train_errors = []

        # Build the trees one by one
        for i in range(self.n_estimators):
            # Calculate the residuals
            residuals = y - current_pred

            # Subsampling
            # If subsample < 1.0, we don't use all the data to train this specific tree.
            if self.subsample < 1.0:
                sample_size = max(1, int(n * self.subsample))
                # Randomly pick rows without replacement
                idx = rng.choice(n, sample_size, replace=False)
            else:
                # Use all the data
                idx = np.arange(n)

            # Create blank RegressionTree
            tree = RegressionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )

            tree.fit(X[idx], residuals[idx])
            self.trees.append(tree)

            # Ask the newly trained tree to predict the mistakes for the whole dataset
            update = tree.predict(X)

            # Update our running predictions
            current_pred += self.learning_rate * update

            mse = mse_metric(y, current_pred)
            self.train_errors.append(mse)

        return self

    def predict(self, X):
        """Makes predictions by passing data through the every tree"""
        pred = np.full(X.shape[0], self.init_pred)

        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)

        return pred


class RandomForestClassifier:
    """
    A Random Forest is essentially a 'committee' of Decision Trees. Instead of trusting one giant, overconfident tree, we train many slightly different trees and have them vote on the final answer. This prevents overfitting and increases accuracy.
    """

    def __init__(
        self, n_estimators=10, max_depth=10, min_samples_split=5, max_features="sqrt",
        random_state=None,
    ):
        # Bootstrap draws and per-tree feature subsets both came from the global
        # np.random, so this forest scored 0.937 on one run and 0.951 on the next
        # while the seeded sklearn forest it is compared against never moved.
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = (
            min_samples_split  # Minimum samples needed to split a node
        )

        # 'max_features' is crucial for forest diversity. By limiting the features each tree
        # is allowed to look at, we force them to learn different patterns instead of all
        # memorizing the exact same strong feature.
        self.max_features = max_features
        self.trees = []

    def _get_max_features(self, n_features):
        """calculates exactly how many features each tree gets to see."""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def fit(self, X, y):
        """Builds the forest by training each tree on a slightly different version of the data."""
        n_samples, n_features = X.shape
        mf = self._get_max_features(n_features)
        self.trees = []
        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # Create blank ClassificationTree
            tree = ClassificationTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=mf,
                # each tree gets its own stream, derived from the forest's seed
                random_state=None if self.random_state is None else int(rng.integers(2**32)),
            )

            # Train the individual tree on its unique bootstrapped dataset
            tree.fit(X_boot, y_boot)

            self.trees.append(tree)

        return self

    def predict(self, X):
        """Asks every tree for its opinion, then takes a democratic vote."""
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        result = np.zeros(X.shape[0], dtype=int)

        # Loop through the columns (the individual data points we are trying to predict)
        for j in range(X.shape[0]):
            # Count how many trees voted for class 0, class 1, etc.
            counts = np.bincount(all_preds[:, j])
            # The final prediction is simply the class that got the most votes
            result[j] = np.argmax(counts)

        return result

    def predict_proba(self, X):
        """
        returns the percentage of trees that voted for a particular class
        """
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        # Calculate the average across the columns (trees).
        proba = np.mean(all_preds, axis=0)

        return proba


class BaggingKNNRegressor:
    """
    A 'committee' of KNN models.
    Instead of having one algorithm memorize the whole dataset, we have multiple algorithms
    memorize slightly different, randomly sampled versions of the dataset.
    They then average their answers to give a much more stable prediction.
    """

    def __init__(self, n_estimators=10, k=5, metric="euclidean", sample_ratio=1.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.k = k
        self.metric = metric
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        """Builds the committee by giving each member a slightly different dataset to memorize."""
        n = len(y)
        sample_size = max(1, int(n * self.sample_ratio))
        self.models = []
        rng = np.random.default_rng(self.random_state)

        # Create our committee members one by one
        for _ in range(self.n_estimators):
            # Randomly pick rows with replacement.
            # Some rows will be picked multiple times, some will be completely ignored.
            # This ensures every KNN model "learns" a slightly different version of reality.
            boot_idx = rng.choice(n, sample_size, replace=True)

            # Initialize a blank KNN
            knn = KNNRegressor(k=self.k, metric=self.metric)

            # Train the KNN
            knn.fit(X[boot_idx], y[boot_idx])

            # Add the initialized model to our committee
            self.models.append(knn)

        return self

    def predict(self, X):
        """Asks the committee for their predictions and takes the average."""
        all_preds = np.array([m.predict(X) for m in self.models])
        return np.mean(all_preds, axis=0)

    def predict_std(self, X):
        """
        Measures how much the committee disagrees!
        """
        # Get all predictions
        all_preds = np.array([m.predict(X) for m in self.models])

        # calculate the spread (Standard Deviation)
        return np.std(all_preds, axis=0)
