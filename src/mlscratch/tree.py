"""
Decision trees from scratch: a classifier (HW2) and regression/classification trees
(HW3). Recursive greedy splits on Gini impurity / variance reduction.
"""
from __future__ import annotations

import numpy as np
from collections import Counter


class DecisionTreeNode:

    """Represents a single node - could be a split or a leaf."""

    def __init__(self, *, predicted_class=None, feature_index=None,
                 threshold=None, is_categorical=False,
                 left=None, right=None, children=None):

        self.predicted_class = predicted_class
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_categorical = is_categorical
        self.left = left # <= threshold (numerical)
        self.right = right # >  threshold (numerical)
        self.children = children


class DecisionTreeClassifier:

    """
    Decision tree with support for gini and entropy, categorical + numerical features.
    """

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):

        assert criterion in ('gini', 'entropy'), "criterion must be 'gini' or 'entropy'"

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.categorical_features = set()


    # impurity measures

    @staticmethod
    def _gini(y):

        """Gini = 1 - sum(p_i^2)"""

        counts = Counter(y)
        n = len(y)

        if n == 0:
            return 0.0

        # higher value = more mixed classes
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    @staticmethod
    def _entropy(y):

        """entropy = -sum(p_i * log2(p_i))"""

        counts = Counter(y)
        n = len(y)

        if n == 0:
            return 0.0

        ent = 0.0

        for c in counts.values():
            p = c / n

            # skip zero to avoid log(0)
            if p > 0:
                ent -= p * np.log2(p)

        return ent

    # pick whichever criterion was chosen at init
    def _impurity(self, y):
        return self._gini(y) if self.criterion == 'gini' else self._entropy(y)

    # finding best split across all features
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best = None

        # impurity before any split - we want to reduce this
        parent_imp = self._impurity(y)

        for feat_idx in range(n_features):
            col = X[:, feat_idx]

            if feat_idx in self.categorical_features:

                # categorical: one branch per unique value
                unique_vals = set(col)

                # nothing to split on if only 1 value
                if len(unique_vals) <= 1:
                    continue

                # group row indices by category
                partitions = {v: [] for v in unique_vals}

                for i, v in enumerate(col):
                    partitions[v].append(i)

                # weighted impurity after this split
                weighted_imp = sum(
                    len(idx) / n_samples * self._impurity(y[np.array(idx)])
                    for idx in partitions.values()
                )
                # info gain = how much impurity dropped
                gain = parent_imp - weighted_imp

                if gain > best_gain:
                    best_gain = gain
                    best = {'feature': feat_idx, 'categorical': True,
                            'partitions': {v: np.array(idx) for v, idx in partitions.items()}}

            else:
                # numerical: try midpoints between consecutive unique values
                sorted_vals = np.unique(col)

                if len(sorted_vals) <= 1:
                    continue

                # midpoints give us candidate thresholds to try
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

                for thr in thresholds:
                    left_mask  = col <= thr
                    right_mask = ~left_mask
                    n_l, n_r = left_mask.sum(), right_mask.sum()

                    # skip if everything lands on one side
                    if n_l == 0 or n_r == 0:
                        continue

                    w_imp = (n_l / n_samples * self._impurity(y[left_mask])
                           + n_r / n_samples * self._impurity(y[right_mask]))
                    gain = parent_imp - w_imp

                    # keep track of the best one so far
                    if gain > best_gain:
                        best_gain = gain
                        best = {'feature': feat_idx, 'categorical': False,
                                'threshold': thr}

        return best, best_gain

    # recursive tree building
    def _build(self, X, y, depth):

        # stop if pure node, too few samples, or hit max depth
        if (len(set(y)) == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)):
            return DecisionTreeNode(predicted_class=Counter(y).most_common(1)[0][0])

        split, gain = self._best_split(X, y)

        # no useful split found, just make a leaf
        if split is None or gain <= 0:
            return DecisionTreeNode(predicted_class=Counter(y).most_common(1)[0][0])

        # store majority class as fallback for unseen categories
        majority = Counter(y).most_common(1)[0][0]

        if split['categorical']:
            # build a subtree for each category value
            children = {}

            for val, idx in split['partitions'].items():
                children[val] = self._build(X[idx], y[idx], depth + 1)

            return DecisionTreeNode(feature_index=split['feature'],
                                    is_categorical=True,
                                    children=children,
                                    predicted_class=majority)
        else:
            # binary split: left goes <= threshold, right goes >
            mask = X[:, split['feature']] <= split['threshold']
            left  = self._build(X[mask],  y[mask],  depth + 1)
            right = self._build(X[~mask], y[~mask], depth + 1)

            return DecisionTreeNode(feature_index=split['feature'],
                                    threshold=split['threshold'],
                                    is_categorical=False,
                                    left=left, right=right,
                                    predicted_class=majority)

    def fit(self, X, y, categorical_features=None):
        X = np.array(X)
        y = np.array(y)
        self.categorical_features = categorical_features if categorical_features else set()

        # kick off the recursive build starting at depth 0
        self.tree = self._build(X, y, depth=0)

        return self

    def _predict_one(self, x, node):

        """Walk down the tree for a single sample."""

        # reached a leaf
        if node.left is None and node.right is None and node.children is None:
            return node.predicted_class

        if node.is_categorical:
            val = x[node.feature_index]

            # if we've seen this category before, go down that branch
            if node.children and val in node.children:
                return self._predict_one(x, node.children[val])
            else:
                return node.predicted_class  # unseen category, fall back to majority

        else:
            # go left or right depending on the threshold
            if x[node.feature_index] <= node.threshold:
                return self._predict_one(x, node.left)

            else:
                return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.array(X)

        # just run each row through the tree one by one
        return np.array([self._predict_one(x, self.tree) for x in X])


class RegressionTree:
    """A simple regression tree that splits data to minimize the Mean Squared Error (MSE)"""

    def __init__(self, max_depth=3, min_samples_split=5):
        # Set the rules for how big the tree can grow to prevent overfitting
        self.max_depth = max_depth  # How many levels deep the tree can go
        self.min_samples_split = (
            min_samples_split  # Minimum data points needed to justify a new split
        )
        self.tree = None  # This will hold our trained tree structure later

    def _mse(self, y):
        """
        Calculates the impurity of a group of target values i.e. returns the Sum of Squared Errors (Variance * N)
        """
        # If there's no data, there's no error
        if len(y) == 0:
            return 0.0

        return np.var(y) * len(y)

    def _best_split(self, X, y):
        """Looks at every feature and every possible cut-off point to find the best way to split the data"""
        n_samples, n_features = X.shape
        best_gain = -np.inf  # Keep track of the best improvement we've seen so far
        best_feat = None  # The column index of the best feature
        best_thresh = None  # The cut-off value for that feature

        # Calculate the error of the current group of data before any splitting
        parent_mse = self._mse(y)

        # Check each column (feature) one by one
        for feat in range(n_features):
            vals = X[:, feat]
            thresholds = np.unique(vals)  # All unique values in this column

            # Optimization: If there are too many unique values, it takes too long to check them all
            # So, we just pick 50 evenly spaced values (percentiles) to test instead
            if len(thresholds) > 50:
                thresholds = np.percentile(vals, np.linspace(0, 100, 50))

            # Try splitting the data at every potential cut-off point
            for t in thresholds:
                left_mask = vals <= t  # Data that goes to the left branch
                right_mask = (
                    ~left_mask
                )  # Data that goes to the right branch (greater than t)

                # Skip this split if it pushes all the data to one side
                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue

                # Calculate how much we reduced the error by making this split
                # Gain = (Error before) - (Error of left child + Error of right child)
                gain = parent_mse - self._mse(y[left_mask]) - self._mse(y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = t

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        """It recursively splits the data to build branches"""

        # STOPPING CRITERIAS:
        if (
            depth >= self.max_depth  # We've reached the maximum allowed depth
            or len(y)
            < self.min_samples_split  # Not enough data points left to split safely
            or np.var(y)
            < 1e-10  # All the target values are basically the same (pure node)
        ):
            # Create a "leaf" node. The prediction is just the average of the targets in this group.
            return {"leaf": True, "value": np.mean(y)}

        # Find the absolute best way to split the data we currently have
        feat, thresh = self._best_split(X, y)

        # If we couldn't find a valid split that improves anything, make it a leaf
        if feat is None:
            return {"leaf": True, "value": np.mean(y)}

        # Split the data based on the best feature and threshold we found
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        # Create a "decision" node, and recursively tell it to build its left and right child nodes
        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        """Starts the building process."""
        self.tree = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """Navigates a single data point down the tree until it hits a leaf"""
        # If we hit a leaf, return the final prediction
        if node["leaf"]:
            return node["value"]

        # Is our feature value less than or equal to the threshold?
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(
                x, node["left"]
            )  # Yes then it goes down the left branch
        else:
            return self._predict_one(
                x, node["right"]
            )  # No then it goes down the right branch

    def predict(self, X):
        """Takes a whole dataset and gets predictions for every single row"""
        return np.array([self._predict_one(x, self.tree) for x in X])


class ClassificationTree:
    """A decision tree that categorizes data by minimizing Gini impurity."""

    def __init__(self, max_depth=10, min_samples_split=5, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # force the tree to only pick from a random subset of columns.
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        """
        Calculates Gini Impurity
        """
        if len(y) == 0:
            return 0.0

        # Count how many of each class we have, then turn those into percentages (probabilities)
        counts = np.bincount(y)
        probs = counts / len(y)

        # Gini formula: 1 - sum(squared probabilities)
        return 1.0 - np.sum(probs**2)

    def _best_split(self, X, y):
        """Finds the best column and cut-off point to split the data into purer groups."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feat = None
        best_thresh = None

        # calculates gini for parent node
        parent_gini = self._gini(y)

        # If max_features is set, randomly pick a handful of columns to evaluate.
        # Otherwise, check all of them.
        if self.max_features and self.max_features < n_features:
            feat_indices = np.random.choice(
                n_features, self.max_features, replace=False
            )
        else:
            feat_indices = np.arange(n_features)

        # Loop through the chosen features
        for feat in feat_indices:
            vals = X[:, feat]
            thresholds = np.unique(vals)

            # Optimization: Use percentiles if there are too many unique values
            if len(thresholds) > 50:
                thresholds = np.percentile(vals, np.linspace(0, 100, 50))

            for t in thresholds:
                # Split the targets based on this threshold
                left = y[vals <= t]
                right = y[vals > t]

                if len(left) < 1 or len(right) < 1:
                    continue

                # Calculate the weighted average of the Gini impurity of the two new branches
                weighted_gini = (
                    len(left) * self._gini(left) + len(right) * self._gini(right)
                ) / n_samples

                # Gain (impurity reduced)
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = t

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        """Recursively splits the data to grow the tree branches."""
        # STOPPING CRITERIA: Max depth reached, too few samples, OR node is perfectly pure
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            # prediction is whatever class is most common in this group
            counts = np.bincount(y)
            return {"leaf": True, "value": np.argmax(counts)}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            counts = np.bincount(y)
            return {"leaf": True, "value": np.argmax(counts)}

        left_mask = X[:, feat] <= thresh
        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """Flowchart navigation for a single prediction."""
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def predict_proba(self, X):
        """
        Creates pseudo-probabilities
        """
        preds = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[np.arange(len(X)), preds] = 1.0
        return proba
