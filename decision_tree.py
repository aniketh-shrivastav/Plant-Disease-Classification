import numpy as np


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, max_thresholds=32, seed=42):
        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.max_features = max_features  # None or int
        self.max_thresholds = max(1, int(max_thresholds))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -----------------------
    # Compute Gini Impurity
    # -----------------------
    def gini(self, y):
        # count unique class occurrences manually
        class_counts = {}
        for label in y:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        gini = 1.0
        total = len(y)
        for label in class_counts:
            p = class_counts[label] / total
            gini -= p * p
        return gini

    # -----------------------
    # Split dataset manually
    # -----------------------
    def split(self, X, y, feature_index, threshold):
        left_X, right_X = [], []
        left_y, right_y = [], []

        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return np.array(left_X), np.array(left_y), np.array(right_X), np.array(right_y)

    # -----------------------
    # Find the best split
    # -----------------------
    def best_split(self, X, y):
        best_gini = 1.0
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        # choose subset of features for split (random feature subsampling)
        if self.max_features is None or self.max_features >= n_features:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = self.rng.choice(n_features, size=self.max_features, replace=False)

        # loop over selected features
        for f in feature_indices:
            col = X[:, f]
            # determine candidate thresholds using quantiles to limit count
            # exclude endpoints to avoid empty splits
            if self.max_thresholds >= len(col):
                # fallback to unique values if few
                values = np.unique(col)
                # drop endpoints (min/max) to reduce empty-split risk
                if values.size > 2:
                    cand = values[1:-1]
                else:
                    cand = values
            else:
                qs = np.linspace(0.0, 1.0, num=self.max_thresholds + 2)[1:-1]
                cand = np.quantile(col, qs)

            for t in np.unique(cand):
                X_left, y_left, X_right, y_right = self.split(X, y, f, t)
                if len(y_left) < 1 or len(y_right) < 1:
                    continue

                g_left = self.gini(y_left)
                g_right = self.gini(y_right)
                weighted_gini = (len(y_left) * g_left + len(y_right) * g_right) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = f
                    best_threshold = float(t)

        return best_feature, best_threshold

    # -----------------------
    # Find majority class manually
    # -----------------------
    def majority_class(self, y):
        class_counts = {}
        for label in y:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        max_class = None
        max_count = -1
        for label in class_counts:
            if class_counts[label] > max_count:
                max_class = label
                max_count = class_counts[label]
        try:
            return int(max_class)
        except Exception:
            return max_class

    # -----------------------
    # Recursive tree building
    # -----------------------
    def build_tree(self, X, y, depth=0):
        # stopping conditions
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            return self.majority_class(y)

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return self.majority_class(y)

        X_left, y_left, X_right, y_right = self.split(X, y, feature, threshold)

        node = {
            "feature": int(feature),
            "threshold": float(threshold),
            "left": self.build_tree(X_left, y_left, depth + 1),
            "right": self.build_tree(X_right, y_right, depth + 1)
        }
        return node

    # -----------------------
    # Train the model
    # -----------------------
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # -----------------------
    # Predict single sample
    # -----------------------
    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node

        feature = node["feature"]
        threshold = node["threshold"]

        if x[feature] <= threshold:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    # -----------------------
    # Predict for all samples
    # -----------------------
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predictions.append(self._predict_one(X[i], self.tree))
        return np.array(predictions)


# Example Usage
if __name__ == "__main__":
    X = np.array([
        [2, 3],
        [1, 5],
        [2, 8],
        [6, 8],
        [5, 2],
        [7, 3]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTreeScratch(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)
    print("Predictions:", preds)
