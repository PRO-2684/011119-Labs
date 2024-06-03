from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

# Features
continuous_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
discrete_features = {
    "Gender": 2,
    "CALC": 4,
    "FAVC": 2,
    "SCC": 2,
    "SMOKE": 2,
    "family_history_with_overweight": 2,
    "CAEC": 4,
    "MTRANS": 5,
}


# Metrics
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Utility functions for decision tree
def entropy(y):
    """Calculate the entropy of a label array."""
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    """Calculate the Gini impurity of a label array."""
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum([p**2 for p in ps])


def information_gain(y, mask, criterion="entropy"):
    """Calculate the information gain of a split."""
    if criterion == "entropy":
        criterion_func = entropy
    else:
        criterion_func = gini

    n = len(y)
    n_l, n_r = np.sum(mask), np.sum(~mask)
    if n_l == 0 or n_r == 0:
        return 0
    e_l, e_r = criterion_func(y[mask]), criterion_func(y[~mask])
    e = criterion_func(y)
    return e - (n_l / n) * e_l - (n_r / n) * e_r


def best_split(x, y, criterion="entropy"):
    """Find the best split for the data."""
    best_gain = 0
    split_index, split_threshold = None, None
    for i in range(x.shape[1]):
        thresholds, classes = zip(*sorted(zip(x[:, i], y)))
        for j in range(1, len(thresholds)):
            if classes[j] == classes[j - 1]:
                continue
            mask = x[:, i] <= thresholds[j]
            gain = information_gain(y, mask, criterion)
            if gain > best_gain:
                best_gain = gain
                split_index = i
                split_threshold = thresholds[j]
    return split_index, split_threshold


# Decision Tree Node
@dataclass
class Node:
    """A class representing a single node in the decision tree."""
    gini: float
    num_samples: int
    num_samples_per_class: list
    predicted_class: int
    feature_index: int = 0
    threshold: float = 0.0
    left: Optional['Node'] = None
    right: Optional['Node'] = None


# Decision Tree Classifier
class DecisionTreeClassifier:
    """A basic implementation of a decision tree classifier."""

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, x, y):
        """Fit the decision tree to the data."""
        self.n_classes = len(set(y))
        self.tree = self._grow_tree(x.values, y)

    def _grow_tree(self, x, y, depth=0):
        """Recursively build the decision tree."""
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            index, threshold = best_split(x, y)
            if index is not None:
                indices_left = x[:, index] <= threshold
                x_left, y_left = x[indices_left], y[indices_left]
                x_right, y_right = x[~indices_left], y[~indices_left]
                node.feature_index = index
                node.threshold = threshold
                node.left = self._grow_tree(x_left, y_left, depth + 1)
                node.right = self._grow_tree(x_right, y_right, depth + 1)
        return node

    def predict(self, x):
        """Predict the class labels for the input samples."""
        return [self._predict(inputs) for inputs in x.values]

    def _predict(self, inputs):
        """Predict the class label for a single sample."""
        node = self.tree
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


def load_data(datapath: str = "./data/ObesityDataSet_raw_and_data_sinthetic.csv"):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(datapath)
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    # Encode discrete str to number, e.g. male&female to 0&1
    encoder = LabelEncoder()
    for col in discrete_features.keys():
        x[col] = encoder.fit_transform(x[col])
    y = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(
        "./data/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("Accuracy:", accuracy(y_test, y_pred))
