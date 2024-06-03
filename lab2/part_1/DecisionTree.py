from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


# Metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)


# Model
class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.tree = None

    def fit(self, x, y):
        # x: [n_samples_train, n_features],
        # y: [n_samples_train, ],
        # TODO: implement decision tree algorithm to train the model
        pass

    def predict(self, x):
        # x: [n_samples_test, n_features],
        # return: y: [n_samples_test, ]
        y = np.zeros(x.shape[0])
        # TODO:
        return y


def load_data(datapath: str = "./data/ObesityDataSet_raw_and_data_sinthetic.csv"):
    df = pd.read_csv(datapath)
    continue_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    discrete_features = [
        "Gender",
        "CALC",
        "FAVC",
        "SCC",
        "SMOKE",
        "family_history_with_overweight",
        "CAEC",
        "MTRANS",
    ]
    discrete_features_size = {
        "Gender": 2,
        "CALC": 4,
        "FAVC": 2,
        "SCC": 2,
        "SMOKE": 2,
        "family_history_with_overweight": 2,
        "CAEC": 4,
        "MTRANS": 5,
    }

    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    # Encode discrete str to number, e.g. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        x[col] = labelencoder.fit(x[col]).transform(x[col])
    y = labelencoder.fit(y).fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(
        "./data/ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(accuracy(y_test, y_pred))
