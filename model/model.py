from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn


class MLModel:
    def __init__(self, type, test_train_ratio, trees, layer_sizes, activation, solver, batch_size, learning_rate):
        data_path = "Data Storage\\ML Ready Data\\"
        X = pd.read_csv(data_path+"input.csv").fillna(0)
        Y = pd.read_csv(data_path+"labels.csv")
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_train_ratio)
        if type == "forest":
            self.model = RandomForestClassifier(n_estimators=trees)
        if type == "perceptron":
            self.model = Perceptron()
        if type == "neuralnet":
            self.model = MLPClassifier(hidden_layer_sizes=layer_sizes, activation=activation, solver=solver, batch_size=batch_size, learning_rate_init=learning_rate)

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)
        prediction = self.model.predict(self.X_test)
        self.generate_confusion_matrix(self.Y_test, prediction)


    @staticmethod
    def generate_confusion_matrix(GT, prediction):
        print(classification_report(GT, prediction))
        # cm = confusion_matrix(GT, prediction)
        # plt.figure(figsize=(10, 7))
        # sn.heatmap(cm, annot=True)
        # plt.xlabel("Predicted")
        # plt.ylabel("Ground Truth")
        # plt.title(title)
        # plt.plot()
        # plt.show()