#modified from this example:
#https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/
#get the input data from repo bayesian_logistic_regression
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression


# Logistic Regression
class LogitRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape #m is numer of data points n is numer of features
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):
        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        # calculate gradients
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )

    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y


# Driver code

def main():
    # Importing dataset
    df = pd.read_csv("../data/Bay_LR_Covid/covid_updated.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    # Model training
    model_scratch = LogitRegression(learning_rate=0.01, iterations=1000)

    model_scratch.fit(X, y)
    model_sklearn = LogisticRegression()
    model_sklearn.fit(X, y)

    # Prediction on train set
    y_pred_scratch = model_scratch.predict(X)
    y_pred_sklearn = model_sklearn.predict(X)


    print("Accuracy on train set by built-from-scratch model:",
          round(float((np.dot(y.flatten(), y_pred_scratch) +
                 np.dot(1 - y.flatten(), 1 - y_pred_scratch)) / float(y.size) * 100),2)
          )

    print("Accuracy on train set by sklearn model:",
          round(float((np.dot(y.flatten(), y_pred_sklearn) +
                 np.dot(1 - y.flatten(), 1 - y_pred_sklearn)) / float(y.size) * 100),2)
          )

if __name__ == "__main__":
    main()