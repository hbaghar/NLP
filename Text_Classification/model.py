import numpy as np
from tqdm import tqdm
from process_files import ProcessFiles
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# TODO: Cleanup of code, remove loss function warnings
class MyLogisticRegresiion(object):
    """docstring for MyLogisticRegresiion."""

    def __init__(self):
        self.theta = None
        np.random.seed(12)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    # def compute_loss(self, X, y):
    #     return np.mean(-y*np.log(self.sigmoid(np.dot(X, self.theta))) - (1-y)*np.log(1-self.sigmoid(np.dot(X, self.theta))))

    def compute_gradient(self, X, y):
        return (1/X.shape[0])*np.dot(X.T, self.sigmoid(np.dot(X, self.theta)) - y)

    def train(self, X, y, alpha = 0.05, epochs = 250):
        """
        Train logistic regression model
        """
        self.theta = np.r_[np.ones((1,)), np.random.rand(X.shape[1] - 1, )]
        print(X.shape, self.theta.shape, y.shape)

        # cost = np.zeros((epochs, 1))

        for e in tqdm(range(epochs)):
            # cost[e] = self.compute_loss(X, y)
            g = self.compute_gradient(X, y)
            self.theta = self.theta - alpha*g

    def predict(self, X):
        return np.rint(self.sigmoid(np.dot(X, self.theta)))

    def metrics(self, X, y):
        preds = self.predict(X)

        tp = sum([1 for i in range(len(y)) if y[i] == 1 and preds[i] == 1])
        fp = sum([1 for i in range(len(y)) if y[i] == 0 and preds[i] == 1])
        tn = sum([1 for i in range(len(y)) if y[i] == 0 and preds[i] == 0])
        fn = sum([1 for i in range(len(y)) if y[i] == 1 and preds[i] == 0])

        accuracy = sum(preds == y)/len(y)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        print(preds.shape, y.shape)
        print(accuracy, precision, recall, f1)

        print(confusion_matrix(y, preds))

if __name__ == "__main__":
    p = ProcessFiles()
    X, y = p.get_train_data()
    X = np.c_[np.ones((X.shape[0], 1)), X]


    model = MyLogisticRegresiion()
    model.train(X,y)

    test_X, test_y = p.get_test_data()
    test_X = np.c_[np.ones((test_X.shape[0], 1)), test_X]
    model.metrics(test_X, test_y)
