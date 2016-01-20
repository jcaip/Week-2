import numpy as np
import matplotlib as mpl

class linearRegression:
    X, y, theta = 0, 0,0

    def __init__(self, dataName):
        #load data from the dataset(Should be a string)
        self.X =  np.loadtxt(dataName)[:, [0, 1]]
        self.y =  np.loadtxt(dataName)[:, 2]
        self.theta = np.zeros(X.shape(axis=2)).T

    def train(self, use_norm):
        if use_norm:
            __normalEquation()

    def predict(self):
        return self.theta*self.X

    def __costFunction(self, X ,y):
        #write the cost function
        m = len(y)
        J = ((theta.dot(X) - y) ** 2).sum() / 2*m
        return J

    def __featureNormalize(self, X):
        mu = np.zeros(1, X.shape)

    def __gradientDescent(self, alpha, num_iter):
        m = len(self.y)
        print("Curent theta - " + str(self.theta))
        for i in range(0,num_iter):
            theta = theta - (alpha/m)*X.T.dot(

    def __normalEquation(self):
        pass

    def plot(self):
        con
