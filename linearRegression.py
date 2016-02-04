import numpy as np
import numpy.linalg as npla
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__

class linearRegression:
    X, y, theta = 0, 0,0

    def predict(self, X):
        #This function will return prediction vector

    def costFunction(self):
        #evaluates the costFunction of the object right now

    def featureNormalize(self):
        # should normalize the feature except the bias term


    def gradientDescent(self, alpha, num_iter):
        #runs gradient descent on the current object to minimize the cost function

    def normalEquation(self):
        #solves using normal equations
    
    def plot(self):
        #plot the regression model and the dataset
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X[:, 1] ,self.X[:,2], self.y);
        x = np.linspace(min(self.X[:,1]), max(self.X[:,1]), 100)
        y = np.linspace(min(self.X[:,2]), max(self.X[:,2]), 100)

        z = x*self.theta[1] + y*self.theta[2] + self.theta[0]
        ax.plot(x,y,z)

        plt.show()

    def __init__(self, dataset_name):
        #load data from the dataset(Should be a string filename)
   
   @classmethod
    def trainNE(cls, dataset_name): #initializes the lin reg to run with normal equatioins
        lin_reg = cls(dataset_name)
        lin_reg.normalEquation()
        return lin_reg

    @classmethod
    def trainGD(cls, dataset_name, alpha, num_iter=400): # initializes the lin reg to run with grad descent
        lin_reg = cls(dataset_name)
        lin_reg.featureNormalize()
        lin_reg.gradientDescent(alpha, num_iter)
        return lin_reg

lin_reg = linearRegression.trainGD("ex1.txt", 0.01)
lin_reg = linearRegression.trainNE("ex1.txt")
lin_reg.plot()
