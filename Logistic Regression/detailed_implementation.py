import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def sigmoid(X):
    return 1 / (1 + np.exp(- X))

def cost(theta, X, y):
    p_1 = sigmoid(np.dot(X, theta)) 
    log_l = (-y)*np.log(p_1) - (1-y)*np.log(1-p_1) 

    return log_l.mean()

def grad(theta, X, y):
    """The gradient of the cost function for logistic regresssion"""
    p_1 = sigmoid(np.dot(X, theta))
    error = p_1 - y 
    grad = np.dot(error, X) / y.size 

    return grad

def train(X, y):
    """Train a logistic regression model for data X and labels y.
    returns the learned parameter.
    """
    theta = 0.1*np.random.randn(len(X[0]))
    theta_best = opt.fmin_bfgs(cost, theta, fprime=grad, args=(X, y))
    return theta_best

def predict(theta_best, Xtest):
    """Using the learned parameter theta_best, predict on data Xtest"""
    p = sigmoid(theta_best.dot(Xtest.T))
    for i in range(len(p)):
        if p[i] > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def confusion_matrix(prediction, labels):
    assert len(prediction) == len(labels)
    def f(p, l):
        n = 0
        for i in range(len(prediction)):
            if prediction[i] == p and labels[i] == l:
                n += 1
        return n
    return np.matrix([[f(1, 1), f(1, 0)], [f(0, 1), f(0, 0)]])

def confusion_matrix_advanced(prediction, labels):
    assert len(prediction) == len(labels)
    f = lambda p, l: len(list(filter(lambda x: x == (p, l), zip(prediction, labels))))
    return np.matrix([[f(1, 1), f(1, 0)], [f(0, 1), f(0, 0)]])

def accuracy(cmatrix):
    tp, fp, fn, tn = cmatrix.flatten().tolist()[0]
    return (tp + tn) / (tp + fp + fn + tn)

def balanced_accuracy(cmatrix):
    """Returns the balanced accuracy of a confusion matrix"""
    tp, fp, fn, tn = cmatrix.flatten().tolist()[0]
    return tp / 2 / (tp + fn) + tn / 2 / (tn + fp)


names = ['medv', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
data = np.loadtxt('housing_scale.csv', delimiter=',')
to_drop = names.index('chas')
data = np.delete(data, to_drop, axis=1)

data[:,0] = (data[:, 0]>27)

y = data[:,0]
X = data[:,1:]
theta_best = train(X, y)
pred = predict(theta_best, X)
cmatrix = confusion_matrix(pred, y)
print(accuracy(cmatrix))
print(balanced_accuracy(cmatrix))
