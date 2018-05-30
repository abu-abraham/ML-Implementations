'''Code is a copy of the COMP8600 tutorial code'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class RLSPrimal(object):
    """Primal Regularized Least Squares"""
    def __init__(self, reg_param):
        self.reg_param = reg_param
        self.w = np.array([])   # This should be the number of features long
    
    def train(self, X, y):
        """Find the maximum likelihood parameters for the data X and labels y"""
        Phi = feature_map(X)
        num_ex = (Phi.T).shape[0]
        A = np.dot(Phi.T, Phi) + self.reg_param * np.eye(num_ex)
        b = np.dot(Phi.T, y)
        self.w = np.linalg.solve(A, b)
    
    def predict(self, X):
        """Assume trained. Predict on data X."""
        Phi = feature_map(X)
        return np.dot(Phi, self.w)

class RLSDual(object):
    def __init__(self, reg_param):
        self.reg_param = reg_param
        self.a = np.array([])    # This should be number of examples long

    def train(self, K, y):
        """Find the maximum likelihood parameters for the kernel matrix K and labels y"""
        num_ex = K.shape[0]
        A = K + self.reg_param * np.eye(num_ex)
        self.a = np.linalg.solve(A, y)
    
    def predict(self, K):
        """Assume trained. Predict on test kernel matrix K."""
        return np.dot(K, self.a)
    
def phi_quadratic(x):
    """Compute phi(x) for a single training example using quadratic basis function."""
    D = len(x)
    # Features are (1, {x_i}, {cross terms and squared terms})
    # Cross terms x_i x_j and squared terms x_i^2 can be taken from upper triangle of outer product of x with itself
    return np.hstack((1, np.sqrt(2)*x, np.sqrt(2)*np.outer(x, x)[np.triu_indices(D,k=1)], x ** 2))

def feature_map(X):
    """Return the matrix of the feature map."""
    num_ex, num_feat = X.shape
    Phi = np.zeros((num_ex, int(1+num_feat+num_feat*(num_feat+1)/2)))
    for ix in range(num_ex):
        Phi[ix,:] = phi_quadratic(X[ix,:])
    return Phi

def dotprod_quadratic(X):
    """Compute the kernel matrix using an explicit feature map of
    the inhomogeneous polynomial kernel of degree 2"""
    Phi = feature_map(X)
    return np.dot(Phi, Phi.T)

def kernel_quadratic(X,Y):
    """Compute the inhomogenous polynomial kernel matrix of degree 2"""
    lin_dot = np.dot(X,Y.T)
    dotprod = (lin_dot+1.)*(lin_dot + 1.)
    return dotprod
def split_data(data):
    """Randomly split data into two equal groups"""
    np.random.seed(1)
    N = len(data)
    idx = np.arange(N)
    np.random.shuffle(idx)
    train_idx = idx[:int(N/2)]
    test_idx = idx[int(N/2):]

    # Assume label is in the first column
    X_train = data[train_idx, 1:]
    t_train = data[train_idx, 0]
    X_test = data[test_idx, 1:]
    t_test = data[test_idx, 0]
    
    return X_train, t_train, X_test, t_test

def rmse(label, prediction):
    N = len(label)
    return np.linalg.norm(label - prediction) / np.sqrt(N)
names = ['medv', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
data = np.loadtxt('housing_scale.csv', delimiter=',')
X_train, t_train, X_test, t_test = split_data(data)
P = RLSPrimal(1.1)
P.train(X_train, t_train)
pP = P.predict(X_test)

K_train = kernel_quadratic(X_train, X_train)
K_test = kernel_quadratic(X_test, X_train)   # This is not square
D = RLSDual(1.1)
D.train(K_train, t_train)
pD = D.predict(K_test)
print('RMSE: primal = %f, dual = %f' % (rmse(t_test, pP), rmse(t_test, pD)))
