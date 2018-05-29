'''Learnt as part of COMP8600 course at ANU'''
import numpy as np

def w_ml_unregularised(Phi, t):
    return np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T), t)

def phi_quadratic(x):
    """phi(x) for a single training example using quadratic basis function."""
    D = len(x)
    return np.hstack((1, x, np.outer(x, x)[np.triu_indices(D)]))

def w_ml_regularised(Phi, t, _lambda):
    return np.dot(np.dot(np.linalg.inv(_lambda * np.eye(Phi.shape[1]) + np.dot(Phi.T, Phi)), Phi.T), t)

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

def rmse(X_train, t_train, X_test, t_test, w):

    N_train = len(X_train)
    N_test = len(X_test)

    # Training set error
    t_train_pred = np.dot(X_train, w)
    rmse_train = np.linalg.norm(t_train_pred - t_train) / np.sqrt(N_train)

    # Test set error
    t_test_pred = np.dot(X_test, w)
    rmse_test = np.linalg.norm(t_test_pred - t_test) / np.sqrt(N_test)

    return rmse_train, rmse_test

names = ['medv', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
data = np.loadtxt('housing_scale.csv', delimiter=',')
to_drop = names.index('chas')
data = np.delete(data, to_drop, axis=1)

X_train, t_train, X_test, t_test = split_data(data)
w_unreg = w_ml_unregularised(X_train, t_train)
print(rmse(X_train, t_train, X_test, t_test, w_unreg)[1])