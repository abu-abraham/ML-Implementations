
import pandas as pd
import sklearn
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from math import sqrt



data_frame = pd.read_csv('housing_scale.csv', names = ['medv', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'])
data_frame.drop(data_frame.columns[5], axis=1)

T = data_frame['medv']
X = data_frame.drop('medv',axis=1)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, T, test_size = 0.33, random_state = 5)


lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(sqrt(mse))