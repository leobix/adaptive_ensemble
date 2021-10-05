import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path_X,path_y,test_size=0.5,standardize_X=False, standardize_y=False):
    X_test_adaptive = pd.read_csv(path_X, index_col = 'Unnamed: 0')
    y_test = pd.read_csv(path_y, index_col = 'Unnamed: 0')

    if path_X=="traffic_predictions_test_val.csv":
        X_test_adaptive = X_test_adaptive.drop(columns = ['DummyRegressor', 'Lasso', 'LassoLars'])
    elif path_X=="data/X_test_adaptive.csv":
        X_test_adaptive = X_test_adaptive.drop(
            columns=['RANSACRegressor', 'GaussianProcessRegressor', 'KernelRidge', 'Lars', 'AdaBoostRegressor',
                     'DummyRegressor', 'ExtraTreeRegressor', 'Lasso', 'LassoLars', 'PassiveAggressiveRegressor'])

    X_t, X_te, y_t, y_te = train_test_split(X_test_adaptive, y_test, test_size = test_size, shuffle = False, random_state = 6)

    if standardize_X:
        scaler_X = StandardScaler()
        scaler_X.fit(X_t)
        X_t = scaler_X.transform(X_t)
        X_te = scaler_X.transform(X_te)

    scaler_y = StandardScaler()
    if standardize_y:
        scaler_y.fit(y_t)
        y_t = scaler_y.transform(y_t)
        y_te = scaler_y.transform(y_te)

    return np.array(X_t), np.array(X_te), np.array(y_t).reshape(-1), np.array(y_te).reshape(-1), scaler_y


def get_X_Z_y(X, y, T):
    '''
    Input: training data X and corresponding labels y ; how many time-steps from the past to be used
    Output: the past features X with past targets y as a Z training data (no present features)
    '''
    n, p = X.shape
    #T past time steps * p features + T targets + intercept term
    Z = np.ones((n-T, T*p+T+1))
    for i in range(T, n):
        for t in range(T):
            Z[i-T,p*t:p*(t+1)] = X[i-t-1]
        Z[i-T, p*T:-1] = y[i-T:i]
    return X[T:], Z, y[T:]

def get_X_Z_y_standard_regression(X, y, T):
    '''
    Input: training data X and corresponding labels y ; how many time-steps from the past to be used
    Output: the past features X with past targets y as a Z training data + present features
    '''
    n, p = X.shape
    # T past time steps * p features + p present features + T targets + intercept term
    Z = np.ones((n-T, (T+1)*p+T+1))
    for i in range(T, n):
        for t in range(T+1):
            Z[i-T,p*t:p*(t+1)] = X[i-t]
        Z[i-T, p*(T+1):-1] = y[i-T:i]
    return X[T:], Z, y[T:]