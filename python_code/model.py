from least_squares import *
from least_absolute_deviation import *
from adaptive_ridge import *
from adaptive_lad import *

def model(method, X, y, reg_beta0, reg_V0, alpha, delta, T, intercept_V0):

    if method == 'lad_lasso':
        m, beta, intercept_beta =  lasso_lasso_regression(X, y, reg_beta0)
        return beta, intercept_beta

    if method == 'lad_lasso_CVaR':
        m, beta, intercept_beta =  lasso_lasso_regression_CVaR(X, y, reg_beta0, alpha)
        return beta, intercept_beta

    if method == 'ols_ridge':
        m, beta, intercept_beta =  ridge_regression_standard(X, y, reg_beta0)
        return beta, intercept_beta

    if method == 'ols_ridge_CVaR':
        m, beta, intercept_beta =  ridge_regression_CVaR(X, y, reg_beta0, alpha)
        return beta, intercept_beta

    if method == 'ols_lasso_CVaR':
        m, beta, intercept_beta =  lasso_regression_CVaR(X, y, reg_beta0, alpha)
        return beta, intercept_beta

    if method == 'adaptive_lad_ridge':
        m, beta, V0 = adaptive_lad_regression_standard(X, y, reg_beta0, reg_V0, T, intercept_V0=intercept_V0)
        return beta, V0

    if method == 'adaptive_lad_ridge_CVaR':
        m, beta, V0 = adaptive_lad_regression_CVaR(X, y, reg_beta0, reg_V0, T, alpha, intercept_V0=intercept_V0)
        return beta, V0

    if method == 'adaptive_lad_ridge_slowly_varying':
        m, beta, V0 = adaptive_lad_regression_slowly_varying_l2(X, y, reg_beta0, reg_V0, T, delta)
        return beta, V0

