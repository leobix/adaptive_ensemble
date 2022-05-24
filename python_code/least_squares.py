from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

def ridge_regression_standard(X, y, rho):
    '''
    Least Squares with Ridge penalty
    :param X: data
    :param y: target
    :param rho: reg factor
    :return: beta values and intercept term
    '''

    n, p = X.shape

    # Create model
    m = ConcreteModel()

    # Add variables
    m.beta = Var(range(p))
    m.b0 = Var()

    # Add objective
    m.obj = Objective(sense=minimize, expr=1 / n * sum(
        pow(y[i] - sum(X[i, j] * m.beta[j] for j in range(p)) - m.b0, 2)
        for i in range(n)) + rho * sum(pow(m.beta[j], 2) for j in range(p)))

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)

    ## tee=True enables solver output
    # results = solver.solve(m, tee=True)
    results = solver.solve(m, tee=False)
    return m, np.array([m.beta[j].value for j in range(p)]), m.b0.value


def ridge_regression_CVaR(X, y, rho, alpha):
    '''
    Non-adaptive version, that optimizes wrt CVaR, with OLS, and Ridge penalty
    alpha: Risk value
    '''
    n, p = X.shape

    # Create model
    m = ConcreteModel()

    # Add variables
    m.beta = Var(range(p))
    m.b0 = Var()
    m.tau = Var()
    m.z = Var(range(n), domain=NonNegativeReals)

    # Add objective
    m.obj = Objective(sense=minimize, expr=1 / (alpha * n) * sum(m.z[i] for i in range(n)) + m.tau
                                           + rho * sum(pow(m.beta[j], 2) for j in range(p)))

    def reluMax(m, i):
        return pow(y[i] - sum(X[i, j] * m.beta[j] for j in range(p)) - m.b0, 2) - m.tau <= m.z[i]

    m.relu_max = Constraint(range(n), rule=reluMax)

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)

    ## tee=True enables solver output
    results = solver.solve(m, tee=False)

    return m, np.array([m.beta[j].value for j in range(p)]), m.b0.value


def lasso_regression_CVaR(X, y, rho, alpha):
    '''
    Non-adaptive version, that optimizes wrt CVaR, with OLS, and lasso penalty
    alpha: Risk value
    '''
    n, p = X.shape

    # Create model
    m = ConcreteModel()

    # Add variables
    m.beta = Var(range(p))
    m.b0 = Var()
    m.tau = Var()
    m.z = Var(range(n), domain=NonNegativeReals)
    m.beta_abs = Var(range(p), domain=NonNegativeReals)

    # Add objective
    m.obj = Objective(sense=minimize, expr=1 / (alpha * n) * sum(m.z[i] for i in range(n)) + m.tau
                                           + rho * sum(m.beta_abs[j] for j in range(p)))

    def reluMax(m, i):
        return pow(y[i] - sum(X[i, j] * m.beta[j] for j in range(p)) - m.b0, 2) - m.tau <= m.z[i]

    m.relu_max = Constraint(range(n), rule=reluMax)

    def betaAbsPlus(m, i):
        return m.beta[i] <= m.beta_abs[i]

    m.beta_abs_plus = Constraint(range(p), rule=betaAbsPlus)

    def betaAbsNeg(m, i):
        return - m.beta[i] <= m.beta_abs[i]

    m.beta_abs_neg = Constraint(range(p), rule=betaAbsNeg)

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)

    ## tee=True enables solver output
    results = solver.solve(m, tee=False)

    return m, np.array([m.beta[j].value for j in range(p)]), m.b0.value