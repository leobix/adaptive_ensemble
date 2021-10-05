from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

def lasso_lasso_regression(X, y, rho):
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
    m.z = Var(range(n), domain=NonNegativeReals)
    m.beta_abs = Var(range(p), domain=NonNegativeReals)

    # Add objective
    m.obj = Objective(sense=minimize, expr=1 / n * sum(m.z[i] for i in range(n))
                                           + rho * sum(m.beta_abs[j] for j in range(p)))

    def reluMaxPlus(m, i):
        return y[i] - sum(X[i, j] * m.beta[j] for j in range(p)) - m.b0 <= m.z[i]

    m.relu_max_plus = Constraint(range(n), rule=reluMaxPlus)

    def reluMaxNeg(m, i):
        return - y[i] + sum(X[i, j] * m.beta[j] for j in range(p)) + m.b0 <= m.z[i]

    m.relu_max_neg = Constraint(range(n), rule=reluMaxNeg)

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

def lasso_lasso_regression_CVaR(X, y, rho, alpha):
    '''
    Non-adaptive version, that optimizes wrt CVaR, with LAD, and lasso penalty
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

    def reluMaxPlus(m, i):
        return y[i] - sum(X[i, j] * m.beta[j] for j in range(p)) - m.b0 - m.tau <= m.z[i]

    m.relu_max_plus = Constraint(range(n), rule=reluMaxPlus)

    def reluMaxNeg(m, i):
        return - y[i] + sum(X[i, j] * m.beta[j] for j in range(p)) + m.b0 - m.tau <= m.z[i]

    m.relu_max_neg = Constraint(range(n), rule=reluMaxNeg)

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