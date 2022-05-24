from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import time
from load_data import *

def adaptive_lad_regression_standard(X, y, rho_1, rho_2, T, intercept_V0=True):
    '''
    Adaptive LAD with Ridge penalty. Scales fine.
    '''

    # Create model
    m = ConcreteModel()
    X, Z, y = get_X_Z_y(X, y, T)

    N, P = X.shape
    # Add variables
    m.beta0 = Var(range(P + 1))
    m.V0 = Var(range(P + 1), range(T * P + T + 1))
    m.u = Var(range(N), domain=NonNegativeReals)
    # Add objective
    print(Z.shape)
    m.obj = Objective(sense=minimize, expr=1 / N * sum(m.u[i] for i in range(N)) + rho_1 * sum(
        pow(m.beta0[j], 2) for j in range(P)) + rho_2 * sum(
        pow(m.V0[i, j], 2) for i in range(P) for j in range(T * P + T + 1)))

    # Linearization of the objective function
    def reluMaxPlus(m, i):
        pred = sum(X[i, j] * (m.beta0[j] + sum(m.V0[j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P))
        if intercept_V0:
            intercept = m.beta0[P] + sum(m.V0[P, l] * Z[i, l] for l in range(T * P + T + 1))
        else:
            intercept = m.beta0[P]
        return m.u[i] >= y[i] - pred - intercept

    m.relu_max_plus = Constraint(range(N), rule=reluMaxPlus)

    def reluMaxNeg(m, i):
        pred = sum(X[i, j] * (m.beta0[j] + sum(m.V0[j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P))
        if intercept_V0:
            intercept = m.beta0[P] + sum(m.V0[P, l] * Z[i, l] for l in range(T * P + T + 1))
        else:
            intercept = m.beta0[P]
        return m.u[i] >= - y[i] + pred + intercept
        # eturn m.u[i] >= - y[i] + m.beta0[P] + sum(m.V0[P,l]*Z[i,l] for l in range(T*P+T+1))#+ sum(X[i, j]*(m.beta0[j]+sum(m.V0[j,l]*Z[i,l] for l in range(T*P+T+1))) for j in range(P)) #

    m.relu_max_neg = Constraint(range(N), rule=reluMaxNeg)

    solver = SolverFactory('gurobi')
    start_time = time.time()
    ## tee=True enables solver output
    results = solver.solve(m, tee=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    V0 = np.array([[m.V0[j, l].value for l in range(P * T + T + 1)] for j in range(P + 1)])
    return m, np.array([m.beta0[j].value for j in range(P + 1)]), V0


def adaptive_lad_regression_CVaR(X, y, rho_1, rho_2, T, alpha, intercept_V0=True):
    '''
    Adaptive LAD with Ridge penalty + CVaR. Scales fine.
    '''

    # Create model
    m = ConcreteModel()
    X, Z, y = get_X_Z_y(X, y, T)

    N, P = X.shape
    # Add variables
    m.beta0 = Var(range(P + 1))
    m.V0 = Var(range(P + 1), range(T * P + T + 1))
    m.tau = Var()
    m.u = Var(range(N), domain=NonNegativeReals)
    # Add objective
    print(Z.shape)
    m.obj = Objective(sense=minimize, expr=1 / (alpha * N) * sum(m.u[i] for i in range(N)) + m.tau
                                           + rho_1 * sum(pow(m.beta0[j], 2) for j in range(P))
                                           + rho_2 * sum(
        pow(m.V0[i, j], 2) for i in range(P) for j in range(T * P + T + 1)))

    # Linearization of the objective function
    def reluMaxPlus(m, i):
        pred = sum(X[i, j] * (m.beta0[j] + sum(m.V0[j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P))
        if intercept_V0:
            intercept = m.beta0[P] + sum(m.V0[P, l] * Z[i, l] for l in range(T * P + T + 1))
        else:
            intercept = m.beta0[P]
        return m.u[i] >= y[i] - pred - intercept - m.tau

    m.relu_max_plus = Constraint(range(N), rule=reluMaxPlus)

    def reluMaxNeg(m, i):
        pred = sum(X[i, j] * (m.beta0[j] + sum(m.V0[j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P))
        if intercept_V0:
            intercept = m.beta0[P] + sum(m.V0[P, l] * Z[i, l] for l in range(T * P + T + 1))
        else:
            intercept = m.beta0[P]
        return m.u[i] >= - y[i] + pred + intercept - m.tau
        # eturn m.u[i] >= - y[i] + m.beta0[P] + sum(m.V0[P,l]*Z[i,l] for l in range(T*P+T+1))#+ sum(X[i, j]*(m.beta0[j]+sum(m.V0[j,l]*Z[i,l] for l in range(T*P+T+1))) for j in range(P)) #

    m.relu_max_neg = Constraint(range(N), rule=reluMaxNeg)

    solver = SolverFactory('gurobi')
    start_time = time.time()
    ## tee=True enables solver output
    results = solver.solve(m, tee=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    V0 = np.array([[m.V0[j, l].value for l in range(P * T + T + 1)] for j in range(P + 1)])
    return m, np.array([m.beta0[j].value for j in range(P + 1)]), V0


def adaptive_lad_regression_slowly_varying_l2(X, y, rho_1, rho_2, T, delta):
    '''
    Adaptive LAD with Ridge penalty and slowly varying. Scales slow.
    '''
    # Create model
    m = ConcreteModel()
    X, Z, y = get_X_Z_y(X, y, T)

    N, P = X.shape
    # Add variables
    m.beta0 = Var(range(P + 1))
    m.V0 = Var(range(N), range(P + 1), range(T * P + T + 1))
    m.u = Var(range(N), domain=NonNegativeReals)
    # Add objective
    print(Z.shape)
    m.obj = Objective(sense=minimize, expr=1 / N * sum(m.u[i] for i in range(N)) + rho_1 * sum(
        pow(m.beta0[j], 2) for j in range(P)) + 1/N * rho_2 * sum(
        pow(m.V0[i, j, l], 2) for i in range(N) for j in range(P) for l in range(T * P + T + 1)))

    def reluMaxPlus(m, i):
        pred = sum(
            X[i, j] * (m.beta0[j] + sum(m.V0[i, j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P))
        intercept = m.beta0[P] + sum(m.V0[i, P, l] * Z[i, l] for l in range(T * P + T + 1))
        return m.u[i] >= y[i] - pred - intercept

    m.relu_max_plus = Constraint(range(N), rule=reluMaxPlus)

    def reluMaxNeg(m, i):
        pred = sum(
            X[i, j] * (m.beta0[j] + sum(m.V0[i, j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P))
        intercept = m.beta0[P] + sum(m.V0[i, P, l] * Z[i, l] for l in range(T * P + T + 1))
        return m.u[i] >= - y[i] + pred + intercept

    m.relu_max_neg = Constraint(range(N), rule=reluMaxNeg)

    def slowlyVarying(m, i):
        s = sum(pow(m.V0[i, j, l] - m.V0[i - 1, j, l], 2) for l in range(T * P + T + 1) for j in range(P + 1))
        return s <= delta

    m.slowlyVarying = Constraint(range(1, N), rule=slowlyVarying)

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)
    start_time = time.time()
    ## tee=True enables solver output
    results = solver.solve(m, tee=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    # results = solver.solve(m, tee=False)
    V0 = np.array([[[m.V0[i, j, l].value for l in range(P * T + T + 1)] for j in range(P + 1)] for i in range(N)])
    return m, np.array([m.beta0[j].value for j in range(P + 1)]), V0