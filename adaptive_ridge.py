from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import time
from load_data import *

def ridge_regression_adaptive_alpha_CVaR(X, y, rho, rho_a, rho_a_diff, alpha, T):
    '''
    Adaptive version, that optimizes wrt CVaR, with OLS, and Ridge penalty
    alpha: Risk value
    rho: Ridge penalty
    rho_a: ridge penalty for alpha adaptive

    '''
    N, P = X.shape

    # Create model
    m = ConcreteModel()

    # Add variables
    m.beta = Var(range(P))
    m.b0 = Var()
    m.alpha = Var(range(N), range(P + 1))
    m.tau = Var()
    m.z = Var(range(N), domain=NonNegativeReals)

    # Add objective
    m.obj = Objective(sense=minimize, expr=1 / (alpha * N) * sum(m.z[i] for i in range(N)) + m.tau
                                           + rho * sum(pow(m.beta[j], 2) for j in range(P))
                                           + rho_a / N * sum(
        sum(pow(m.alpha[i, p], 2) for p in range(P + 1)) for i in range(N - T))
                                           + rho_a_diff / N * sum(
        sum(
            sum(
                pow(m.alpha[i, p] - m.alpha[i - k, p], 2)
                for p in range(P + 1))
            for k in range(1, T))
        for i in range(T, N))
                      )

    def reluMax(m, i):
        return pow(y[i] - sum(X[i, j] * (m.beta[j] + m.alpha[i, j]) for j in range(P)) - m.b0 - m.alpha[i, P],
                   2) - m.tau <= m.z[i]

    m.relu_max = Constraint(range(N), rule=reluMax)

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)

    ## tee=True enables solver output
    results = solver.solve(m, tee=False)

    return m, np.array([m.beta[j].value for j in range(P)]), m.b0.value


def adaptive_ridge_regression_standard(X, y, rho_beta0, rho_V0, T):
    '''
    Adaptive ridge: does not run fast
    TODO Misses the V0 reg
    '''

    # Create model
    m = ConcreteModel()
    X, Z, y = get_X_Z_y(X, y, T)

    N, P = X.shape
    # Add variables
    m.beta0 = Var(range(P + 1))
    m.V0 = Var(range(P + 1), range(T * P + T + 1))
    m.t = Var(domain=NonNegativeReals)
    # Add objective
    print(Z.shape)
    m.obj = Objective(sense=minimize, expr=m.t + rho_beta0 * sum(pow(m.beta0[j], 2) for j in range(P))
                                           + rho_V0 * sum(
        pow(m.V0[i, j], 2) for i in range(P) for j in range(T * P + T + 1))
                      )

    m.quadratic = Constraint(expr=m.t >= 1 / N * sum(
        pow(y[i] -
            sum(
                X[i, j] *
                (m.beta0[j]
                 + sum(m.V0[j, l] * Z[i, l] for l in range(T * P + T + 1))
                 # +np.matmul(m.V0,Z[i].reshape(97,1))[j]
                 )
                for j in range(P))
            - m.beta0[P] - sum(m.V0[P, l] * Z[i, l] for l in range(T * P + T + 1))
            , 2)
        for i in range(N)))

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)
    start_time = time.time()
    ## tee=True enables solver output
    results = solver.solve(m, tee=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    # results = solver.solve(m, tee=False)
    V0 = np.array([[m.V0[j, l].value for l in range(P * T + T + 1)] for j in range(P + 1)])
    return m, np.array([m.beta0[j].value for j in range(P + 1)]), V0