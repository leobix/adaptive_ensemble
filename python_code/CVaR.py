from pyomo.environ import *
from pyomo.opt import SolverFactory

def compute_CVaR(X, y, alpha, beta, b0, errs=False):
    '''
    Compute the Conditional Value at Risk
    :param X: Either the data matrix X or the errors
    :param y: the target values
    :param alpha: the risk value
    :param beta:
    :param b0:
    :param errs: whether you want to input the data matrix X or the errors
    :return: the model and the CVaR
    '''
    n = len(X)
    # Create model
    m = ConcreteModel()

    if errs:
        def s_init(m, i):
            return X[i]

        m.errs = Param(range(n), initialize=s_init)
    else:
        n, p = X.shape

        def s_init(m, i):
            return pow(y[i] - sum(X[i, j] * beta[j] for j in range(p)) - b0, 2)

        m.errs = Param(range(n), initialize=s_init)
    # Add variables
    m.tau = Var()
    m.z = Var(range(n), domain=NonNegativeReals)

    # Add objective
    m.obj = Objective(sense=minimize, expr=1 / (alpha * n) * sum(m.z[i] for i in range(n)) + m.tau)

    def reluMax(m, i):
        return - m.tau + m.errs[i] <= m.z[i]

    m.relu_max = Constraint(range(n), rule=reluMax)

    solver = SolverFactory('gurobi')  # 'ipopt', executable=executable)

    ## tee=True enables solver output
    results = solver.solve(m, tee=False)

    return m, value(m.obj)