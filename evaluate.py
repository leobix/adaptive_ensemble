from sklearn.metrics import mean_absolute_error
import numpy as np
from load_data import get_X_Z_y
from CVaR import compute_CVaR

def evaluate(X,y,beta,b0):
    pred = np.matmul(X,beta)+np.ones(y.shape[0])*b0
    print(mean_absolute_error(y,pred))
    return pred, mean_absolute_error(y,pred)

def evaluate_adaptive(X, y, beta, V0, T, intercept_V0=True):
    X, Z, y = get_X_Z_y(X, y, T)
    N, P = X.shape

    pred = np.array(
        [sum(X[i, j] * (beta[j] + sum(V0[j, l] * Z[i, l] for l in range(T * P + T + 1))) for j in range(P)) for i in
         range(N)])
    if intercept_V0:
        intercept = np.array([beta[-1] + sum(V0[-1, l] * Z[i, l] for l in range(T * P + T + 1)) for i in range(N)])
    else:
        intercept = np.array([beta[-1]])
    preds = pred + intercept
    try:
        #TODO arrange the scaler
        mae = mean_absolute_error(y * scaler_y.scale_, preds * scaler_y.scale_)
        preds = preds * scaler_y.scale_ + scaler_y.mean_
        # print(mean_absolute_error(y_te[T:],preds*scaler_y.scale_+scaler_y.mean_))
    except:
        mae = mean_absolute_error(y, preds)
    print(mae)
    return preds, mae

def evaluate_method(method, X, y, beta, V0, T, intercept_V0=True):
    if method[:3] == 'ols' or method[:3] == 'lad':
        return evaluate(X,y,beta,V0)
    elif method[:8] == 'adaptive':
        return evaluate_adaptive(X, y, beta, V0, T, intercept_V0=True)


def compute_statistics(method, preds, y_te, alphas, T):
    if method[:8]=='adaptive':
        y_te = y_te[T:]
    print(preds.shape)
    print(y_te.shape)
    errs_1 = np.abs(preds - y_te)
    errs_2 = np.square(preds - y_te)

    print('\n#### Errors LAD ####')
    print("Errs 1.5:", np.sum(errs_1 > 1.5))
    print("Errs 2:", np.sum(errs_1 > 2))
    print("Errs 2.5:", np.sum(errs_1 > 2.5))
    print("Errs 3:", np.sum(errs_1 > 3))

    for alpha in alphas:
        m, obj = compute_CVaR(errs_1, np.array(y_te).reshape(-1)[T:], alpha, None, None, errs=True)
        print("\n#### CVaR L1 ####")
        print("CVaR", alpha, ":", obj)

        m, obj = compute_CVaR(errs_2, np.array(y_te).reshape(-1)[T:], alpha, None, None, errs=True)
        print("\n#### CVaR L2 ####")
        print("CVaR", alpha, ":", obj)
