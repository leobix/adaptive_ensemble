#!/usr/bin/env python3

import argparse
import warnings
warnings.filterwarnings('ignore')


from evaluate import *
from model import *
from load_data import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--T", type=int, default=2,
                            help="number of past time steps to use")

parser.add_argument("--steps-out", type=int, default=3,
                            help="time step in the future to predict")
#TODO: compute different alphas
parser.add_argument("--alpha", type=float, default=0.15,
                            help="alpha value to compute")

parser.add_argument("--test_size", type=float, default=0.5,
                            help="alpha value to compute")

parser.add_argument("--reg_beta0", type=float, default=0.1,
                            help="regularizer for beta coefs or beta_0 coefs")

parser.add_argument("--reg_V0", type=float, default=1,
                            help="regularizer for beta coefs or beta_0 coefs")

parser.add_argument("--delta", type=float, default=0.01,
                            help="regularizer for beta coefs or beta_0 coefs")

parser.add_argument("--path_X", type=str, default='traffic_predictions_test_val.csv',
                            help="filename for X data")

parser.add_argument("--path_y", type=str, default='traffic_test_val_scaled.csv',
                            help="filename for X data")

parser.add_argument("--method", type=str, default='lad_lasso',
                            help="filename for Trees")

parser.add_argument("--standardize_X", action='store_true',
                            help="regression models will run")

parser.add_argument("--standardize_y", action='store_true',
                            help="for data unbalance")




def main(args):
    X_t, X_te, y_t, y_te, scaler_y = load_data(args.path_X, args.path_y, args.test_size, args.standardize_X, args.standardize_y)
    alphas = [0.15, 0.05]
    intercept_V0 = True
    beta, V0 = model(args.method, X_t, y_t, args.reg_beta0, args.reg_V0, args.alpha, delta = args.delta, T = args.T, intercept_V0 = intercept_V0)

    preds, mae = evaluate_method(args.method, X_te, y_te, beta, V0, args.T, intercept_V0)
    compute_statistics(args.method, preds, y_te, alphas, args.T)



if __name__ == "__main__":
   args = parser.parse_args()
   print(args)

   main(args)