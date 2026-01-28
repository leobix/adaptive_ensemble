import Pkg
#be careful: implies you created a robust virtual environment
#Pkg.activate("robust")

using JuMP

using LinearAlgebra
using Statistics
using StatsBase
using Gurobi
using ArgParse
using CSV
using Random

include("eval2.jl")
include("utils.jl")
include("synthetic_experiments/utils_synthetic_experiments.jl")


const GRB_ENV = Gurobi.Env()

#parser
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--filename-results"
            help = "filename to save results file"
            default = "results.csv"
            arg_type = String

        "--results_dir"
            help = "directory to save results CSV files"
            default = "results"
            arg_type = String

        "--train_test_split"
            help = "train_test_split value"
            arg_type = Float64
            default = 0.5

        "--data"
            help = "filename of targets y"
            default = "synthetic"
            arg_type = String

        "--past"
            help = "number of max cuts in Benders problem"
            arg_type = Int
            default = 5

        "--num-past"
            help = "number of past"
            arg_type = Int
            default = 200

        "--val"
            help = "number of max cuts in Benders problem"
            arg_type = Int
            default = 365

        "--begin-id"
            help = ""
            arg_type = Int
            default = 1

        "--end-id"
            help = ""
            arg_type = Int
            default = 10

        "--lead_time"
            help = "Lead time in timesteps"
            arg_type = Int
            default = 1

        "--CVAR"
            help = "fix beta0"
            action = :store_true

        "--err_rule"
            help = "Build Z with the regrets instead of the raw history"
            action = :store_true

        "--err_rule_norm"
            help = "Build Z with the regrets instead of the raw history, and normalized regrets step-wise"
            action = :store_true

        "--uncertainty"
            help = "uncertainty percentage parameter"
            arg_type = Float64
            default = 0.0

        "--hedge_eta"
            help = "Hedge (EWA) learning rate; if <= 0 uses default sqrt(8*log(m)/n)"
            arg_type = Float64
            default = 0.0

        "--rls_lambda"
            help = "RLS forgetting factor in (0,1], e.g., 0.99"
            arg_type = Float64
            default = 0.99

        "--epsilon-inf"
            help = "epsilon inf"
            arg_type = Float64
            default = 0.01

        "--delta-inf"
            help = "delta inf"
            arg_type = Float64
            default = 0.01

        "--epsilon-l2"
            help = "epsilon l2"
            arg_type = Float64
            default = 0.01

        "--more_data_for_beta0"
            help = "more_data_for_beta0"
            action = :store_true

        "--last_yT"
            help = "last_yT"
            action = :store_true

        "--delta-l2"
            help = "delta l2"
            arg_type = Float64
            default = 0.01

        "--rho_beta"
            help = "rho for Beta in Adaptive Ridge Linear Decision Rule"
            arg_type = Float64
            default = 0.1

        "--rho_stat"
            help = "rho for statistical error in standard ridge"
            arg_type = Float64
            default = 0.1

        "--rho"
            help = "rho for Beta 0 and standard ridge"
            arg_type = Float64
            default = 0.1

        "--rho_V"
            help = "rho for V0"
            arg_type = Float64
            default = 0.1

        "--period"
            help = "Period for the sine in the simulation"
            arg_type = Int
            default = 1

        "--std_pert"
            help = "Std perturbation for y"
            arg_type = Float64
            default = 0.1

        "--N_models"
            help = "Number of models in the ensemble"
            arg_type = Int
            default = 10

        "--tree_max_depth"
            help = "Decision tree ensembler max depth"
            arg_type = Int
            default = 2

        "--tree_min_leaf"
            help = "Decision tree ensembler minimum samples per leaf"
            arg_type = Int
            default = 5

        "--tree_num_thresholds"
            help = "Number of candidate thresholds per feature for tree splits"
            arg_type = Int
            default = 25

        "--gbrt_estimators"
            help = "GBRT number of boosting stages"
            arg_type = Int
            default = 30

        "--gbrt_lr"
            help = "GBRT learning rate"
            arg_type = Float64
            default = 0.1

        "--gbrt_max_depth"
            help = "GBRT base tree max depth"
            arg_type = Int
            default = 2

        "--gbrt_min_leaf"
            help = "GBRT base tree minimum samples per leaf"
            arg_type = Int
            default = 5

        "--fedformer"
            help = "Enable FEDformer baseline"
            action = :store_true

        "--fedformer_seq_len"
            help = "FEDformer encoder sequence length"
            arg_type = Int
            default = 24

        "--fedformer_label_len"
            help = "FEDformer decoder label length"
            arg_type = Int
            default = 12

        "--fedformer_pred_len"
            help = "FEDformer prediction length"
            arg_type = Int
            default = 1

        "--fedformer_d_model"
            help = "FEDformer model dimension"
            arg_type = Int
            default = 64

        "--fedformer_n_heads"
            help = "FEDformer attention heads"
            arg_type = Int
            default = 4

        "--fedformer_e_layers"
            help = "FEDformer encoder layers"
            arg_type = Int
            default = 2

        "--fedformer_d_layers"
            help = "FEDformer decoder layers"
            arg_type = Int
            default = 1

        "--fedformer_d_ff"
            help = "FEDformer feed-forward dimension"
            arg_type = Int
            default = 128

        "--fedformer_dropout"
            help = "FEDformer dropout"
            arg_type = Float64
            default = 0.1

        "--fedformer_activation"
            help = "FEDformer activation (gelu|relu|linear)"
            arg_type = String
            default = "gelu"

        "--fedformer_moving_avg"
            help = "FEDformer moving average window"
            arg_type = Int
            default = 5

        "--fedformer_freq_mode"
            help = "FEDformer frequency mode (fourier|wavelet)"
            arg_type = String
            default = "fourier"

        "--fedformer_modes"
            help = "FEDformer number of frequency modes / wavelet levels"
            arg_type = Int
            default = 16

        "--fedformer_epochs"
            help = "FEDformer training epochs"
            arg_type = Int
            default = 5

        "--fedformer_batch_size"
            help = "FEDformer batch size"
            arg_type = Int
            default = 16

        "--fedformer_lr"
            help = "FEDformer learning rate"
            arg_type = Float64
            default = 0.001

        "--fedformer_save_preds"
            help = "Save FEDformer predictions to results_beta/"
            action = :store_true

        "--gbrt_num_thresholds"
            help = "GBRT base tree number of thresholds per feature"
            arg_type = Int
            default = 25

        "--T"
            help = "Number of total timesteps available in the train data"
            arg_type = Int
            default = 365

        "--train_length"
            help = "Number of timesteps used in the train data"
            arg_type = Int
            default = 365

        "--num_exp"
            help = "Number of same experiments with different seeds to perform"
            arg_type = Int
            default = 1

        "--seed"
            help = "Seed to use, will change for each exp"
            arg_type = Int
            default = 1

        "--bias_range"
            help = "Bias range for the underlying ensemble members"
            arg_type = Float64
            default = 0.1

        "--std_range"
            help = "Std range for the underlying ensemble members"
            arg_type = Float64
            default = 0.1

        "--bias_drift"
            help = "Error bias drift of the ensemble members"
            arg_type = Float64
            default = 0.1

        "--std_drift"
            help = "Error std drift of the ensemble members"
            arg_type = Float64
            default = 0.1

        "--total_drift_additive"
            help = "Build Z with the regrets instead of the raw history, and normalized regrets step-wise"
            action = :store_true

    end
    return parse_args(s)
end



####TODO ADD SEED
function main()
    args = parse_commandline()
    println("Arguments:")
    for (arg,val) in args
        println("  $arg  =>  $val")
    end

    y = create_y(args["T"], args["period"], args["bias"], args["std_pert"], args["seed"])

    #TODO check the n/2 to split index
    n = size(y)[1]
    split_index = floor(Int,n*args["train_test_split"])

    mean_y = mean(y[1:floor(Int, split_index),:])
    std_y = std(y[1:floor(Int, split_index),:])
    y = (y .- mean_y)./std_y;

    for i=1:args["num_exp"]
        args["seed"] = i
        X_test_adaptive = create_ensemble_values(y, args["N_models"], args["bias_range"], args["std_range"], args["bias_drift"], args["std_drift"], args["total_drift_additive"], args["seed"])

y, N_models, bias_range, std_range, δ_pert, σ_pert, total_drift_additive, y_bias_drift, y_std_drift, seed, p_ber = 1
        if args["end-id"] == -1
            args["end-id"] = size(X_test_adaptive)[2]
        end

        #TODO check assert end id < number of models
        if args["end-id"] > size(X_test_adaptive)[2]
            println("There are only ", size(X_test_adaptive)[2], " ensemble members.")
            args["end-id"] = min(args["end-id"],size(X_test_adaptive)[2])
        end

        X = Matrix(X_test_adaptive)[:,args["begin-id"]:args["end-id"]]
        n, p = size(X)

        X = (X .- mean_y)./std_y;

        if args["lead_time"]>1
            Z = (Z .- mean_y)./std_y
        end
        println("Mean target: ", mean_y, " Std target: ", std_y)

        #TODO code all_past -1

        val = min(args["val"], n-split_index-1)

        if args["lead_time"]>1
            eval_method_hurricane(args, X, Z, y, y, args["train_test_split"], args["past"], args["num-past"], val, mean_y, std_y)
        else
            eval_method(args, X, y, y, args["train_test_split"], args["past"], args["num-past"], val, mean_y, std_y)
        end
        println("Results completed")

    end
end

main()
