#Note: this file is mostly the same as main.jl but is more convenient to hypertune in a given range of features to batch jobs

import Pkg

using JuMP

using LinearAlgebra
using Statistics
using StatsBase
using DataFrames
using Gurobi
using ArgParse
using CSV
using Random

#NOTE: IMPORTANT Change from eval2.jl to eval_hurricane.jl for the specific hurricane dataset that requires adjustment in building the Z matrix
# (since we skip 4 timesteps ahead instead of 1 and want to avoid cheating in hindsight, see paper if interested)
include("eval2.jl")
include("utils.jl")
include("utils_hurricane.jl")


const GRB_ENV = Gurobi.Env()

#parser
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--filename-results"
            help = "filename to save results file"
            default = "results.csv"
            arg_type = String

        "--filename-X"
            help = "filename of data X"
            default = "data/X_toy_test.csv"
            arg_type = String

        "--filename-y"
            help = "filename of targets y"
            default = "data/y_toy_test.csv"
            arg_type = String

        "--data"
            help = "filename of targets y"
            default = "energy"
            arg_type = String

        "--train_test_split"
            help = "train_test_split value"
            arg_type = Float64
            default = 0.5

        "--past"
            help = "window size to take into account for the past"
            arg_type = Int
            default = 10

        "--num-past"
            help = "number of past"
            arg_type = Int
            default = 100

        "--val"
            help = "number of max cuts in Benders problem"
            arg_type = Int
            default = 10

        "--begin-id"
            help = ""
            arg_type = Int
            default = 2

        "--end-id"
            help = ""
            arg_type = Int
            default = -1

        "--lead_time"
            help = "Lead time for predictions in the future. Typically 1, i.e., predict next timestep."
            arg_type = Int
            default = 1

        "--last_yT"
            help = "last_yT"
            action = :store_true

        "--more_data_for_beta0"
            help = "more_data_for_beta0"
            action = :store_true

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

        "--reg"
            help = "Deprecated. Regularization factor for the adaptive term"
            arg_type = Float64
            default = 0.1

        "--train_length"
            help = "Number of timesteps used in the train data. Here by default for later use in other functions. useless as an argument."
            arg_type = Int
            default = 0

        "--param_combo"
            help = "hyperparameters combo to be tested"
            arg_type = Int
            default = 0


    end
    return parse_args(s)
end

function main()
    args = parse_commandline()

    rho_beta_list = [0, 0.001, 0.01, 0.1, 1]
    rho_list = [0, 0.001, 0.01, 0.1, 1]
    rho_V_list = [0, 0.001, 0.01, 0.1, 1]
    past_list = [2, 3, 4, 5]
    hyper_number = size(rho_beta_list)[1]*size(rho_list)[1]*size(rho_V_list)[1]*size(past_list)[1]
    hyperparam_combos = zeros(hyper_number, 4)
    acc = 1
    for r_beta in  rho_beta_list
        for rho in rho_list
            for rho_V in rho_V_list
                for past in past_list
                    hyperparam_combos[acc,:] = [r_beta, rho, rho_V, past]
                    acc += 1
                end
            end
        end
    end

    combo = hyperparam_combos[args["param_combo"],:]
    args["rho_beta"] = combo[1]
    args["rho"] = combo[2]
    args["rho_V"] = combo[3]
    args["past"] = Int(combo[4])

    println("Arguments:")
    for (arg,val) in args
        println("  $arg  =>  $val")
    end

    if args["data"] == "mydata"
        X_test_adaptive = DataFrame(CSV.File(args["filename-X"]))
        y_test = Matrix(DataFrame(CSV.File(args["filename-y"])))
    end

    if args["data"] == "energy"
        X_test_adaptive = DataFrame(CSV.File("data/energy_predictions_test_val2.csv"))
        y_test = Matrix(DataFrame(CSV.File("data/energy_y_test_val.csv")))
    end

    if args["data"] == "safi_speed"
        X_test_adaptive = DataFrame(CSV.File("data/X_test_speed_adaptive_out1.csv"))
        y_test = DataFrame(CSV.File("data/y_test_speed_adaptive_out1.csv"))
        y_test = y_test[!, "speed"]
    end

    if args["data"] == "safi_speed_3"
        X_test_adaptive = DataFrame(CSV.File("data/X_test_adaptive.csv"))
        y_test = DataFrame(CSV.File("data/y_test_speed.csv"))
        y_test = y_test[!, "speed"]
        X_test_adaptive = Matrix(X_test_adaptive)[:,1:args["end-id"]]
        X_test_adaptive[:,1] .= 1
        X_test_adaptive, Z, y_test = get_X_Z_y(args, X_test_adaptive, y_test[:, 1], args["past"])
    end

    if args["data"] == "safi_cos"
        X_test_adaptive = DataFrame(CSV.File("data/X_test_cos_adaptive_out1.csv"))
        y_test = DataFrame(CSV.File("data/y_test_cos_adaptive_out1.csv"))
        y_test = y_test[!, "cos_wind_dir"]
    end

    if args["data"] == "safi_sin"
        X_test_adaptive = DataFrame(CSV.File("data/X_test_sin_adaptive_out1.csv"))
        y_test = DataFrame(CSV.File("data/y_test_sin_adaptive_out1.csv"))
        y_test = y_test[!, "sin_wind_dir"]
    end

    if args["data"] == "traffic"
        X_test_adaptive = DataFrame(CSV.File("data/traffic_predictions_test_val.csv"))
        X_test_adaptive = X_test_adaptive[!,[2, 3,5,8,9,11,12,13,25]]
        y_test = DataFrame(CSV.File("data/traffic_test_val_scaled.csv"))
        y_test = y_test[!, "target"]
    end

    if args["data"][1:3] == "M3F"
        id = args["data"][4:end]
        X_test_adaptive = DataFrame(CSV.File("data/data_M3F/data/M3F_data_"*id*".csv"))
        y_test = DataFrame(CSV.File("data/data_M3F/M3F_targets/M3F_target_"*id*".csv"))
        y_test = y_test[!, "target"]
    end

    if args["data"][1:end-3] == "hurricane"
    ### Choose END-ID 12 for DL/ML only
    ### Choose END-ID 16 for ML+OP
    ### FSSE 17
    ### OFCL 18
        if args["data"][end-1:end] == "EP"
            X_test_adaptive = DataFrame(CSV.File("data/EP_ARO_Intensity_2014_clean_v2.csv"))
        else
            X_test_adaptive = DataFrame(CSV.File("data/NA_ARO_Intensity_2014_clean_v2.csv"))
        end
        X_test_adaptive, Z, y_test = prepare_data_storms(X_test_adaptive, args["past"], args["end-id"])
        args["end-id"] = size(X_test_adaptive)[2]
    end

    if args["end-id"] == -1
        args["end-id"] = size(X_test_adaptive)[2]
    end

    #TODO check assert end id < number of models
    if args["end-id"] > size(X_test_adaptive)[2]
        println("There are only ", size(X_test_adaptive)[2], " ensemble members.")
        args["end-id"] = min(args["end-id"],size(X_test_adaptive)[2])
    end

    X = Matrix(X_test_adaptive)[:,args["begin-id"]:args["end-id"]]
    println(X[1:5,:])
    n, p = size(X)

    #TODO check the n/2 to split index
    split_index = floor(Int,n*args["train_test_split"])

    #X = (X .- mean(X[1:floor(Int, split_index),:], dims =1))./ std(X[1:floor(Int, split_index),:], dims=1)#[!,1]
    y = y_test[:, 1];
    mean_y = mean(y[1:floor(Int, split_index),:])
    std_y = std(y[1:floor(Int, split_index),:])

    X = (X .- mean_y)./std_y;
    y = (y .- mean_y)./std_y;
    if args["data"][1:end-3] == "hurricane" || args["lead_time"]>1
        Z = (Z .- mean_y)./std_y
    end
    println("Mean target: ", mean_y, " Std target: ", std_y)

    #TODO code all_past -1

    val = min(args["val"], n-split_index-1)

    #TODO: Clean with only args to be passed
    try
        if args["data"][1:end-3] == "hurricane" || args["lead_time"]>1
            eval_method_hurricane(args, X, Z, y, y, args["train_test_split"], args["past"], args["num-past"], val, mean_y, std_y)
        else
            eval_method(args, X, y, y, args["train_test_split"], args["past"], args["num-past"], val, mean_y, std_y)
        end
        println("Results completed")
    catch e
        println(e)
        println("Problem with Dataset ", args["data"])
        #errors = DataFrame(Dataset = Int64[], Dataset2 = Int64[])

        errors = DataFrame(CSV.File("errors.csv"))

        push!(errors, (parse(Int,args["data"][4:end]), 0) )
        CSV.write("errors.csv", errors)
    end
end

main()
