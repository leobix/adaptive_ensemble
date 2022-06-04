import Pkg
#be careful: implies you created a robust virtual environment
#Pkg.activate("robust")

using JuMP

using LinearAlgebra
using Statistics
using StatsBase
using DataFrames
using Gurobi
using ArgParse
using CSV
using Random
#using Mosek
#using MosekTools
using Convex

include("eval2.jl")
include("utils.jl")
include("utils_hurricane.jl")


const GRB_ENV = Gurobi.Env()

#parser
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--filename-results"
            help = "filename path to save results file"
            default = "results.csv"
            arg_type = String

        "--filename-X"
            help = "filename path of the data X"
            default = "data/energy_predictions_test_val2.csv"
            arg_type = String

        "--filename-y"
            help = "filename path of targets y"
            default = "data/energy_y_test_val.csv"
            arg_type = String

        "--data"
            help = "for specific use case where we precoded the data paths, supports energy, hurricane_NA, hurricane_EP, safi"
            default = "energy"
            arg_type = String

        "--train_test_split"
            help = "train test split value, between 0 and 1"
            arg_type = Float64
            default = 0.5

        "--past"
            help = "window size of past data used in the decision rule of the adaptive ridge model"
            arg_type = Int
            default = 3

        "--num-past"
            help = "training size will be past*num_past, so both parameters are connected"
            arg_type = Int
            default = 100

        "--val"
            help = "size of the test set, -1 is all data after training set"
            arg_type = Int
            default = -1

        "--max-cuts"
            help = "deprecated (useless) number of max cuts in Benders problem"
            arg_type = Int
            default = 10

        "--begin-id"
            help = "when several ensemble members are available allows to only use models with index greater or equal to this id"
            arg_type = Int
            default = 2

        "--end-id"
            help = "when several ensemble members are available allows to only use models with index smaller or equal to this id"
            arg_type = Int
            default = 25

        "--lead_time"
            help = "Lead time in timesteps for predictions, in general 1, otherwise may need coding adjustments"
            arg_type = Int
            default = 1

        "--last_yT"
            help = "deprecated (useless)"
            action = :store_true

        "--benders"
            help = "deprecated (useless)"
            action = :store_true

        "--more_data_for_beta0"
            help = "deprecated (useless)"
            action = :store_true

        "--CVAR"
            help = "will compute the CVaR value (takes more computational time)"
            action = :store_true

        "--verbose"
            help = "useless"
            action = :store_true

        "--ridge"
            help = "useless"
            action = :store_true

        "--err_rule"
            help = "Build Z with the regrets instead of the raw history, no need to touch this parameter"
            action = :store_true

        "--err_rule_norm"
            help = "Build Z with the regrets instead of the raw history, and normalized regrets step-wise, no need to touch this parameter"
            action = :store_true

        "--uncertainty"
            help = "useless"
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
            help = "rho for Beta_t in Adaptive Ridge Linear Decision Rule"
            arg_type = Float64
            default = 0.1

        "--rho_stat"
            help = "(useless) rho for statistical error in standard ridge"
            arg_type = Float64
            default = 0.1

        "--rho"
            help = "rho for Beta 0 and standard ridge and parameter for PA and Exp3"
            arg_type = Float64
            default = 0.1

        "--rho_V"
            help = "rho for V0 in Adaptive Ridge"
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


    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    println("Arguments:")
    for (arg,val) in args
        println("  $arg  =>  $val")
    end

    if args["data"] == "energy"
        #predecided path
        X_test_adaptive = DataFrame(CSV.File(args["filename-X"]))
        y_test = DataFrame(CSV.File(args["filename-y"]))
    end

    if args["data"] == "imports"
        X_test_adaptive = DataFrame(CSV.File("MYPATH"))
        y_test = DataFrame(CSV.File("MYPATH"))
    end

    if args["data"] == "safi_speed"
    #predecided path
        X_test_adaptive = DataFrame(CSV.File("data/X_test_speed_adaptive_out1.csv"))
        y_test = DataFrame(CSV.File("data/y_test_speed_adaptive_out1.csv"))
        y_test = y_test[!, "speed"]
    end

    if args["data"] == "safi_speed_3"
    #predecided path
    #because we predict 3 steps in advance, the data needs small adjustments
        X_test_adaptive = DataFrame(CSV.File("data/X_test_adaptive.csv"))
        y_test = DataFrame(CSV.File("data/y_test_speed.csv"))
        y_test = y_test[!, "speed"]
        X_test_adaptive = Matrix(X_test_adaptive)[:,1:args["end-id"]]
        X_test_adaptive[:,1] .= 1
        X_test_adaptive, Z, y_test = get_X_Z_y(args, X_test_adaptive, y_test[:, 1], args["past"])
    end

    if args["data"] == "safi_cos"
    #predecided path
        X_test_adaptive = DataFrame(CSV.File("data/X_test_cos_adaptive_out1.csv"))
        y_test = DataFrame(CSV.File("data/y_test_cos_adaptive_out1.csv"))
        y_test = y_test[!, "cos_wind_dir"]
    end

    if args["data"] == "traffic"
    #predecided path
        X_test_adaptive = DataFrame(CSV.File("data/traffic_predictions_test_val.csv"))
        X_test_adaptive = X_test_adaptive[!,[2, 3,5,8,9,11,12,13,25]]
        y_test = DataFrame(CSV.File("data/traffic_test_val_scaled.csv"))
        y_test = y_test[!, "target"]
    end

    if args["data"][1:3] == "M3F"
    #predecided path
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
    # We catch up a potential error that the last model id is greater than the actual number of models available
    if args["end-id"] > size(X_test_adaptive)[2]
        println("There are only ", size(X_test_adaptive)[2], " ensemble members.")
        args["end-id"] = min(args["end-id"],size(X_test_adaptive)[2])
    end

    #Loads the desired forecasts
    X = Matrix(X_test_adaptive)[:,args["begin-id"]:args["end-id"]]

    n, p = size(X)

    # We determine the data sample index on which we switch from training to testing
    split_index = floor(Int,n*args["train_test_split"])

    #X = (X .- mean(X[1:floor(Int, split_index),:], dims =1))./ std(X[1:floor(Int, split_index),:], dims=1)#[!,1]

    #Depending on how the test data is organized, this may need adjustments
    y = y_test[:, 1];

    #To standardize we compute the mean and std of the target
    mean_y = mean(y[1:floor(Int, split_index),:])
    std_y = std(y[1:floor(Int, split_index),:])

    #We substract the mean and std of the targets to the training data
    X = (X .- mean_y)./std_y;
    y = (y .- mean_y)./std_y;

    #TODO Check why I put this here and not 2 lines higher
    if args["data"][1:end-3] == "hurricane" || args["lead_time"]>1
        Z = (Z .- mean_y)./std_y
    end

    println("Mean target: ", mean_y, " Std target: ", std_y)

    #TODO code all_past -1

    #determine the last sample index of the validation/test set
    val = min(args["val"], n-split_index-1)

    try
        #Hurricane forecasting use case requires some adjustments
        if args["data"][1:end-3] == "hurricane" || args["lead_time"]>1
            eval_method_hurricane(args, X, Z, y, y, args["train_test_split"], args["past"], args["num-past"], val, mean_y, std_y)

        #Launch the evaluation
        else
            eval_method(args, X, y, y, args["train_test_split"], args["past"], args["num-past"], val, mean_y, std_y)
        end
        println("Results completed")
    catch e
        println(e)
        println("Problem with Dataset ", args["data"])

        errors = DataFrame(CSV.File("errors.csv"))

        push!(errors, (parse(Int,args["data"][4:end]), 0) )
        CSV.write("errors.csv", errors)
    end
end

main()
