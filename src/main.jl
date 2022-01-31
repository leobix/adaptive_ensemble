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

include("eval.jl")
include("utils.jl")

include("algos/benders.jl")
include("algos/master_primal.jl")
include("algos/OLS.jl")
include("algos/adaptive_linear_decision_rule.jl")

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
            default = "data/energy_predictions_test_val.csv"
            arg_type = String

        "--filename-y"
            help = "filename of targets y"
            default = "data/energy_y_test_val.csv"
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
            help = "number of max cuts in Benders problem"
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

        "--max-cuts"
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
            default = 25

        "--last_yT"
            help = "last_yT"
            action = :store_true

        "--benders"
            help = "ridge"
            action = :store_true

        "--more_data_for_beta0"
            help = "more_data_for_beta0"
            action = :store_true

        "--fix-beta0"
            help = "fix beta0"
            action = :store_true

        "--verbose"
            help = "verbose"
            action = :store_true

        "--ridge"
            help = "verbose"
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

        "--rho"
            help = "rho"
            arg_type = Float64
            default = 0.1

        "--reg"
            help = "rho"
            arg_type = Float64
            default = 0.1


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
        X_test_adaptive = CSV.read(args["filename-X"], DataFrame)
        y_test = CSV.read(args["filename-y"], DataFrame, header = 0)
    end

    if args["data"] == "safi"
        X_test_adaptive = CSV.read("data/X_test_adaptive.csv", DataFrame)
        y_test = CSV.read("data/y_test_speed.csv", DataFrame)
        y_test = y_test[!, "speed"]
    end

    if args["data"] == "traffic"
        X_test_adaptive = CSV.read("data/traffic_predictions_test_val.csv", DataFrame)
        X_test_adaptive = X_test_adaptive[!,[2, 3,5,8,9,11,12,13,25]]
        y_test = CSV.read("data/traffic_test_val_scaled.csv", DataFrame)
        y_test = y_test[!, "target"]
    end

    if args["data"][1:3] == "M3F"
        id = args["data"][9:end]
        X_test_adaptive = CSV.read("data/M3F_data_"+id+".csv", DataFrame)
        y_test = CSV.read("data/M3F_target_"+id+".csv", DataFrame)
        y_test = y_test[!, "target"]
    end

    #TODO check end-id -1
    if args["end-id"] == -1
        args["end-id"] = size(X_test_adaptive)[2]
    end

    #TODO assert end id < number of models
    X = Matrix(X_test_adaptive)[:,args["begin-id"]:args["end-id"]]
    n, p = size(X)

    #TODO check the n/2 to split index
    split_index = floor(Int,n*args["train_test_split"])
    X = (X .- mean(X[1:floor(Int, split_index),:], dims =1))./ std(X[1:floor(Int, split_index),:], dims=1)#[!,1]
    #X[:,1] = ones(n)

    y = y_test[:, 1];
    y = (y .- mean(y[1:floor(Int, split_index),:], dims =1))./ std(y[1:floor(Int, split_index),:], dims=1);

    if args["reg"] == -1
        reg = 1/(args["past"]*args["num-past"])
    else
        reg = args["reg"]
    end

    #TODO code all_past -1

    eval_method(X, y, y, args["train_test_split"], args["past"], args["num-past"], args["val"], args["uncertainty"], args["epsilon-inf"], args["delta-inf"], args["last_yT"],
        args["epsilon-l2"], args["delta-l2"], args["rho"], reg, args["max-cuts"], args["verbose"],
        args["fix-beta0"], args["more_data_for_beta0"], args["benders"], args["ridge"])

    println("Results completed")
end

main()
