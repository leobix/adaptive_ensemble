function prepare_data_from_y(X, y, n0, n, m, uncertainty, last_yT = false)


#     # Input:
#     X: features
#     y: targets
#     n0:  the index where the training data starts for beta 0
#     n: the number of samples for learning beta 0, there is a total of n+1
#     m: the number of samples for the adaptive part

    #TODO HAndle edge case when n0+n too big
    X0 = Matrix(X[n0:n0+n,:])
    X0[:,1] = ones(n+1)
    y0 = y[n0:n0+n,:][:]

    yt_true = y[n0+n+1:n0+n+m,:][:]
    Xt = Matrix(X[n0+n+1:n0+n+m,:])
    Xt[:,1] = ones(m)
    yt = yt_true
    if last_yT
        yt_true[m] = mean(Xt[m])
    end

    D_min = yt .- uncertainty.*abs.(yt)
    D_max = yt .+ uncertainty.*abs.(yt)
    return X0, y0, Xt, yt, yt_true, D_min, D_max
end




function prepare_data_from_y_hurricane(X, Z, y, n0, n, m, uncertainty, last_yT = false)

    """ Nuance with previous: takes Z as input as well
    #     # Input:
    #     X: features
    #     y: targets
    #     n0:  the index where the training data starts for beta 0
    #     n: the number of samples for learning beta 0, there is a total of n+1
    #     m: the number of samples for the adaptive part
    """
    #TODO HAndle edge case when n0+n too big
    X0 = Matrix(X[n0:n0+n,:])
    Z0 = Matrix(Z[n0:n0+n,:])
    y0 = y[n0:n0+n,:][:]

    yt_true = y[n0+n+1:n0+n+m,:][:]
    Xt = Matrix(X[n0+n+1:n0+n+m,:])
    Zt = Matrix(Z[n0+n+1:n0+n+m,:])
    yt = yt_true
    if last_yT
        yt_true[m] = mean(Xt[m])
    end

    D_min = yt .- uncertainty.*abs.(yt)
    D_max = yt .+ uncertainty.*abs.(yt)
    return X0, Z0, y0, Xt, Zt, yt, yt_true, D_min, D_max
end



function prepare_data_from_X(X, y, n0, n, m, std_factor)

    X0 = Matrix(X[n0:n0+n,:])
    X0[:,1] = ones(n+1)
    y0 = y[n0:n0+n,:][:]

    yt_true = y[n0+n+1:n0+n+m,:][:]
    Xt = Matrix(X[n0+n+1:n0+n+m,:])
    Xt[:,1] = ones(m)
    yt = mean(Xt, dims = 2)

    D_min = yt .- std_factor*std(Xt, dims = 2)
    D_max = yt .+ std_factor*std(Xt, dims = 2)

    return X0, y0, Xt, yt, yt_true, D_min, D_max
end



function save_array_as_csv(args, arr, folder_name::String, file_name::String)
    if !ispath(folder_name)
        mkpath(folder_name)
    end
    filepath = joinpath(folder_name, "results_"*args["data"]*"_"*string(args["rho_beta"])*"_"*string(args["rho"])*"_"*string(args["rho_V"]) *"_"* file_name * ".csv")
    try
        df = DataFrame(arr, :auto)
        CSV.write(filepath, df)
    catch e
        CSV.write(filepath, arr)
    end
    println("Array saved as $filepath")
end