function prepare_data_from_y(X, y, n0, n, m, uncertainty, last_yT = false)

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