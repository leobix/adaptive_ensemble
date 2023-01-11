function get_storms(df, storm_len)
    long_storms = []
    short_storms = []
    acc = 1
    for i=2:size(df,1)-1
        if abs(df[!, "days"][i] - df[!, "days"][i-1]) > 1
            if df[!, "days"][i] != 1
                if i-acc>storm_len
                    push!(long_storms,df[acc:i-1, :])
                else
                    push!(short_storms,df[acc:i-1, :])
                end
                acc = i
            else
                if df[!, "hours"][i] != df[!, "hours"][i-1]+6
                    if df[!, "hours"][i] == 0 && df[!, "hours"][i-1]==18
                    else
                        if i-acc>storm_len
                            push!(long_storms,df[acc:i-1, :])
                        else
                            push!(short_storms,df[acc:i-1, :])
                        end
                        acc = i
                    end
                end
            end
        else
            if df[!, "hours"][i] != df[!, "hours"][i-1]+6
                if df[!, "hours"][i] == 0 && df[!, "hours"][i-1]==18
                else
                    if i-acc>storm_len
                        push!(long_storms,df[acc:i-1, :])
                    else
                        push!(short_storms,df[acc:i-1, :])
                    end
                    acc = i
                end
            elseif df[!, "days"][i] - df[!, "days"][i-1] > 0
                if i-acc>storm_len
                        push!(long_storms,df[acc:i-1, :])
                    else
                        push!(short_storms,df[acc:i-1, :])
                    end
                    acc = i
            end
        end
    end
    return long_storms, short_storms
end


function separate_storms(storms)
    """
    Separate storms between forecasts and ground truth
    """
    n = size(storms,1)
    data_storms = copy(storms)
    y_storms = copy(storms)
    current_speed_storms = copy(storms)
    for i = 1:n
        #forecasts start at index 8, the rest is truth, current speed and some ids
        #to test only using operational models, choose 13 instead of 8
        data_storms[i] = storms[i][!, 14:end]
        y_storms[i] = storms[i][!, "TRUTH"]
        current_speed_storms[i] = storms[i][!, "CURRENT_SPEED"]
    end
    return data_storms, y_storms, current_speed_storms
end

function get_X_Z_y_hurricane(X, y, current_speed, T)
    """
    Input: training data X and corresponding labels y ; how many time-steps from the past to be used
    Output: the past features X with past targets y as a Z training data (no present features)
    """

    n, p = size(X)
    #T past time steps * p features + T targets
    Z = ones(n-T, T*p+T)
    for i=T+1:n
        for t=1:T
            #we can't use err_rule because we don't have access to the ground truth values yet.
            Z[i-T,1+p*(t-1):p*t] = X[i-t,:]
        end
        # here we put the current speed and past current speed instead of the targets since it would be cheating
        Z[i-T, (p*T+1):end] = current_speed[i-T+1:i]
    end
    return X[T+1:end,:], Z, y[T+1:end]
end



function get_full_X_Z_y(data, y, current_speeds, past)
    """
    Concatenate all X, Z, y into a time series
    """
    #number of storms
    n = size(data,1)
    X_tot, Z_tot, y_tot = get_X_Z_y_hurricane(Matrix(data[1]), y[1], current_speeds[1], past)
    for i=2:n
        X_, Z_, y_ = get_X_Z_y_hurricane(Matrix(data[i]), y[i], current_speeds[i], past)
        X_tot, Z_tot, y_tot = vcat(X_tot, X_), vcat(Z_tot, Z_), vcat(y_tot, y_)
    end
    return X_tot, Z_tot, y_tot
end



function prepare_data_storms(data_or, past, e_id)
    data = data_or[!,1:e_id]
    data[!,"intercept"] .= 1
    #get storms with more than past+1 timesteps
    long_storms, short_storms = get_storms(data, past+1)
    #get the forecasts corresponding and the truths, while keeping data until the final id
    forecasts, truths, current_speeds = separate_storms(long_storms)

    #reconcatenate everything into a single time series
    X_tot, Z_tot, y_tot = get_full_X_Z_y(forecasts, truths, current_speeds, past)
    println("X_tot", X_tot[4,:])
    println("Z_tot", Z_tot[4,:])
    println("y_tot", y_tot[1:4])
    return X_tot, Z_tot, y_tot
end