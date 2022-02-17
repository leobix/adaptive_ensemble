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
    for i = 1:n
        #forecasts start at index 7
        data_storms[i] = storms[i][!, 7:end]
        y_storms[i] = storms[i][!, "TRUTH"]
    end
    return data_storms, y_storms
end



function get_full_X_Z_y(data, y, past)
    """
    Concatenate all X, Z, y into a time series
    """
    #number of storms
    n = size(data,1)
    X_tot, Z_tot, y_tot = get_X_Z_y(Matrix(data[1]), y[1], past)
    for i=2:n
        X_, Z_, y_ = get_X_Z_y(Matrix(data[i]), y[i], past)
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
    forecasts, truths = separate_storms(long_storms)

    #reconcatenate everything into a single time series
    X_tot, Z_tot, y_tot = get_full_X_Z_y(forecasts, truths, past)
    println("X_tot", X_tot[4,:])
    println("Z_tot", Z_tot[4,:])
    println("y_tot", y_tot[1:4])
    return X_tot, Z_tot, y_tot
end