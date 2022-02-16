function get_storms(df, storm_len)
    long_storms = []
    short_storms = []
    acc = 0
    for i=2:size(df,1)
        if df['hours'][i] != df['hours'][i-1]+6:
            if df['hours'][i] == 0 && df['hours'][i-1]==18
                pass
            else:
                if i-acc>storm_len:
                    long_storms.append(df[acc:i])
                else:
                    short_storms.append(df[acc:i])
                end
                acc = i
            end
        end
    end
    return long_storms, short_storms
end
