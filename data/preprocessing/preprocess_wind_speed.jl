using CSV
using DataFrames
using Printf
using Statistics

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(ROOT, "data")
const OUTPUT_DIR = @__DIR__

const INPUT_X = joinpath(DATA_DIR, "X_test_speed_adaptive_out1.csv")
const INPUT_Y = joinpath(DATA_DIR, "y_test_speed_adaptive_out1.csv")
const OUTPUT_X = joinpath(OUTPUT_DIR, "wind_speed_X_preprocessed.csv")
const OUTPUT_Y = joinpath(OUTPUT_DIR, "wind_speed_y_preprocessed.csv")
const REPORT_PATH = joinpath(OUTPUT_DIR, "wind_speed_preprocessing_report.md")

const TARGET_COL = "speed"
const TRAIN_TEST_SPLIT = 0.5

function numeric_vector(col)
    return Float64[Float64(x) for x in collect(skipmissing(col))]
end

function is_index_like(col)
    vals = numeric_vector(col)
    length(vals) < 3 && return false
    diffs = diff(vals)
    return all(isfinite, vals) && all(abs.(diffs .- 1.0) .<= 1e-8)
end

function remove_leading_index_columns!(df::DataFrame)
    removed = String[]
    while ncol(df) > 0 && is_index_like(df[!, 1])
        push!(removed, names(df)[1])
        select!(df, Not(names(df)[1]))
    end
    return removed
end

function assert_no_missing_or_nonfinite(df::DataFrame, label::String)
    for name in names(df)
        vals = numeric_vector(df[!, name])
        length(vals) == nrow(df) || error("$label column $name contains missing/non-numeric values")
        all(isfinite, vals) || error("$label column $name contains non-finite values")
    end
end

function target_scaled_feature_ranges(x_clean::DataFrame, y_clean::DataFrame)
    split_index = floor(Int, nrow(x_clean) * TRAIN_TEST_SPLIT)
    y = Vector{Float64}(y_clean[!, TARGET_COL])
    mean_y = mean(y[1:split_index])
    std_y = std(y[1:split_index])
    rows = DataFrame(
        feature = String[],
        target_scaled_min = Float64[],
        target_scaled_max = Float64[],
        target_scaled_mean = Float64[],
        target_scaled_std = Float64[],
    )
    for name in names(x_clean)
        vals = (Vector{Float64}(x_clean[!, name]) .- mean_y) ./ std_y
        push!(rows, (
            name,
            minimum(vals),
            maximum(vals),
            mean(vals),
            std(vals),
        ))
    end
    return rows, mean_y, std_y, split_index
end

function write_report(path, x_input, y_input, x_clean, y_clean, removed_x, removed_y, ranges, mean_y, std_y, split_index)
    open(path, "w") do io
        println(io, "# Wind-Speed Preprocessing Report")
        println(io)
        println(io, "Input X: `$(relpath(INPUT_X, ROOT))`")
        println(io, "Input y: `$(relpath(INPUT_Y, ROOT))`")
        println(io, "Output X: `$(relpath(OUTPUT_X, ROOT))`")
        println(io, "Output y: `$(relpath(OUTPUT_Y, ROOT))`")
        println(io)
        println(io, "## Actions")
        println(io)
        println(io, "- Removed leading index-like X columns: `$(join(removed_x, "`, `"))`")
        println(io, "- Removed leading index-like y columns: `$(join(removed_y, "`, `"))`")
        println(io, "- Added first X column `intercept` with value `1.0`.")
        println(io, "- Preserved all real wind forecast columns in their original target units.")
        println(io, "- Preserved target column `$TARGET_COL` in its original target units.")
        println(io)
        println(io, "## Shape Check")
        println(io)
        println(io, "- Raw X shape: `$(nrow(x_input)) x $(ncol(x_input))`")
        println(io, "- Clean X shape: `$(nrow(x_clean)) x $(ncol(x_clean))`")
        println(io, "- Raw y shape: `$(nrow(y_input)) x $(ncol(y_input))`")
        println(io, "- Clean y shape: `$(nrow(y_clean)) x $(ncol(y_clean))`")
        println(io)
        println(io, "## Target-Scaled Feature Ranges")
        println(io)
        println(io, "The main pipeline target-standardizes `X` using the training target mean/std. After this cleanup, the former row index no longer appears as a huge feature.")
        println(io)
        println(io, "- Training split index: `$split_index`")
        println(io, "- Training target mean: `$(@sprintf("%.8f", mean_y))`")
        println(io, "- Training target std: `$(@sprintf("%.8f", std_y))`")
        println(io)
        println(io, "| feature | min | max | mean | std |")
        println(io, "|---|---:|---:|---:|---:|")
        for row in eachrow(ranges)
            println(io, "| $(row.feature) | $(@sprintf("%.4f", row.target_scaled_min)) | $(@sprintf("%.4f", row.target_scaled_max)) | $(@sprintf("%.4f", row.target_scaled_mean)) | $(@sprintf("%.4f", row.target_scaled_std)) |")
        end
    end
end

function preprocess_wind_speed()
    mkpath(OUTPUT_DIR)

    x_input = CSV.read(INPUT_X, DataFrame)
    y_input = CSV.read(INPUT_Y, DataFrame)
    x_clean = copy(x_input)
    y_clean = copy(y_input)

    removed_x = remove_leading_index_columns!(x_clean)
    removed_y = remove_leading_index_columns!(y_clean)

    TARGET_COL in names(y_clean) || error("Target column `$TARGET_COL` was not found after removing index columns")
    y_clean = DataFrame(TARGET_COL => Vector{Float64}(y_clean[!, TARGET_COL]))

    forecast_cols = names(x_clean)
    nrow(x_clean) == nrow(y_clean) || error("X and y row counts differ after preprocessing")
    assert_no_missing_or_nonfinite(x_clean, "X")
    assert_no_missing_or_nonfinite(y_clean, "y")

    x_out = DataFrame(intercept = ones(Float64, nrow(x_clean)))
    for name in forecast_cols
        x_out[!, name] = Vector{Float64}(x_clean[!, name])
    end

    ncol(x_out) == ncol(x_clean) + 1 || error("Expected exactly one added intercept column")
    all(x_out[!, :intercept] .== 1.0) || error("Intercept column is not constant 1.0")

    ranges, mean_y, std_y, split_index = target_scaled_feature_ranges(x_out, y_clean)
    non_intercept = ranges[ranges.feature .!= "intercept", :]
    maximum(abs.(non_intercept.target_scaled_max)) < 20 || error("A cleaned forecast column is unexpectedly large after target scaling")
    maximum(abs.(non_intercept.target_scaled_min)) < 20 || error("A cleaned forecast column is unexpectedly large after target scaling")

    CSV.write(OUTPUT_X, x_out)
    CSV.write(OUTPUT_Y, y_clean)
    write_report(REPORT_PATH, x_input, y_input, x_out, y_clean, removed_x, removed_y, ranges, mean_y, std_y, split_index)

    println("Wrote ", relpath(OUTPUT_X, ROOT))
    println("Wrote ", relpath(OUTPUT_Y, ROOT))
    println("Wrote ", relpath(REPORT_PATH, ROOT))
    println("Rows: ", nrow(x_out), ", features including intercept: ", ncol(x_out))
    println("Removed X index columns: ", isempty(removed_x) ? "(none)" : join(removed_x, ", "))
    println("Removed y index columns: ", isempty(removed_y) ? "(none)" : join(removed_y, ", "))
end

preprocess_wind_speed()
