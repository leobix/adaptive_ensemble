using CSV
using DataFrames
using Printf
using Statistics

const ROOT = abspath(joinpath(@__DIR__, ".."))
const RESULT_ROOT = get(ENV, "AE_RESULT_ROOT", joinpath(ROOT, "results_transformer_comparison"))
const FIG_ROOT = get(ENV, "AE_FIG_ROOT", joinpath(ROOT, "figures", "transformer_comparison"))
const RUN_MODE = get(ENV, "AE_RUN_MODE", "showcase")

mkpath(FIG_ROOT)

const RUNS = if RUN_MODE == "full"
    [
        (
            dataset = "energy",
            title = "Energy Forecasting",
            result_dir = joinpath(RESULT_ROOT, "energy"),
            target_file = joinpath(ROOT, "data", "energy_y_test_val.csv"),
            target_col = 1,
            x_file = joinpath(ROOT, "data", "energy_predictions_test_val2.csv"),
            val = 2000,
            split = 0.5,
        ),
        (
            dataset = "safi_speed",
            title = "Wind Speed Forecasting",
            result_dir = joinpath(RESULT_ROOT, "safi_speed"),
            target_file = joinpath(ROOT, "data", "y_test_speed_adaptive_out1.csv"),
            target_col = "speed",
            x_file = joinpath(ROOT, "data", "X_test_speed_adaptive_out1.csv"),
            val = 1699,
            split = 0.5,
        ),
    ]
elseif RUN_MODE == "wind_preprocessed"
    [
        (
            dataset = "safi_speed",
            title = "Wind Speed Forecasting (Preprocessed)",
            result_dir = joinpath(RESULT_ROOT, "safi_speed"),
            target_file = joinpath(ROOT, "data", "preprocessing", "wind_speed_y_preprocessed.csv"),
            target_col = "speed",
            x_file = joinpath(ROOT, "data", "preprocessing", "wind_speed_X_preprocessed.csv"),
            val = 1699,
            split = 0.5,
        ),
    ]
else
    [
        (
            dataset = "energy",
            title = "Energy Forecasting",
            result_dir = joinpath(RESULT_ROOT, "energy"),
            target_file = joinpath(ROOT, "data", "energy_y_test_val.csv"),
            target_col = 1,
            x_file = joinpath(ROOT, "data", "energy_predictions_test_val2.csv"),
            val = 24,
            split = 0.5,
        ),
        (
            dataset = "safi_speed",
            title = "Wind Speed Forecasting",
            result_dir = joinpath(RESULT_ROOT, "safi_speed"),
            target_file = joinpath(ROOT, "data", "y_test_speed_adaptive_out1.csv"),
            target_col = "speed",
            x_file = joinpath(ROOT, "data", "X_test_speed_adaptive_out1.csv"),
            val = 24,
            split = 0.5,
        ),
    ]
end

const METHOD_ORDER = [
    "best_model",
    "mean",
    "ridge",
    "adaptive_ridge_exact",
    "adaptive_ridge_standard",
    "adaptive_ridge_exact_err_rule",
    "adaptive_ridge_standard_err_rule",
    "bandits_full",
    "bandits_recent",
    "PA",
    "fedformer",
    "informer",
]

const METHOD_LABELS = Dict(
    "best_model" => "Best model",
    "mean" => "Mean",
    "ridge" => "Ridge",
    "adaptive_ridge_exact" => "Adaptive exact",
    "adaptive_ridge_standard" => "Adaptive std.",
    "adaptive_ridge_exact_err_rule" => "Adaptive exact err",
    "adaptive_ridge_standard_err_rule" => "Adaptive std. err",
    "bandits_full" => "Bandits full",
    "bandits_recent" => "Bandits recent",
    "PA" => "Passive Agg.",
    "fedformer" => "FEDFormer",
    "informer" => "Informer",
)

const COLORS = Dict(
    "best_model" => "#1b9e77",
    "mean" => "#999999",
    "ridge" => "#7570b3",
    "adaptive_ridge_exact" => "#2f6f9f",
    "adaptive_ridge_standard" => "#75aadb",
    "adaptive_ridge_exact_err_rule" => "#4c9f70",
    "adaptive_ridge_standard_err_rule" => "#78c6a3",
    "bandits_full" => "#a6761d",
    "bandits_recent" => "#e7298a",
    "PA" => "#d95f02",
    "fedformer" => "#5b3f8c",
    "informer" => "#c43b4d",
)

const METRICS = [
    ("MAE", "MAE", true),
    ("RMSE", "RMSE", true),
    ("MAPE", "MAPE (%)", true),
    ("R2", "R2", false),
    ("Time", "Time (ms)", true),
]

escape_xml(x) = replace(string(x), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

function fmt(x)
    if !isfinite(x)
        return string(x)
    elseif abs(x) >= 1000
        return @sprintf("%.2e", x)
    elseif abs(x) >= 100
        return @sprintf("%.0f", x)
    elseif abs(x) >= 10
        return @sprintf("%.1f", x)
    elseif abs(x) >= 1
        return @sprintf("%.2f", x)
    else
        return @sprintf("%.3f", x)
    end
end

function method_sort_key(method)
    idx = findfirst(==(method), METHOD_ORDER)
    return isnothing(idx) ? length(METHOD_ORDER) + 1 : idx
end

function ordered_methods(df)
    methods = unique(String.(df.Method))
    return sort(methods, by=method_sort_key)
end

function metric_values(df, metric, methods)
    values = Float64[]
    for method in methods
        rows = df[df.Method .== method, :]
        if nrow(rows) == 0 || !(metric in names(rows))
            push!(values, NaN)
        else
            push!(values, Float64(rows[end, metric]))
        end
    end
    return values
end

function positive_log_scale(values)
    finite_vals = filter(x -> isfinite(x) && x > 0, values)
    if length(finite_vals) < 2
        return false
    end
    return maximum(finite_vals) / max(minimum(finite_vals), eps()) > 100
end

function scale_values(values; log_scale=false, pad_fraction=0.08)
    finite_vals = filter(isfinite, values)
    if isempty(finite_vals)
        return (0.0, 1.0)
    end
    if log_scale
        finite_vals = filter(>(0), finite_vals)
        ymin = log10(minimum(finite_vals))
        ymax = log10(maximum(finite_vals))
    else
        ymin = min(0.0, minimum(finite_vals))
        ymax = maximum(finite_vals)
    end
    if ymax == ymin
        ymax += 1.0
    end
    pad = (ymax - ymin) * pad_fraction
    return (ymin - pad, ymax + pad)
end

function draw_metric_panel(df, dataset_title, metric, title, allow_log, x0, y0, width, height)
    methods = ordered_methods(df)
    values = metric_values(df, metric, methods)
    log_scale = allow_log && positive_log_scale(values)
    ymin, ymax = scale_values(values; log_scale=log_scale)
    left = x0 + 58
    right = x0 + width - 12
    top = y0 + 38
    bottom = y0 + height - 76
    plot_w = right - left
    plot_h = bottom - top
    bar_gap = 6
    bar_w = max(8.0, (plot_w - bar_gap * (length(methods) + 1)) / length(methods))

    out = IOBuffer()
    println(out, """<rect x="$x0" y="$y0" width="$width" height="$height" fill="#ffffff" stroke="#d6d6d6"/>""")
    suffix = log_scale ? " (log)" : ""
    println(out, """<text x="$(x0 + 14)" y="$(y0 + 23)" font-size="15" font-weight="600" fill="#222222">$(escape_xml(title * suffix))</text>""")

    for t in 0:4
        scaled = ymin + (ymax - ymin) * t / 4
        label_val = log_scale ? 10.0^scaled : scaled
        y = bottom - (scaled - ymin) / (ymax - ymin) * plot_h
        println(out, """<line x1="$left" x2="$right" y1="$y" y2="$y" stroke="#e8e8e8"/>""")
        println(out, """<text x="$(left - 8)" y="$(y + 4)" text-anchor="end" font-size="10" fill="#666666">$(fmt(label_val))</text>""")
    end

    if !log_scale
        zero_y = bottom - (0 - ymin) / (ymax - ymin) * plot_h
        if top <= zero_y <= bottom
            println(out, """<line x1="$left" x2="$right" y1="$zero_y" y2="$zero_y" stroke="#9a9a9a"/>""")
        end
    end

    for (i, method) in enumerate(methods)
        val = values[i]
        (!isfinite(val) || (log_scale && val <= 0)) && continue
        scaled = log_scale ? log10(val) : val
        base = log_scale ? ymin : 0.0
        x = left + bar_gap + (i - 1) * (bar_w + bar_gap)
        y = bottom - (scaled - ymin) / (ymax - ymin) * plot_h
        ybase = bottom - (base - ymin) / (ymax - ymin) * plot_h
        h = max(1.0, abs(ybase - y))
        rect_y = min(y, ybase)
        color = get(COLORS, method, "#555555")
        println(out, """<rect x="$x" y="$rect_y" width="$bar_w" height="$h" rx="2" fill="$color"/>""")
        println(out, """<text x="$(x + bar_w / 2)" y="$(rect_y - 5)" text-anchor="middle" font-size="9" fill="#333333">$(fmt(val))</text>""")
        label = get(METHOD_LABELS, method, method)
        println(out, """<text x="$(x + bar_w / 2)" y="$(bottom + 18)" text-anchor="middle" font-size="8" fill="#333333" transform="rotate(45 $(x + bar_w / 2) $(bottom + 18))">$(escape_xml(label))</text>""")
    end

    println(out, """<text x="$(x0 + width - 14)" y="$(y0 + height - 10)" text-anchor="end" font-size="10" fill="#777777">$(escape_xml(dataset_title))</text>""")
    return String(take!(out))
end

function write_metric_plot(path, df, dataset_title)
    panel_w = 520
    panel_h = 280
    gap = 18
    width = panel_w * 2 + gap + 40
    height = panel_h * 3 + gap * 2 + 78
    out = IOBuffer()
    println(out, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(out, """<rect width="100%" height="100%" fill="#f7f7f7"/>""")
    println(out, """<text x="20" y="32" font-size="22" font-weight="700" fill="#222222">Transformer comparison: $(escape_xml(dataset_title))</text>""")
    println(out, """<text x="20" y="54" font-size="12" fill="#666666">FEDFormer and Informer use the same train/test split and matched transformer dimensions.</text>""")
    for (k, (metric, title, allow_log)) in enumerate(METRICS)
        row = div(k - 1, 2)
        col = (k - 1) % 2
        x = 20 + col * (panel_w + gap)
        y = 76 + row * (panel_h + gap)
        println(out, draw_metric_panel(df, dataset_title, metric, title, allow_log, x, y, panel_w, panel_h))
    end
    println(out, "</svg>")
    write(path, String(take!(out)))
end

function read_target_vector(path, target_col)
    df = CSV.read(path, DataFrame)
    return Vector{Float64}(df[:, target_col])
end

function read_prediction_vector(path)
    df = CSV.read(path, DataFrame)
    return Vector{Float64}(df[:, 1])
end

function draw_prediction_plot(path, run)
    y = read_target_vector(run.target_file, run.target_col)
    x_rows = nrow(CSV.read(run.x_file, DataFrame))
    split_index = floor(Int, x_rows * run.split)
    mean_y = mean(y[1:split_index])
    std_y = std(y[1:split_index])
    idx = (split_index + 2):(split_index + run.val + 1)
    truth = y[idx]

    fed_path = joinpath(run.result_dir, "fedformer_preds.csv")
    inf_path = joinpath(run.result_dir, "informer_preds.csv")
    fed = read_prediction_vector(fed_path) .* std_y .+ mean_y
    inf = read_prediction_vector(inf_path) .* std_y .+ mean_y
    n = minimum([length(truth), length(fed), length(inf), run.val])
    truth = truth[1:n]
    fed = fed[1:n]
    inf = inf[1:n]

    width = 980
    height = 420
    left = 72
    right = width - 30
    top = 76
    bottom = height - 60
    plot_w = right - left
    plot_h = bottom - top
    vals = vcat(truth, fed, inf)
    ymin, ymax = scale_values(vals; pad_fraction=0.12)

    function points(series)
        pts = String[]
        for i in 1:n
            x = n == 1 ? left + plot_w / 2 : left + (i - 1) / (n - 1) * plot_w
            yv = bottom - (series[i] - ymin) / (ymax - ymin) * plot_h
            push!(pts, @sprintf("%.2f,%.2f", x, yv))
        end
        return join(pts, " ")
    end

    out = IOBuffer()
    println(out, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(out, """<rect width="100%" height="100%" fill="#f7f7f7"/>""")
    println(out, """<rect x="20" y="20" width="$(width - 40)" height="$(height - 40)" fill="#ffffff" stroke="#d6d6d6"/>""")
    println(out, """<text x="40" y="48" font-size="22" font-weight="700" fill="#222222">Transformer predictions: $(escape_xml(run.title))</text>""")
    println(out, """<text x="40" y="68" font-size="12" fill="#666666">Held-out steps $(first(idx))-$(first(idx)+n-1), unstandardized to the target scale.</text>""")
    for t in 0:4
        val = ymin + (ymax - ymin) * t / 4
        ytick = bottom - (val - ymin) / (ymax - ymin) * plot_h
        println(out, """<line x1="$left" x2="$right" y1="$ytick" y2="$ytick" stroke="#e8e8e8"/>""")
        println(out, """<text x="$(left - 10)" y="$(ytick + 4)" text-anchor="end" font-size="11" fill="#666666">$(fmt(val))</text>""")
    end
    println(out, """<polyline points="$(points(truth))" fill="none" stroke="#222222" stroke-width="2.4"/>""")
    println(out, """<polyline points="$(points(fed))" fill="none" stroke="$(COLORS["fedformer"])" stroke-width="2.0"/>""")
    println(out, """<polyline points="$(points(inf))" fill="none" stroke="$(COLORS["informer"])" stroke-width="2.0"/>""")
    legend_y = 92
    legend = [("Actual", "#222222"), ("FEDFormer", COLORS["fedformer"]), ("Informer", COLORS["informer"])]
    for (i, (label, color)) in enumerate(legend)
        x = right - 260 + (i - 1) * 92
        println(out, """<line x1="$x" x2="$(x + 22)" y1="$legend_y" y2="$legend_y" stroke="$color" stroke-width="3"/>""")
        println(out, """<text x="$(x + 28)" y="$(legend_y + 4)" font-size="12" fill="#333333">$(escape_xml(label))</text>""")
    end
    println(out, """<text x="$right" y="$(bottom + 34)" text-anchor="end" font-size="11" fill="#777777">held-out step</text>""")
    println(out, "</svg>")
    write(path, String(take!(out)))
end

function main()
    outputs = String[]
    for run in RUNS
        metrics_path = joinpath(run.result_dir, "metrics.csv")
        if !isfile(metrics_path)
            @warn "Skipping $(run.dataset); missing curated metrics CSV" metrics_path
            continue
        end
        df = CSV.read(metrics_path, DataFrame)
        metric_plot = joinpath(FIG_ROOT, "metrics_$(run.dataset).svg")
        pred_plot = joinpath(FIG_ROOT, "predictions_$(run.dataset).svg")
        write_metric_plot(metric_plot, df, run.title)
        draw_prediction_plot(pred_plot, run)
        push!(outputs, metric_plot)
        push!(outputs, pred_plot)
    end

    println("Generated plots:")
    foreach(println, outputs)
end

main()
