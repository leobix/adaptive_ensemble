using CSV
using DataFrames
using Printf

const ROOT = abspath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results_11_12_id14_16")
const BETA_DIR = joinpath(ROOT, "results_beta")
const OUT_DIR = joinpath(ROOT, "figures", "smoke_results")

mkpath(OUT_DIR)

const METHOD_ORDER = [
    "adaptive_ridge_exact_err_rule",
    "adaptive_ridge_standard_err_rule",
    "adaptive_ridge_exact",
    "adaptive_ridge_standard",
    "PA",
    "ridge",
    "bandits_recent",
    "bandits_full",
    "best_model",
    "last_timestep",
    "mean",
]

const METHOD_LABELS = Dict(
    "adaptive_ridge_exact_err_rule" => "Adaptive Ridge\nerror rule",
    "adaptive_ridge_standard_err_rule" => "Adaptive Ridge std.\nerror rule",
    "adaptive_ridge_exact" => "Adaptive Ridge\nexact",
    "adaptive_ridge_standard" => "Adaptive Ridge\nstandard",
    "PA" => "Passive-\nAggressive",
    "ridge" => "Ridge",
    "bandits_recent" => "Exp3\nrecent",
    "bandits_full" => "Exp3\nfull",
    "best_model" => "Best model\nin hindsight",
    "last_timestep" => "Last\ntimestep",
    "mean" => "Ensemble\nmean",
)

const COLORS = Dict(
    "adaptive_ridge_exact_err_rule" => "#2f6f9f",
    "adaptive_ridge_standard_err_rule" => "#4c9f70",
    "adaptive_ridge_exact" => "#75aadb",
    "adaptive_ridge_standard" => "#78c6a3",
    "PA" => "#d95f02",
    "ridge" => "#7570b3",
    "bandits_recent" => "#e7298a",
    "bandits_full" => "#a6761d",
    "best_model" => "#1b9e77",
    "last_timestep" => "#666666",
    "mean" => "#999999",
)

const METRICS = [
    ("MAE", "MAE"),
    ("RMSE", "RMSE"),
    ("MAPE", "MAPE (%)"),
    ("R2", "R2"),
    ("CVAR_05", "CVaR 5%"),
    ("Time", "Time (ms)"),
]

escape_xml(x) = replace(string(x), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

function fmt(x)
    if !isfinite(x)
        return string(x)
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

function method_sort_key(m)
    idx = findfirst(==(m), METHOD_ORDER)
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
            push!(values, Float64(rows[1, metric]))
        end
    end
    return values
end

function scale_values(values; pad_fraction=0.08)
    finite_vals = filter(isfinite, values)
    if isempty(finite_vals)
        return (0.0, 1.0)
    end
    ymin = min(0.0, minimum(finite_vals))
    ymax = maximum(finite_vals)
    if ymax == ymin
        ymax = ymax + 1.0
    end
    pad = (ymax - ymin) * pad_fraction
    return (ymin - pad, ymax + pad)
end

function multiline_text(label, x, y; anchor="middle", size=11, fill="#333333")
    lines = split(label, "\n")
    out = IOBuffer()
    for (i, line) in enumerate(lines)
        dy = i == 1 ? 0 : size + 2
        println(out, """<text x="$x" y="$(y + dy)" text-anchor="$anchor" font-size="$size" fill="$fill">$(escape_xml(line))</text>""")
    end
    return String(take!(out))
end

function draw_metric_panel(df, dataset, metric, title, x0, y0, width, height)
    methods = ordered_methods(df)
    values = metric_values(df, metric, methods)
    ymin, ymax = scale_values(values)
    left = x0 + 54
    right = x0 + width - 12
    top = y0 + 36
    bottom = y0 + height - 72
    plot_w = right - left
    plot_h = bottom - top
    bar_gap = 8
    bar_w = max(8.0, (plot_w - bar_gap * (length(methods) + 1)) / length(methods))

    out = IOBuffer()
    println(out, """<rect x="$x0" y="$y0" width="$width" height="$height" fill="#ffffff" stroke="#d6d6d6"/>""")
    println(out, """<text x="$(x0 + 14)" y="$(y0 + 22)" font-size="15" font-weight="600" fill="#222222">$(escape_xml(title))</text>""")

    for t in 0:4
        val = ymin + (ymax - ymin) * t / 4
        y = bottom - (val - ymin) / (ymax - ymin) * plot_h
        println(out, """<line x1="$left" x2="$right" y1="$y" y2="$y" stroke="#e8e8e8"/>""")
        println(out, """<text x="$(left - 8)" y="$(y + 4)" text-anchor="end" font-size="10" fill="#666666">$(fmt(val))</text>""")
    end

    zero_y = bottom - (0 - ymin) / (ymax - ymin) * plot_h
    if top <= zero_y <= bottom
        println(out, """<line x1="$left" x2="$right" y1="$zero_y" y2="$zero_y" stroke="#9a9a9a"/>""")
    end

    for (i, method) in enumerate(methods)
        val = values[i]
        !isfinite(val) && continue
        x = left + bar_gap + (i - 1) * (bar_w + bar_gap)
        y = bottom - (max(val, 0) - ymin) / (ymax - ymin) * plot_h
        yzero = bottom - (0 - ymin) / (ymax - ymin) * plot_h
        h = abs(yzero - y)
        rect_y = min(y, yzero)
        color = get(COLORS, method, "#555555")
        println(out, """<rect x="$x" y="$rect_y" width="$bar_w" height="$h" rx="2" fill="$color"/>""")
        println(out, """<text x="$(x + bar_w / 2)" y="$(rect_y - 5)" text-anchor="middle" font-size="9" fill="#333333">$(fmt(val))</text>""")
        label = get(METHOD_LABELS, method, method)
        println(out, multiline_text(label, x + bar_w / 2, bottom + 15; size=9))
    end

    println(out, """<text x="$(x0 + width - 14)" y="$(y0 + height - 10)" text-anchor="end" font-size="10" fill="#777777">$(escape_xml(dataset))</text>""")
    return String(take!(out))
end

function write_metric_grid(path, df, dataset)
    panel_w = 440
    panel_h = 260
    gap = 18
    width = panel_w * 2 + gap + 40
    height = panel_h * 3 + gap * 2 + 78
    out = IOBuffer()
    println(out, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(out, """<rect width="100%" height="100%" fill="#f7f7f7"/>""")
    println(out, """<text x="20" y="32" font-size="22" font-weight="700" fill="#222222">Smoke-run metric comparison: $(escape_xml(dataset))</text>""")
    println(out, """<text x="20" y="54" font-size="12" fill="#666666">Bars compare the methods emitted by eval_method for one tested configuration. Lower is better except R2.</text>""")
    for (k, (metric, title)) in enumerate(METRICS)
        row = div(k - 1, 2)
        col = (k - 1) % 2
        x = 20 + col * (panel_w + gap)
        y = 76 + row * (panel_h + gap)
        println(out, draw_metric_panel(df, dataset, metric, title, x, y, panel_w, panel_h))
    end
    println(out, "</svg>")
    write(path, String(take!(out)))
end

function read_numeric_matrix(path)
    df = CSV.read(path, DataFrame)
    return Matrix{Float64}(df)
end

function beta_files_for(dataset)
    files = String[]
    if !isdir(BETA_DIR)
        return files
    end
    for file in readdir(BETA_DIR; join=true)
        name = basename(file)
        startswith(name, "results_$(dataset)_") || continue
        endswith(name, ".csv") || continue
        push!(files, file)
    end
    return sort(files)
end

function beta_method_name(dataset, file)
    stem = replace(basename(file), ".csv" => "")
    prefix = "results_$(dataset)_"
    rest = replace(stem, prefix => ""; count=1)
    parts = split(rest, "_")
    if length(parts) <= 3
        return rest
    end
    return join(parts[4:end], "_")
end

function draw_line_panel(path, method, x0, y0, width, height)
    mat = read_numeric_matrix(path)
    n, p = size(mat)
    left = x0 + 44
    right = x0 + width - 12
    top = y0 + 34
    bottom = y0 + height - 34
    plot_w = right - left
    plot_h = bottom - top
    vals = collect(skipmissing(vec(mat)))
    ymin, ymax = scale_values(collect(vals))
    colors = ["#2f6f9f", "#d95f02", "#4c9f70", "#7570b3", "#e7298a", "#a6761d", "#666666", "#1b9e77"]

    out = IOBuffer()
    println(out, """<rect x="$x0" y="$y0" width="$width" height="$height" fill="#ffffff" stroke="#d6d6d6"/>""")
    println(out, """<text x="$(x0 + 12)" y="$(y0 + 21)" font-size="13" font-weight="600" fill="#222222">$(escape_xml(get(METHOD_LABELS, method, method)))</text>""")
    for t in 0:3
        val = ymin + (ymax - ymin) * t / 3
        y = bottom - (val - ymin) / (ymax - ymin) * plot_h
        println(out, """<line x1="$left" x2="$right" y1="$y" y2="$y" stroke="#e8e8e8"/>""")
        println(out, """<text x="$(left - 7)" y="$(y + 3)" text-anchor="end" font-size="9" fill="#666666">$(fmt(val))</text>""")
    end
    for j in 1:p
        points = String[]
        for i in 1:n
            x = n == 1 ? (left + plot_w / 2) : left + (i - 1) / (n - 1) * plot_w
            y = bottom - (mat[i, j] - ymin) / (ymax - ymin) * plot_h
            push!(points, @sprintf("%.2f,%.2f", x, y))
        end
        color = colors[mod1(j, length(colors))]
        println(out, """<polyline points="$(join(points, " "))" fill="none" stroke="$color" stroke-width="1.8"/>""")
    end
    println(out, """<text x="$(right)" y="$(bottom + 20)" text-anchor="end" font-size="10" fill="#777777">test step</text>""")
    return String(take!(out))
end

function write_beta_grid(path, dataset, files)
    chosen_methods = [
        "adaptive_ridge_exact_err_rule",
        "adaptive_ridge_standard_err_rule",
        "adaptive_ridge_exact",
        "adaptive_ridge_standard",
        "PA",
        "bandits_full",
    ]
    by_method = Dict(beta_method_name(dataset, file) => file for file in files)
    selected = [(m, by_method[m]) for m in chosen_methods if haskey(by_method, m)]
    isempty(selected) && return false

    panel_w = 330
    panel_h = 210
    gap = 16
    width = panel_w * 2 + gap + 40
    rows = ceil(Int, length(selected) / 2)
    height = rows * panel_h + (rows - 1) * gap + 78

    out = IOBuffer()
    println(out, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(out, """<rect width="100%" height="100%" fill="#f7f7f7"/>""")
    println(out, """<text x="20" y="32" font-size="22" font-weight="700" fill="#222222">Smoke-run coefficient trajectories: $(escape_xml(dataset))</text>""")
    println(out, """<text x="20" y="54" font-size="12" fill="#666666">Each line is one coefficient/ensemble-member weight over the held-out evaluation steps.</text>""")
    for (k, (method, file)) in enumerate(selected)
        row = div(k - 1, 2)
        col = (k - 1) % 2
        x = 20 + col * (panel_w + gap)
        y = 76 + row * (panel_h + gap)
        println(out, draw_line_panel(file, method, x, y, panel_w, panel_h))
    end
    println(out, "</svg>")
    write(path, String(take!(out)))
    return true
end

function main()
    metric_outputs = String[]
    for file in sort(readdir(RESULTS_DIR; join=true))
        endswith(file, ".csv") || continue
        df = CSV.read(file, DataFrame)
        "Method" in names(df) || continue
        nrow(df) > 1 || continue
        dataset = string(df.Dataset[1])
        out = joinpath(OUT_DIR, "metrics_$(dataset).svg")
        write_metric_grid(out, df, dataset)
        push!(metric_outputs, out)
    end

    beta_outputs = String[]
    datasets = unique(replace(basename(file), r"^results_([^_]+).*" => s"\1") for file in readdir(BETA_DIR; join=true) if startswith(basename(file), "results_"))
    for dataset in sort(collect(datasets))
        files = beta_files_for(dataset)
        isempty(files) && continue
        out = joinpath(OUT_DIR, "coefficients_$(dataset).svg")
        if write_beta_grid(out, dataset, files)
            push!(beta_outputs, out)
        end
    end

    println("Metric plots:")
    foreach(println, metric_outputs)
    println("Coefficient plots:")
    foreach(println, beta_outputs)
end

main()
