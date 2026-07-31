const _SVG_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#000000", "#999999", "#F0E442", "#882255", "#44AA99", "#117733",
]

function _svg_escape(value)
    text = string(value)
    text = replace(text, "&" => "&amp;")
    text = replace(text, "<" => "&lt;")
    text = replace(text, ">" => "&gt;")
    text = replace(text, "\"" => "&quot;")
    return replace(text, "'" => "&apos;")
end

function _svg_begin(io, width::Int, height::Int, title::AbstractString)
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
    println(io, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>")
    println(io, "<text x=\"$(width / 2)\" y=\"28\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"18\" font-weight=\"bold\">$(_svg_escape(title))</text>")
end

_svg_end(io) = println(io, "</svg>")

function _atomic_svg_write(draw::Function, path::AbstractString)
    mkpath(dirname(path))
    temporary = _temporary_path(path)
    try
        open(temporary, "w") do io
            draw(io)
            flush(io)
        end
        filesize(temporary) > 0 || error("SVG renderer produced an empty file: $(path)")
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

function _finite_vector(values::AbstractVector{<:Real}, label::AbstractString)
    converted = Float64.(values)
    isempty(converted) && throw(ArgumentError("$(label) must not be empty"))
    all(isfinite, converted) || throw(ArgumentError("$(label) contains NaN or Inf"))
    return converted
end

function _expanded_range(minimum_value::Float64, maximum_value::Float64; include_zero::Bool=false)
    include_zero && (minimum_value = min(minimum_value, 0.0); maximum_value = max(maximum_value, 0.0))
    if minimum_value == maximum_value
        scale = max(abs(minimum_value), 1.0)
        return minimum_value - 0.05 * scale, maximum_value + 0.05 * scale
    end
    padding = 0.06 * (maximum_value - minimum_value)
    return minimum_value - padding, maximum_value + padding
end

function svg_bar_chart(path::AbstractString, labels::Vector{String}, values::Vector{<:Real};
                       title::String="", ylabel::String="", lower_better::Bool=true)
    length(labels) == length(values) || throw(DimensionMismatch("labels and values differ in length"))
    vals = _finite_vector(values, "bar values")

    width, height = 1000, 580
    left, right, top, bottom = 105, 35, 65, 165
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_slot = plot_width / length(vals)
    y_min, y_max = _expanded_range(minimum(vals), maximum(vals); include_zero=true)
    y_span = y_max - y_min
    pixel_y(value) = top + plot_height - plot_height * (value - y_min) / y_span
    baseline = pixel_y(0.0)

    order = sortperm(vals; rev=!lower_better)
    ranks = zeros(Int, length(vals))
    for (rank, index) in enumerate(order)
        ranks[index] = rank
    end

    return _atomic_svg_write(path) do io
        _svg_begin(io, width, height, title)
        for tick in 0:5
            value = y_min + tick * y_span / 5
            py = pixel_y(value)
            println(io, "<line x1=\"$left\" y1=\"$py\" x2=\"$(left + plot_width)\" y2=\"$py\" stroke=\"#dddddd\"/>")
            println(io, "<text x=\"$(left - 8)\" y=\"$(py + 4)\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"12\">$(@sprintf("%.4g", value))</text>")
        end
        println(io, "<line x1=\"$left\" y1=\"$baseline\" x2=\"$(left + plot_width)\" y2=\"$baseline\" stroke=\"black\"/>")
        println(io, "<line x1=\"$left\" y1=\"$top\" x2=\"$left\" y2=\"$(top + plot_height)\" stroke=\"black\"/>")

        for index in eachindex(vals)
            x = left + (index - 1) * bar_slot + 0.12 * bar_slot
            bar_width = 0.76 * bar_slot
            value_y = pixel_y(vals[index])
            rectangle_y = min(value_y, baseline)
            rectangle_height = max(abs(baseline - value_y), 1.0)
            color = _SVG_COLORS[mod1(ranks[index], length(_SVG_COLORS))]
            println(io, "<rect x=\"$x\" y=\"$rectangle_y\" width=\"$bar_width\" height=\"$rectangle_height\" fill=\"$color\"/>")
            label_y = vals[index] >= 0 ? value_y - 6 : value_y + 15
            println(io, "<text x=\"$(x + bar_width / 2)\" y=\"$label_y\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"11\">$(@sprintf("%.5g", vals[index]))</text>")
            println(io, "<text transform=\"translate($(x + bar_width / 2),$(top + plot_height + 14)) rotate(50)\" text-anchor=\"start\" font-family=\"Arial, sans-serif\" font-size=\"12\">$(_svg_escape(labels[index]))</text>")
        end
        println(io, "<text transform=\"translate(27,$(top + plot_height / 2)) rotate(-90)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"14\">$(_svg_escape(ylabel))</text>")
        _svg_end(io)
    end
end

function _clean_line_series(series::AbstractDict, logx::Bool, logy::Bool)
    cleaned = Vector{Tuple{String,Vector{Float64},Vector{Float64}}}()
    for (name, pair) in sort(collect(pairs(series)); by=item -> string(first(item)))
        xs_raw, ys_raw = pair
        length(xs_raw) == length(ys_raw) || throw(DimensionMismatch("line series $(name) has different x/y lengths"))
        xs = Float64[]
        ys = Float64[]
        for (x_raw, y_raw) in zip(xs_raw, ys_raw)
            x = Float64(x_raw)
            y = Float64(y_raw)
            isfinite(x) && isfinite(y) || continue
            logx && x <= 0 && continue
            logy && y <= 0 && continue
            push!(xs, x)
            push!(ys, y)
        end
        isempty(xs) || push!(cleaned, (string(name), xs, ys))
    end
    isempty(cleaned) && throw(ArgumentError("line chart has no finite plottable points"))
    return cleaned
end

function svg_line_chart(path::AbstractString,
                        series::Dict{String,Tuple{Vector{Float64},Vector{Float64}}};
                        title::String="", xlabel::String="", ylabel::String="",
                        logx::Bool=false, logy::Bool=false)
    cleaned = _clean_line_series(series, logx, logy)
    width, height = 1000, 620
    left, right, top, bottom = 100, 250, 60, 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    transform_x(value) = logx ? log10(value) : value
    transform_y(value) = logy ? log10(value) : value
    all_x = reduce(vcat, [transform_x.(entry[2]) for entry in cleaned])
    all_y = reduce(vcat, [transform_y.(entry[3]) for entry in cleaned])
    x_min, x_max = _expanded_range(minimum(all_x), maximum(all_x))
    y_min, y_max = _expanded_range(minimum(all_y), maximum(all_y))
    pixel_x(value) = left + plot_width * (transform_x(value) - x_min) / (x_max - x_min)
    pixel_y(value) = top + plot_height - plot_height * (transform_y(value) - y_min) / (y_max - y_min)

    return _atomic_svg_write(path) do io
        _svg_begin(io, width, height, title)
        for tick in 0:5
            transformed_x = x_min + tick * (x_max - x_min) / 5
            transformed_y = y_min + tick * (y_max - y_min) / 5
            px = left + plot_width * tick / 5
            py = top + plot_height - plot_height * tick / 5
            x_value = logx ? 10.0^transformed_x : transformed_x
            y_value = logy ? 10.0^transformed_y : transformed_y
            println(io, "<line x1=\"$px\" y1=\"$top\" x2=\"$px\" y2=\"$(top + plot_height)\" stroke=\"#eeeeee\"/>")
            println(io, "<line x1=\"$left\" y1=\"$py\" x2=\"$(left + plot_width)\" y2=\"$py\" stroke=\"#eeeeee\"/>")
            println(io, "<text x=\"$px\" y=\"$(top + plot_height + 20)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"11\">$(@sprintf("%.4g", x_value))</text>")
            println(io, "<text x=\"$(left - 8)\" y=\"$(py + 4)\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"11\">$(@sprintf("%.4g", y_value))</text>")
        end
        println(io, "<rect x=\"$left\" y=\"$top\" width=\"$plot_width\" height=\"$plot_height\" fill=\"none\" stroke=\"black\"/>")

        for (index, (name, xs, ys)) in enumerate(cleaned)
            points = join(("$(pixel_x(x)),$(pixel_y(y))" for (x, y) in zip(xs, ys)), " ")
            color = _SVG_COLORS[mod1(index, length(_SVG_COLORS))]
            println(io, "<polyline points=\"$points\" fill=\"none\" stroke=\"$color\" stroke-width=\"2.2\"/>")
            if length(xs) <= 100
                for (x, y) in zip(xs, ys)
                    println(io, "<circle cx=\"$(pixel_x(x))\" cy=\"$(pixel_y(y))\" r=\"2.7\" fill=\"$color\"/>")
                end
            end
            legend_y = top + 22 * index
            println(io, "<line x1=\"$(left + plot_width + 20)\" y1=\"$legend_y\" x2=\"$(left + plot_width + 50)\" y2=\"$legend_y\" stroke=\"$color\" stroke-width=\"3\"/>")
            println(io, "<text x=\"$(left + plot_width + 58)\" y=\"$(legend_y + 4)\" font-family=\"Arial, sans-serif\" font-size=\"12\">$(_svg_escape(name))</text>")
        end
        println(io, "<text x=\"$(left + plot_width / 2)\" y=\"$(height - 18)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"14\">$(_svg_escape(xlabel))</text>")
        println(io, "<text transform=\"translate(25,$(top + plot_height / 2)) rotate(-90)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"14\">$(_svg_escape(ylabel))</text>")
        _svg_end(io)
    end
end

function svg_heatmap(path::AbstractString, xlabels, ylabels, values::AbstractMatrix;
                     title::String="", xlabel::String="", ylabel::String="")
    n_y, n_x = size(values)
    n_x > 0 && n_y > 0 || throw(ArgumentError("heatmap matrix must be nonempty"))
    length(xlabels) == n_x || throw(DimensionMismatch("x labels differ from heatmap columns"))
    length(ylabels) == n_y || throw(DimensionMismatch("y labels differ from heatmap rows"))
    numeric = Matrix{Float64}(values)
    all(isfinite, numeric) || throw(ArgumentError("heatmap contains NaN or Inf"))

    width, height = 930, 670
    left, right, top, bottom = 145, 80, 70, 135
    cell_width = (width - left - right) / n_x
    cell_height = (height - top - bottom) / n_y
    z_min, z_max = extrema(numeric)
    span = max(z_max - z_min, eps(Float64))

    return _atomic_svg_write(path) do io
        _svg_begin(io, width, height, title)
        for row in 1:n_y, column in 1:n_x
            fraction = clamp((numeric[row, column] - z_min) / span, 0.0, 1.0)
            red = round(Int, 255 * fraction)
            blue = round(Int, 255 * (1 - fraction))
            green = round(Int, 180 * (1 - abs(2 * fraction - 1)))
            color = @sprintf("#%02x%02x%02x", red, green, blue)
            x = left + (column - 1) * cell_width
            y = top + (row - 1) * cell_height
            println(io, "<rect x=\"$x\" y=\"$y\" width=\"$cell_width\" height=\"$cell_height\" fill=\"$color\" stroke=\"white\"/>")
            println(io, "<text x=\"$(x + cell_width / 2)\" y=\"$(y + cell_height / 2 + 4)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"10\">$(@sprintf("%.4g", numeric[row, column]))</text>")
        end
        for column in 1:n_x
            println(io, "<text transform=\"translate($(left + (column - 0.5) * cell_width),$(top + n_y * cell_height + 14)) rotate(45)\" font-family=\"Arial, sans-serif\" font-size=\"11\">$(_svg_escape(xlabels[column]))</text>")
        end
        for row in 1:n_y
            println(io, "<text x=\"$(left - 8)\" y=\"$(top + (row - 0.5) * cell_height + 4)\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"11\">$(_svg_escape(ylabels[row]))</text>")
        end
        println(io, "<text x=\"$(left + n_x * cell_width / 2)\" y=\"$(height - 18)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"14\">$(_svg_escape(xlabel))</text>")
        println(io, "<text transform=\"translate(25,$(top + n_y * cell_height / 2)) rotate(-90)\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"14\">$(_svg_escape(ylabel))</text>")
        _svg_end(io)
    end
end
