using Flux
using FFTW
using Random
using Statistics

# FEDformer implementation inspired by:
# https://github.com/MAZiqing/FEDformer (MIT License)

const FEDFORMER_EPS = 1.0f-6

function all_finite(x)
    if x === nothing
        return true
    elseif x isa Number
        return isfinite(x)
    elseif x isa AbstractArray
        return all(isfinite, x)
    elseif x isa Tuple
        return all(all_finite, x)
    elseif x isa NamedTuple
        return all(all_finite, values(x))
    else
        return true
    end
end

function fedformer_gelu(x)
    return 0.5f0 .* x .* (1.0f0 .+ tanh.(sqrt(2.0f0 / Float32(pi)) .* (x .+ 0.044715f0 .* x .^ 3)))
end

function fedformer_activation(name::String)
    lname = lowercase(name)
    if lname == "gelu"
        return fedformer_gelu
    elseif lname == "relu"
        return x -> max.(x, 0)
    else
        return x -> x
    end
end

function positional_encoding(d_model::Int, seq_len::Int)
    pe = zeros(Float32, d_model, seq_len)
    for pos in 0:(seq_len - 1)
        pos_f = Float32(pos)
        for i in 0:((d_model ÷ 2) - 1)
            denom = 10000.0f0 ^ (2.0f0 * i / d_model)
            pe[2 * i + 1, pos + 1] = sin(pos_f / denom)
            if 2 * i + 2 <= d_model
                pe[2 * i + 2, pos + 1] = cos(pos_f / denom)
            end
        end
    end
    return pe
end

function apply_positional(x, pe)
    return x .+ reshape(pe, size(pe, 1), size(pe, 2), 1)
end

function apply_dense_time(layer::Dense, x)
    d_in, seq_len, batch = size(x)
    x2 = reshape(x, d_in, :)
    y2 = layer(x2)
    return reshape(y2, size(y2, 1), seq_len, batch)
end

function moving_avg(x, kernel::Int)
    if kernel <= 1
        return x
    end
    d_model, seq_len, batch = size(x)
    pad = (kernel - 1) ÷ 2
    left = repeat(view(x, :, 1:1, :), 1, pad, 1)
    right = repeat(view(x, :, seq_len:seq_len, :), 1, pad, 1)
    x_pad = cat(left, x, right; dims=2)
    out_buf = Flux.Zygote.Buffer(similar(x))
    for t in 1:seq_len
        window = view(x_pad, :, t:(t + kernel - 1), :)
        out_buf[:, t, :] = dropdims(mean(window; dims=2), dims=2)
    end
    return copy(out_buf)
end

function series_decomp(x, kernel::Int)
    trend = moving_avg(x, kernel)
    seasonal = x .- trend
    return seasonal, trend
end

function select_modes(seq_len::Int, modes::Int, mode_select::Symbol)
    max_modes = min(modes, seq_len ÷ 2 + 1)
    if mode_select == :random
        return sort!(randperm(seq_len ÷ 2 + 1)[1:max_modes])
    end
    return collect(1:max_modes)
end

struct FourierBlock
    weight_re::Array{Float32, 4}
    weight_im::Array{Float32, 4}
    modes::Vector{Int}
    seq_len::Int
    n_heads::Int
    head_dim::Int
end

Flux.@functor FourierBlock

function FourierBlock(d_model::Int, seq_len::Int; modes::Int=32, mode_select::Symbol=:low, n_heads::Int=1)
    mode_idx = select_modes(seq_len, modes, mode_select)
    n_heads = max(1, n_heads)
    if d_model % n_heads != 0
        n_heads = 1
    end
    head_dim = d_model ÷ n_heads
    weight_re = 0.02f0 .* randn(Float32, n_heads, head_dim, head_dim, length(mode_idx))
    weight_im = 0.02f0 .* randn(Float32, n_heads, head_dim, head_dim, length(mode_idx))
    return FourierBlock(weight_re, weight_im, mode_idx, seq_len, n_heads, head_dim)
end

function (fb::FourierBlock)(x)
    d_model, seq_len, batch = size(x)
    x_h = reshape(x, fb.head_dim, fb.n_heads, seq_len, batch)
    x_ft = rfft(x_h, 3)
    freq_len = size(x_ft, 3)
    out_buf = Flux.Zygote.Buffer(zeros(ComplexF32, fb.head_dim, fb.n_heads, freq_len, batch))
    for (i, idx) in enumerate(fb.modes)
        if idx > freq_len
            continue
        end
        for h in 1:fb.n_heads
            xr = real.(view(x_ft, :, h, idx, :))
            xi = imag.(view(x_ft, :, h, idx, :))
            w_re = view(fb.weight_re, h, :, :, i)
            w_im = view(fb.weight_im, h, :, :, i)
            yr = w_re * xr .- w_im * xi
            yi = w_re * xi .+ w_im * xr
            out_buf[:, h, idx, :] = complex.(yr, yi)
        end
    end
    out = irfft(copy(out_buf), seq_len, 3)
    return reshape(out, d_model, seq_len, batch)
end

struct WaveletBlock
    weight_a::Array{Float32, 4}
    weight_d::Array{Float32, 4}
    levels::Int
    seq_len::Int
    n_heads::Int
    head_dim::Int
end

Flux.@functor WaveletBlock

function WaveletBlock(d_model::Int, seq_len::Int; modes::Int=1, n_heads::Int=1)
    levels = max(1, modes)
    n_heads = max(1, n_heads)
    if d_model % n_heads != 0
        n_heads = 1
    end
    head_dim = d_model ÷ n_heads
    weight_a = 0.02f0 .* randn(Float32, n_heads, head_dim, head_dim, levels)
    weight_d = 0.02f0 .* randn(Float32, n_heads, head_dim, head_dim, levels)
    return WaveletBlock(weight_a, weight_d, levels, seq_len, n_heads, head_dim)
end

function haar_dwt(x)
    time_dim = ndims(x) == 3 ? 2 : 3
    seq_len = size(x, time_dim)
    if isodd(seq_len)
        last_slice = selectdim(x, time_dim, seq_len:seq_len)
        x = cat(x, last_slice; dims=time_dim)
        seq_len += 1
    end
    odd_idx = 1:2:seq_len
    even_idx = 2:2:seq_len
    x_odd = selectdim(x, time_dim, odd_idx)
    x_even = selectdim(x, time_dim, even_idx)
    approx = (x_odd .+ x_even) ./ sqrt(2.0f0)
    detail = (x_odd .- x_even) ./ sqrt(2.0f0)
    return approx, detail, seq_len
end

function haar_idwt(approx, detail, seq_len::Int)
    time_dim = ndims(approx) == 3 ? 2 : 3
    half_len = size(approx, time_dim)
    out_size = collect(size(approx))
    out_size[time_dim] = 2 * half_len
    out_buf = Flux.Zygote.Buffer(zeros(Float32, Tuple(out_size)))
    odd_idx = 1:2:(2 * half_len)
    even_idx = 2:2:(2 * half_len)
    selectdim(out_buf, time_dim, odd_idx) .= (approx .+ detail) ./ sqrt(2.0f0)
    selectdim(out_buf, time_dim, even_idx) .= (approx .- detail) ./ sqrt(2.0f0)
    return selectdim(copy(out_buf), time_dim, 1:seq_len)
end

function match_time_len(x, target_len::Int)
    time_dim = ndims(x) == 3 ? 2 : 3
    cur_len = size(x, time_dim)
    if cur_len == target_len
        return x
    elseif cur_len > target_len
        return selectdim(x, time_dim, 1:target_len)
    else
        pad_len = target_len - cur_len
        last_slice = selectdim(x, time_dim, cur_len:cur_len)
        pad = repeat(last_slice, ntuple(i -> i == time_dim ? pad_len : 1, ndims(x))...)
        return cat(x, pad; dims=time_dim)
    end
end

function apply_linear_time(weight, x)
    d_model, seq_len, batch = size(x)
    x2 = reshape(x, d_model, :)
    y2 = weight * x2
    return reshape(y2, d_model, seq_len, batch)
end

function (wb::WaveletBlock)(x)
    d_model, seq_len, batch = size(x)
    x_h = reshape(x, wb.head_dim, wb.n_heads, seq_len, batch)
    approx = x_h
    details = Vector{Array{Float32, 4}}()
    lengths = Int[]
    for level in 1:wb.levels
        approx, detail, len_lvl = haar_dwt(approx)
        push!(lengths, len_lvl)
        out_detail_buf = Flux.Zygote.Buffer(similar(detail))
        out_approx_buf = Flux.Zygote.Buffer(similar(approx))
        for h in 1:wb.n_heads
            out_detail_buf[:, h, :, :] = apply_linear_time(view(wb.weight_d, h, :, :, level), view(detail, :, h, :, :))
            out_approx_buf[:, h, :, :] = apply_linear_time(view(wb.weight_a, h, :, :, level), view(approx, :, h, :, :))
        end
        push!(details, copy(out_detail_buf))
        approx = copy(out_approx_buf)
    end
    x_rec = approx
    for level in wb.levels:-1:1
        x_rec = haar_idwt(x_rec, details[level], lengths[level])
    end
    x_rec = reshape(x_rec, d_model, size(x_rec, 3), batch)
    return x_rec[:, 1:seq_len, :]
end

struct FrequencyCross
    modes::Vector{Int}
    q_len::Int
    n_heads::Int
    head_dim::Int
end

Flux.@functor FrequencyCross

function FrequencyCross(q_len::Int, k_len::Int; modes::Int=32, mode_select::Symbol=:low, n_heads::Int=1, d_model::Int=1)
    min_len = min(q_len, k_len)
    mode_idx = select_modes(min_len, modes, mode_select)
    n_heads = max(1, n_heads)
    if d_model % n_heads != 0
        n_heads = 1
    end
    head_dim = d_model ÷ n_heads
    return FrequencyCross(mode_idx, q_len, n_heads, head_dim)
end

function (fc::FrequencyCross)(q, kv)
    d_model, q_len, batch = size(q)
    q_h = reshape(q, fc.head_dim, fc.n_heads, q_len, batch)
    kv_h = reshape(kv, fc.head_dim, fc.n_heads, size(kv, 2), batch)
    q_ft = rfft(q_h, 3)
    kv_ft = rfft(kv_h, 3)
    freq_len = size(q_ft, 3)
    out_buf = Flux.Zygote.Buffer(zeros(ComplexF32, fc.head_dim, fc.n_heads, freq_len, batch))
    for idx in fc.modes
        if idx > freq_len || idx > size(kv_ft, 3)
            continue
        end
        for h in 1:fc.n_heads
            q_m = view(q_ft, :, h, idx, :)
            kv_m = view(kv_ft, :, h, idx, :)
            attn = q_m .* conj.(kv_m)
            denom = sum(abs.(attn); dims=2) .+ FEDFORMER_EPS
            weight = attn ./ denom
            out_buf[:, h, idx, :] = weight .* kv_m
        end
    end
    out = irfft(copy(out_buf), q_len, 3)
    return reshape(out, d_model, q_len, batch)
end

struct WaveletCross
    levels::Int
    n_heads::Int
    head_dim::Int
end

Flux.@functor WaveletCross

function WaveletCross(; levels::Int=1, n_heads::Int=1, d_model::Int=1)
    levels = max(1, levels)
    n_heads = max(1, n_heads)
    if d_model % n_heads != 0
        n_heads = 1
    end
    head_dim = d_model ÷ n_heads
    return WaveletCross(levels, n_heads, head_dim)
end

function (wc::WaveletCross)(q, kv)
    d_model, q_len, batch = size(q)
    q_h = reshape(q, wc.head_dim, wc.n_heads, q_len, batch)
    kv_h = reshape(kv, wc.head_dim, wc.n_heads, size(kv, 2), batch)
    approx_q, detail_q, len_q = haar_dwt(q_h)
    approx_k, detail_k, _ = haar_dwt(kv_h)
    target_len = size(approx_q, 3)
    approx_k = match_time_len(approx_k, target_len)
    detail_k = match_time_len(detail_k, target_len)
    approx = approx_q .* approx_k
    detail = detail_q .* detail_k
    for _ in 2:wc.levels
        approx, detail, _ = haar_dwt(approx)
    end
    out = haar_idwt(approx, detail, len_q)
    out = reshape(out, d_model, size(out, 3), batch)
    return out[:, 1:q_len, :]
end

struct FeedForward
    fc1::Dense
    fc2::Dense
    activation
    dropout
end

Flux.@functor FeedForward

function FeedForward(d_model::Int, d_ff::Int, activation, dropout)
    return FeedForward(Dense(d_model, d_ff), Dense(d_ff, d_model), activation, Dropout(dropout))
end

function (ff::FeedForward)(x)
    y = apply_dense_time(ff.fc1, x)
    y = ff.activation(y)
    y = ff.dropout(y)
    y = apply_dense_time(ff.fc2, y)
    return y
end

struct EncoderLayer
    self_attn
    norm1::LayerNorm
    ffn::FeedForward
    norm2::LayerNorm
    dropout
end

Flux.@functor EncoderLayer

function (layer::EncoderLayer)(x)
    y = layer.self_attn(x)
    y = layer.dropout(y)
    x = layer.norm1(x .+ y)
    y2 = layer.ffn(x)
    y2 = layer.dropout(y2)
    x = layer.norm2(x .+ y2)
    return x
end

struct DecoderLayer
    self_attn
    cross_attn
    norm1::LayerNorm
    norm2::LayerNorm
    ffn::FeedForward
    norm3::LayerNorm
    dropout
end

Flux.@functor DecoderLayer

function (layer::DecoderLayer)(x, enc_out)
    y = layer.self_attn(x)
    y = layer.dropout(y)
    x = layer.norm1(x .+ y)
    y2 = layer.cross_attn(x, enc_out)
    y2 = layer.dropout(y2)
    x = layer.norm2(x .+ y2)
    y3 = layer.ffn(x)
    y3 = layer.dropout(y3)
    x = layer.norm3(x .+ y3)
    return x
end

struct Encoder
    layers::Vector{EncoderLayer}
end

Flux.@functor Encoder

function (enc::Encoder)(x)
    for layer in enc.layers
        x = layer(x)
    end
    return x
end

struct Decoder
    layers::Vector{DecoderLayer}
end

Flux.@functor Decoder

function (dec::Decoder)(x, enc_out)
    for layer in dec.layers
        x = layer(x, enc_out)
    end
    return x
end

struct FEDformer
    enc_embed::Dense
    dec_embed::Dense
    encoder::Encoder
    decoder::Decoder
    proj::Dense
    trend_proj::Dense
    pos_enc::Array{Float32, 2}
    pos_dec::Array{Float32, 2}
    moving_avg::Int
    seq_len::Int
    label_len::Int
    pred_len::Int
    input_dim::Int
end

Flux.@functor FEDformer

function (m::FEDformer)(x_enc, x_dec)
    seasonal_enc, _ = series_decomp(x_enc, m.moving_avg)
    seasonal_dec, trend_dec = series_decomp(x_dec, m.moving_avg)

    enc_in = apply_positional(apply_dense_time(m.enc_embed, seasonal_enc), m.pos_enc)
    dec_in = apply_positional(apply_dense_time(m.dec_embed, seasonal_dec), m.pos_dec)

    enc_out = m.encoder(enc_in)
    dec_out = m.decoder(dec_in, enc_out)

    dec_out = dec_out[:, end - m.pred_len + 1:end, :]
    trend_out = trend_dec[:, end - m.pred_len + 1:end, :]

    seasonal_pred = apply_dense_time(m.proj, dec_out)
    trend_pred = apply_dense_time(m.trend_proj, trend_out)

    return seasonal_pred .+ trend_pred
end

function build_fedformer(input_dim::Int;
        seq_len::Int=24,
        label_len::Int=12,
        pred_len::Int=1,
        d_model::Int=64,
        n_heads::Int=4,
        e_layers::Int=2,
        d_layers::Int=1,
        d_ff::Int=128,
        dropout::Float64=0.1,
        activation::String="gelu",
        moving_avg::Int=5,
        freq_mode::String="fourier",
        modes::Int=16)

    freq = Symbol(lowercase(freq_mode))
    act = fedformer_activation(activation)
    enc_embed = Dense(input_dim, d_model)
    dec_embed = Dense(input_dim, d_model)
    pos_enc = positional_encoding(d_model, seq_len)
    pos_dec = positional_encoding(d_model, label_len + pred_len)

    enc_layers = EncoderLayer[]
    for _ in 1:e_layers
        if freq == :wavelet
            self_attn = WaveletBlock(d_model, seq_len; modes=modes, n_heads=n_heads)
        else
            self_attn = FourierBlock(d_model, seq_len; modes=modes, n_heads=n_heads)
        end
        push!(enc_layers, EncoderLayer(self_attn, LayerNorm(d_model), FeedForward(d_model, d_ff, act, dropout), LayerNorm(d_model), Dropout(dropout)))
    end
    encoder = Encoder(enc_layers)

    dec_layers = DecoderLayer[]
    for _ in 1:d_layers
        if freq == :wavelet
            self_attn = WaveletBlock(d_model, label_len + pred_len; modes=modes, n_heads=n_heads)
            cross_attn = WaveletCross(levels=modes, n_heads=n_heads, d_model=d_model)
        else
            self_attn = FourierBlock(d_model, label_len + pred_len; modes=modes, n_heads=n_heads)
            cross_attn = FrequencyCross(label_len + pred_len, seq_len; modes=modes, n_heads=n_heads, d_model=d_model)
        end
        push!(dec_layers, DecoderLayer(self_attn, cross_attn, LayerNorm(d_model), LayerNorm(d_model), FeedForward(d_model, d_ff, act, dropout), LayerNorm(d_model), Dropout(dropout)))
    end
    decoder = Decoder(dec_layers)

    proj = Dense(d_model, 1)
    trend_proj = Dense(input_dim, 1)

    moving_avg = min(moving_avg, seq_len)

    return FEDformer(enc_embed, dec_embed, encoder, decoder, proj, trend_proj,
        pos_enc, pos_dec, moving_avg, seq_len, label_len, pred_len, input_dim)
end

function build_fedformer_dataset(X, y, seq_len, label_len, pred_len, start_idx, end_idx)
    n, input_dim = size(X)
    last_t = end_idx - pred_len
    first_t = start_idx + seq_len - 1
    if last_t < first_t
        return nothing
    end
    num_samples = last_t - first_t + 1
    x_enc = zeros(Float32, input_dim, seq_len, num_samples)
    x_dec = zeros(Float32, input_dim, label_len + pred_len, num_samples)
    y_out = zeros(Float32, 1, pred_len, num_samples)
    for i in 1:num_samples
        t = first_t + i - 1
        enc_slice = permutedims(X[(t - seq_len + 1):t, :], (2, 1))
        dec_slice = permutedims(X[(t - label_len + 1):(t + pred_len), :], (2, 1))
        x_enc[:, :, i] = enc_slice
        x_dec[:, :, i] = dec_slice
        y_out[1, :, i] = y[(t + 1):(t + pred_len)]
    end
    return x_enc, x_dec, y_out
end

function slice_with_pad(X, start_idx, end_idx, desired_len)
    n, input_dim = size(X)
    left_pad = max(1 - start_idx, 0)
    right_pad = max(end_idx - n, 0)
    start_idx = max(start_idx, 1)
    end_idx = min(end_idx, n)
    slice = X[start_idx:end_idx, :]
    if left_pad > 0
        left = repeat(view(X, 1:1, :), left_pad, 1)
        slice = vcat(left, slice)
    end
    if right_pad > 0
        right = repeat(view(X, n:n, :), right_pad, 1)
        slice = vcat(slice, right)
    end
    if size(slice, 1) != desired_len
        # final guard: truncate or pad with last row
        if size(slice, 1) > desired_len
            slice = slice[end - desired_len + 1:end, :]
        else
            pad_len = desired_len - size(slice, 1)
            last_row = view(slice, size(slice, 1):size(slice, 1), :)
            slice = vcat(slice, repeat(last_row, pad_len, 1))
        end
    end
    return slice
end

function build_fedformer_predict_inputs(X, seq_len, label_len, pred_len, t_idx)
    enc_start = t_idx - seq_len + 1
    enc_end = t_idx
    dec_start = t_idx - label_len + 1
    dec_end = t_idx + pred_len
    enc_slice = slice_with_pad(X, enc_start, enc_end, seq_len)
    dec_slice = slice_with_pad(X, dec_start, dec_end, label_len + pred_len)
    enc_slice = permutedims(enc_slice, (2, 1))
    dec_slice = permutedims(dec_slice, (2, 1))
    return enc_slice, dec_slice
end

function train_fedformer!(model::FEDformer, x_enc, x_dec, y_out;
        epochs::Int=5, batch_size::Int=16, lr::Float64=1e-3, verbose::Bool=false)
    opt_state = Flux.setup(Flux.Adam(lr), model)
    n = size(x_enc, 3)
    Flux.trainmode!(model)
    for epoch in 1:epochs
        idx = randperm(n)
        for i in 1:batch_size:n
            batch_idx = idx[i:min(i + batch_size - 1, n)]
            xb_enc = view(x_enc, :, :, batch_idx)
            xb_dec = view(x_dec, :, :, batch_idx)
            yb = view(y_out, :, :, batch_idx)
            grads = Flux.gradient(model) do m
                y_pred = m(xb_enc, xb_dec)
                Flux.Losses.mse(y_pred, yb)
            end
            if all_finite(grads[1])
                Flux.update!(opt_state, model, grads[1])
            end
        end
        if verbose
            y_pred = model(x_enc, x_dec)
            println("FEDformer epoch ", epoch, " loss=", Flux.Losses.mse(y_pred, y_out))
        end
    end
    Flux.testmode!(model)
end

function fedformer_train_predict(args, X, y, train_start, train_end, split_index, val)
    if haskey(args, "seed")
        Random.seed!(Int(args["seed"]))
    end
    if val <= 0
        return nothing
    end
    X = Array{Float32}(X)
    y = Array{Float32}(y)
    input_dim = size(X, 2)
    seq_len = get(args, "fedformer_seq_len", 24)
    label_len = get(args, "fedformer_label_len", 12)
    pred_len = get(args, "fedformer_pred_len", 1)
    d_model = get(args, "fedformer_d_model", 64)
    n_heads = get(args, "fedformer_n_heads", 4)
    e_layers = get(args, "fedformer_e_layers", 2)
    d_layers = get(args, "fedformer_d_layers", 1)
    d_ff = get(args, "fedformer_d_ff", 128)
    dropout = get(args, "fedformer_dropout", 0.1)
    activation = get(args, "fedformer_activation", "gelu")
    moving_avg = get(args, "fedformer_moving_avg", 5)
    freq_mode = get(args, "fedformer_freq_mode", "fourier")
    modes = get(args, "fedformer_modes", 16)
    epochs = get(args, "fedformer_epochs", 5)
    batch_size = get(args, "fedformer_batch_size", 16)
    lr = get(args, "fedformer_lr", 1e-3)

    label_len = min(label_len, seq_len)
    moving_avg = min(moving_avg, seq_len)

    n = size(X, 1)
    train_start = max(train_start, 1)
    train_end = min(train_end, n - pred_len)
    if train_end < train_start
        return nothing
    end

    if train_end - train_start + 1 < seq_len + pred_len
        return nothing
    end

    model = build_fedformer(input_dim;
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        moving_avg=moving_avg,
        freq_mode=freq_mode,
        modes=modes)

    dataset = build_fedformer_dataset(X, y, seq_len, label_len, pred_len, train_start, train_end)
    if dataset === nothing
        return nothing
    end
    x_enc, x_dec, y_out = dataset

    train_fedformer!(model, x_enc, x_dec, y_out;
        epochs=epochs, batch_size=batch_size, lr=lr, verbose=false)

    preds = zeros(Float32, val)
    for s in 1:val
        t_idx = split_index + s - 1
        enc_slice, dec_slice = build_fedformer_predict_inputs(X, seq_len, label_len, pred_len, t_idx)
        enc_in = reshape(enc_slice, input_dim, seq_len, 1)
        dec_in = reshape(dec_slice, input_dim, label_len + pred_len, 1)
        y_pred = model(enc_in, dec_in)
        preds[s] = y_pred[1, 1, 1]
    end
    return preds
end
