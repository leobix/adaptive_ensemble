using Flux
using Random
using Statistics

# Informer-style transformer baseline for adaptive ensemble experiments.
# The public wrapper mirrors fedformer_train_predict so both transformer
# baselines can be trained and evaluated from eval2.jl in the same way.

function informer_all_finite(x)
    if x === nothing
        return true
    elseif x isa Number
        return isfinite(x)
    elseif x isa AbstractArray{<:Number}
        return all(isfinite, x)
    elseif x isa AbstractArray
        return all(informer_all_finite, x)
    elseif x isa Tuple
        return all(informer_all_finite, x)
    elseif x isa NamedTuple
        return all(informer_all_finite, values(x))
    else
        return true
    end
end

function informer_normalize_features(X::Array{Float32, 2}, train_start::Int, train_end::Int)
    train_view = view(X, train_start:train_end, :)
    mu = Float32.(mean(train_view; dims=1))
    sigma = Float32.(std(train_view; dims=1))
    sigma = map(s -> isfinite(s) && s > 1.0f-6 ? s : 1.0f0, sigma)
    return (X .- mu) ./ sigma
end

function informer_gelu(x)
    return 0.5f0 .* x .* (1.0f0 .+ tanh.(sqrt(2.0f0 / Float32(pi)) .* (x .+ 0.044715f0 .* x .^ 3)))
end

function informer_activation(name::String)
    lname = lowercase(name)
    if lname == "gelu"
        return informer_gelu
    elseif lname == "relu"
        return x -> max.(x, 0)
    else
        return x -> x
    end
end

function informer_positional_encoding(d_model::Int, seq_len::Int)
    pe = zeros(Float32, d_model, seq_len)
    for pos in 0:(seq_len - 1)
        pos_f = Float32(pos)
        for i in 0:(div(d_model, 2) - 1)
            denom = 10000.0f0 ^ (2.0f0 * i / d_model)
            pe[2 * i + 1, pos + 1] = sin(pos_f / denom)
            if 2 * i + 2 <= d_model
                pe[2 * i + 2, pos + 1] = cos(pos_f / denom)
            end
        end
    end
    return pe
end

function informer_apply_positional(x, pe)
    return x .+ reshape(pe, size(pe, 1), size(pe, 2), 1)
end

function informer_apply_dense_time(layer::Dense, x)
    d_in, seq_len, batch = size(x)
    x2 = reshape(x, d_in, :)
    y2 = layer(x2)
    return reshape(y2, size(y2, 1), seq_len, batch)
end

struct InformerProbAttention
    q_proj::Dense
    k_proj::Dense
    v_proj::Dense
    out_proj::Dense
    n_heads::Int
    head_dim::Int
    factor::Int
    causal::Bool
    dropout
end

Flux.@functor InformerProbAttention

function InformerProbAttention(d_model::Int; n_heads::Int=4, factor::Int=5, causal::Bool=false, dropout::Float64=0.1)
    n_heads = max(1, n_heads)
    if d_model % n_heads != 0
        n_heads = 1
    end
    head_dim = div(d_model, n_heads)
    return InformerProbAttention(
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        n_heads,
        head_dim,
        max(1, factor),
        causal,
        Dropout(dropout))
end

function informer_select_keys(score_col, allowed_len::Int, factor::Int)
    budget = min(allowed_len, max(1, factor * ceil(Int, log(allowed_len + 1))))
    if budget >= allowed_len
        return collect(1:allowed_len)
    end

    # Selection is discrete. Gradients still flow through the selected scores.
    return Flux.Zygote.ignore() do
        partialsortperm(collect(score_col[1:allowed_len]), 1:budget; rev=true)
    end
end

function (attn::InformerProbAttention)(x)
    return attn(x, x, x)
end

function (attn::InformerProbAttention)(q_in, k_in, v_in)
    q = informer_apply_dense_time(attn.q_proj, q_in)
    k = informer_apply_dense_time(attn.k_proj, k_in)
    v = informer_apply_dense_time(attn.v_proj, v_in)

    d_model, q_len, batch = size(q)
    k_len = size(k, 2)

    q_h = reshape(q, attn.head_dim, attn.n_heads, q_len, batch)
    k_h = reshape(k, attn.head_dim, attn.n_heads, k_len, batch)
    v_h = reshape(v, attn.head_dim, attn.n_heads, k_len, batch)

    out_buf = Flux.Zygote.Buffer(zeros(Float32, attn.head_dim, attn.n_heads, q_len, batch))
    scale = sqrt(Float32(attn.head_dim))

    for b in 1:batch
        for h in 1:attn.n_heads
            qh = view(q_h, :, h, :, b)
            kh = view(k_h, :, h, :, b)
            vh = view(v_h, :, h, :, b)
            scores = (transpose(kh) * qh) ./ scale
            for qi in 1:q_len
                allowed_len = attn.causal ? min(qi, k_len) : k_len
                idx = informer_select_keys(view(scores, :, qi), allowed_len, attn.factor)
                selected_scores = scores[idx, qi]
                weights = Flux.softmax(selected_scores)
                weights = attn.dropout(weights)
                out_buf[:, h, qi, b] = vh[:, idx] * weights
            end
        end
    end

    out = reshape(copy(out_buf), d_model, q_len, batch)
    return informer_apply_dense_time(attn.out_proj, out)
end

struct InformerFeedForward
    fc1::Dense
    fc2::Dense
    activation
    dropout
end

Flux.@functor InformerFeedForward

function InformerFeedForward(d_model::Int, d_ff::Int, activation, dropout)
    return InformerFeedForward(Dense(d_model, d_ff), Dense(d_ff, d_model), activation, Dropout(dropout))
end

function (ff::InformerFeedForward)(x)
    y = informer_apply_dense_time(ff.fc1, x)
    y = ff.activation(y)
    y = ff.dropout(y)
    y = informer_apply_dense_time(ff.fc2, y)
    return y
end

struct InformerEncoderLayer
    self_attn::InformerProbAttention
    norm1::LayerNorm
    ffn::InformerFeedForward
    norm2::LayerNorm
    dropout
end

Flux.@functor InformerEncoderLayer

function (layer::InformerEncoderLayer)(x)
    y = layer.self_attn(x)
    y = layer.dropout(y)
    x = layer.norm1(x .+ y)
    y2 = layer.ffn(x)
    y2 = layer.dropout(y2)
    x = layer.norm2(x .+ y2)
    return x
end

struct InformerDecoderLayer
    self_attn::InformerProbAttention
    cross_attn::InformerProbAttention
    norm1::LayerNorm
    norm2::LayerNorm
    ffn::InformerFeedForward
    norm3::LayerNorm
    dropout
end

Flux.@functor InformerDecoderLayer

function (layer::InformerDecoderLayer)(x, enc_out)
    y = layer.self_attn(x)
    y = layer.dropout(y)
    x = layer.norm1(x .+ y)
    y2 = layer.cross_attn(x, enc_out, enc_out)
    y2 = layer.dropout(y2)
    x = layer.norm2(x .+ y2)
    y3 = layer.ffn(x)
    y3 = layer.dropout(y3)
    x = layer.norm3(x .+ y3)
    return x
end

function informer_downsample(x)
    _, seq_len, _ = size(x)
    if seq_len <= 1
        return x
    end
    if isodd(seq_len)
        x = cat(x, view(x, :, seq_len:seq_len, :); dims=2)
        seq_len += 1
    end
    return (x[:, 1:2:seq_len, :] .+ x[:, 2:2:seq_len, :]) ./ 2.0f0
end

struct InformerEncoder
    layers::Vector{InformerEncoderLayer}
    distil::Bool
end

Flux.@functor InformerEncoder

function (enc::InformerEncoder)(x)
    for (i, layer) in enumerate(enc.layers)
        x = layer(x)
        if enc.distil && i < length(enc.layers)
            x = informer_downsample(x)
        end
    end
    return x
end

struct InformerDecoder
    layers::Vector{InformerDecoderLayer}
end

Flux.@functor InformerDecoder

function (dec::InformerDecoder)(x, enc_out)
    for layer in dec.layers
        x = layer(x, enc_out)
    end
    return x
end

struct Informer
    enc_embed::Dense
    dec_embed::Dense
    encoder::InformerEncoder
    decoder::InformerDecoder
    proj::Dense
    pos_enc::Array{Float32, 2}
    pos_dec::Array{Float32, 2}
    seq_len::Int
    label_len::Int
    pred_len::Int
    input_dim::Int
end

Flux.@functor Informer

function (m::Informer)(x_enc, x_dec)
    enc_in = informer_apply_positional(informer_apply_dense_time(m.enc_embed, x_enc), m.pos_enc)
    dec_in = informer_apply_positional(informer_apply_dense_time(m.dec_embed, x_dec), m.pos_dec)

    enc_out = m.encoder(enc_in)
    dec_out = m.decoder(dec_in, enc_out)
    dec_out = dec_out[:, end - m.pred_len + 1:end, :]

    return informer_apply_dense_time(m.proj, dec_out)
end

function build_informer(input_dim::Int;
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
        factor::Int=5,
        distil::Bool=false)

    act = informer_activation(activation)
    enc_embed = Dense(input_dim, d_model)
    dec_embed = Dense(input_dim, d_model)
    pos_enc = informer_positional_encoding(d_model, seq_len)
    pos_dec = informer_positional_encoding(d_model, label_len + pred_len)

    enc_layers = InformerEncoderLayer[]
    cur_len = seq_len
    for _ in 1:e_layers
        self_attn = InformerProbAttention(d_model; n_heads=n_heads, factor=factor, causal=false, dropout=dropout)
        push!(enc_layers, InformerEncoderLayer(self_attn, LayerNorm(d_model),
            InformerFeedForward(d_model, d_ff, act, dropout), LayerNorm(d_model), Dropout(dropout)))
        if distil
            cur_len = max(1, div(cur_len + 1, 2))
        end
    end
    encoder = InformerEncoder(enc_layers, distil)

    dec_layers = InformerDecoderLayer[]
    for _ in 1:d_layers
        self_attn = InformerProbAttention(d_model; n_heads=n_heads, factor=factor, causal=true, dropout=dropout)
        cross_attn = InformerProbAttention(d_model; n_heads=n_heads, factor=factor, causal=false, dropout=dropout)
        push!(dec_layers, InformerDecoderLayer(self_attn, cross_attn, LayerNorm(d_model), LayerNorm(d_model),
            InformerFeedForward(d_model, d_ff, act, dropout), LayerNorm(d_model), Dropout(dropout)))
    end
    decoder = InformerDecoder(dec_layers)

    proj = Dense(d_model, 1)

    return Informer(enc_embed, dec_embed, encoder, decoder, proj,
        pos_enc, pos_dec, seq_len, label_len, pred_len, input_dim)
end

function build_informer_dataset(X, y, seq_len, label_len, pred_len, start_idx, end_idx)
    _, input_dim = size(X)
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

function informer_slice_with_pad(X, start_idx, end_idx, desired_len)
    n, _ = size(X)
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

function build_informer_predict_inputs(X, seq_len, label_len, pred_len, t_idx)
    enc_start = t_idx - seq_len + 1
    enc_end = t_idx
    dec_start = t_idx - label_len + 1
    dec_end = t_idx + pred_len
    enc_slice = informer_slice_with_pad(X, enc_start, enc_end, seq_len)
    dec_slice = informer_slice_with_pad(X, dec_start, dec_end, label_len + pred_len)
    enc_slice = permutedims(enc_slice, (2, 1))
    dec_slice = permutedims(dec_slice, (2, 1))
    return enc_slice, dec_slice
end

function train_informer!(model::Informer, x_enc, x_dec, y_out;
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
            if informer_all_finite(grads[1])
                Flux.update!(opt_state, model, grads[1])
            end
        end
        if verbose
            y_pred = model(x_enc, x_dec)
            println("Informer epoch ", epoch, " loss=", Flux.Losses.mse(y_pred, y_out))
        end
    end
    Flux.testmode!(model)
end

function informer_train_predict(args, X, y, train_start, train_end, split_index, val)
    if haskey(args, "seed")
        Random.seed!(Int(args["seed"]))
    end
    if val <= 0
        return nothing
    end
    X = Array{Float32}(X)
    y = Array{Float32}(y)
    input_dim = size(X, 2)

    seq_len = get(args, "informer_seq_len", 24)
    label_len = get(args, "informer_label_len", 12)
    pred_len = get(args, "informer_pred_len", 1)
    d_model = get(args, "informer_d_model", 64)
    n_heads = get(args, "informer_n_heads", 4)
    e_layers = get(args, "informer_e_layers", 2)
    d_layers = get(args, "informer_d_layers", 1)
    d_ff = get(args, "informer_d_ff", 128)
    dropout = get(args, "informer_dropout", 0.1)
    activation = get(args, "informer_activation", "gelu")
    factor = get(args, "informer_factor", 5)
    distil = get(args, "informer_distil", false)
    epochs = get(args, "informer_epochs", 5)
    batch_size = get(args, "informer_batch_size", 16)
    lr = get(args, "informer_lr", 1e-3)

    label_len = min(label_len, seq_len)

    n = size(X, 1)
    train_start = max(train_start, 1)
    train_end = min(train_end, n - pred_len)
    if train_end < train_start
        return nothing
    end
    if train_end - train_start + 1 < seq_len + pred_len
        return nothing
    end

    X = informer_normalize_features(X, train_start, train_end)

    model = build_informer(input_dim;
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
        factor=factor,
        distil=distil)

    dataset = build_informer_dataset(X, y, seq_len, label_len, pred_len, train_start, train_end)
    if dataset === nothing
        return nothing
    end
    x_enc, x_dec, y_out = dataset

    train_informer!(model, x_enc, x_dec, y_out;
        epochs=epochs, batch_size=batch_size, lr=lr, verbose=false)

    preds = zeros(Float32, val)
    for s in 1:val
        t_idx = split_index + s - 1
        enc_slice, dec_slice = build_informer_predict_inputs(X, seq_len, label_len, pred_len, t_idx)
        enc_in = reshape(enc_slice, input_dim, seq_len, 1)
        dec_in = reshape(dec_slice, input_dim, label_len + pred_len, 1)
        y_pred = model(enc_in, dec_in)
        preds[s] = y_pred[1, 1, 1]
    end
    return preds
end
