# Full Benchmark Run

This folder contains curated outputs from full real-world benchmark runs with the default FEDFormer and Informer configurations enabled.

## Energy command

```powershell
julia --project=. --startup-file=no src/main_hypertune.jl --data energy --end-id 8 --val 2000 --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5 --param_combo 500 --fedformer --informer --fedformer_save_preds --informer_save_preds
```

This matches the README energy setup for `end-id`, `val`, `past`, `num-past`, `rho`, and split, with transformer baselines added. The result id is `param_combo=500` so the output file does not overwrite earlier shorter runs.

## Wind-speed command

```powershell
julia --project=. --startup-file=no src/main_hypertune.jl --data safi_speed --begin-id 1 --end-id 8 --val 1699 --train_test_split 0.5 --num-past 5000 --param_combo 1 --fedformer --informer --fedformer_save_preds --informer_save_preds
```

This matches the README wind-speed setup, with transformer baselines added.

## Transformer configuration

Both runs used the default transformer settings from `src/transformer_args.jl`:

- FEDFormer and Informer sequence length: `24`
- Label length: `12`
- Prediction length: `1`
- Model dimension: `64`
- Attention heads: `4`
- Encoder layers: `2`
- Decoder layers: `1`
- Feed-forward dimension: `128`
- Dropout: `0.1`
- Epochs: `5`
- Batch size: `16`
- Learning rate: `0.001`

FEDFormer-specific defaults:

- Moving average window: `5`
- Frequency mode: `fourier`
- Modes: `16`

Informer-specific defaults:

- ProbSparse factor: `5`
- Distilling: disabled

## Files

- `energy/metrics.csv`: full energy benchmark metrics.
- `energy/fedformer_preds.csv`: standardized FEDFormer predictions for the full energy horizon.
- `energy/informer_preds.csv`: standardized Informer predictions for the full energy horizon.
- `energy/weights/`: method weight and prediction CSVs copied from `results_beta`.
- `safi_speed/metrics.csv`: full wind-speed benchmark metrics.
- `safi_speed/fedformer_preds.csv`: standardized FEDFormer predictions for the full wind-speed horizon.
- `safi_speed/informer_preds.csv`: standardized Informer predictions for the full wind-speed horizon.
- `safi_speed/weights/`: method weight and prediction CSVs copied from `results_beta`.

Plots are stored in `figures/full_benchmark/`.
