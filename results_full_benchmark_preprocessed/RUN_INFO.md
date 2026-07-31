# Full Preprocessed Wind-Speed Benchmark

This folder contains curated outputs from a full wind-speed benchmark run after preprocessing the raw wind CSVs.

## Preprocessing

The preprocessing script is:

```powershell
julia --project=. --startup-file=no data\preprocessing\preprocess_wind_speed.jl
```

It writes:

- `data/preprocessing/wind_speed_X_preprocessed.csv`
- `data/preprocessing/wind_speed_y_preprocessed.csv`
- `data/preprocessing/wind_speed_preprocessing_report.md`

The cleaned X file removes the unnamed row-index column from the raw wind forecast file, adds an explicit `intercept` column as column 1, and preserves the real wind forecast columns in their original target units.

The transformer wrappers then normalize their feature inputs column-wise using only the training window before constructing FEDFormer/Informer datasets.

## Full Benchmark Command

```powershell
julia --project=. --startup-file=no src\main_hypertune.jl --data mydata --filename-X data/preprocessing/wind_speed_X_preprocessed.csv --filename-y data/preprocessing/wind_speed_y_preprocessed.csv --begin-id 1 --end-id -1 --val 1699 --train_test_split 0.5 --num-past 5000 --param_combo 498 --fedformer --informer --fedformer_save_preds --informer_save_preds
```

This uses the full validation horizon and default full transformer settings:

- Sequence length: `24`
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

## Files

- `safi_speed/metrics.csv`: full benchmark metrics on the cleaned wind-speed data.
- `safi_speed/fedformer_preds.csv`: standardized FEDFormer predictions.
- `safi_speed/informer_preds.csv`: standardized Informer predictions.
- `safi_speed/weights/`: copied method prediction/weight CSVs from `results_beta`.
- `combined_metrics.csv`: aggregate metrics file for this preprocessed benchmark.

Plots are stored in `figures/full_benchmark_preprocessed/`.
