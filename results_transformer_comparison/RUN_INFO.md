# Transformer Comparison Run

This folder contains curated outputs from bounded real-world comparison runs with both transformer baselines enabled.

## Shared evaluation settings

- Test horizon: `--val 24`
- Split: `--train_test_split 0.5`
- Adaptive ridge window: `--past 3 --num-past 20`
- Regularization: `--rho_beta 0.01 --rho 0.01 --rho_V 0.1`
- Result identifier: `--param_combo 426`

## Transformer settings

Both FEDFormer and Informer were run with matched common dimensions:

- Sequence length: `8`
- Label length: `4`
- Prediction length: `1`
- Model dimension: `16`
- Attention heads: `2`
- Encoder layers: `1`
- Decoder layers: `1`
- Feed-forward dimension: `32`
- Epochs: `1`
- Batch size: `8`

FEDFormer-specific setting:

- Frequency modes: `4`

Informer-specific setting:

- ProbSparse factor: `3`

## Files

- `energy/metrics.csv`: method-level metrics for the energy run.
- `energy/fedformer_preds.csv`: standardized FEDFormer predictions.
- `energy/informer_preds.csv`: standardized Informer predictions.
- `safi_speed/metrics.csv`: method-level metrics for the wind-speed run.
- `safi_speed/fedformer_preds.csv`: standardized FEDFormer predictions.
- `safi_speed/informer_preds.csv`: standardized Informer predictions.

Plots are stored in `figures/transformer_comparison/`.
