# Wind-Speed Preprocessing Report

Input X: `data\X_test_speed_adaptive_out1.csv`
Input y: `data\y_test_speed_adaptive_out1.csv`
Output X: `data\preprocessing\wind_speed_X_preprocessed.csv`
Output y: `data\preprocessing\wind_speed_y_preprocessed.csv`

## Actions

- Removed leading index-like X columns: `Column1`
- Removed leading index-like y columns: `Column1`
- Added first X column `intercept` with value `1.0`.
- Preserved all real wind forecast columns in their original target units.
- Preserved target column `speed` in its original target units.

## Shape Check

- Raw X shape: `8494 x 7`
- Clean X shape: `8494 x 7`
- Raw y shape: `8494 x 2`
- Clean y shape: `8494 x 1`

## Target-Scaled Feature Ranges

The main pipeline target-standardizes `X` using the training target mean/std. After this cleanup, the former row index no longer appears as a huge feature.

- Training split index: `4247`
- Training target mean: `3.81149596`
- Training target std: `1.75552624`

| feature | min | max | mean | std |
|---|---:|---:|---:|---:|
| intercept | -1.6015 | -1.6015 | -1.6015 | 0.0000 |
| NUMTECH_speed | -2.1550 | 5.2081 | 0.0019 | 1.0419 |
| speed | -2.2345 | 3.4138 | -0.0294 | 0.8948 |
| DT_speed | -2.0102 | 3.2962 | -0.0440 | 0.8936 |
| RIDGE_speed | -2.5119 | 3.1859 | -0.0403 | 0.8999 |
| LASSO_speed | -2.0850 | 3.1455 | -0.0464 | 0.8482 |
| OCT_speed | -2.0241 | 4.5790 | -0.0446 | 0.8982 |
