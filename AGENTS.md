# Repository Guidelines

## Project Structure & Module Organization
- `src/` (Julia): core implementation
  - `algos/`: ensemble algorithms (e.g., `adaptive_linear_decision_rule.jl`, `OLS.jl`)
  - `eval2.jl`: main training/evaluation logic (preferred over `eval.jl`)
  - `main*.jl`: entrypoints for use cases and synthetic experiments
  - `metrics.jl`, `utils*.jl`: helpers and metrics
- `python_code/`: Pyomo-based baselines and utilities (`main.py`, `adaptive_lad.py`, etc.)
- `data/`: datasets and CSV inputs; keep large files out of PRs
- `notebooks/`, `data_preparation/`, `results_analysis/`: analysis and prep notebooks
- `run_M3F.sh`: helper script for M3F datasets

## Build, Test, and Development Commands
- Julia deps (one-time):
  - `julia -e 'using Pkg; Pkg.add.( ["JuMP","Gurobi","ArgParse","CSV","DataFrames","Convex"] )'`
- Run examples:
  - Energy: `julia src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5`
  - Hypertune: `julia src/main_hypertune.jl --data mydata --filename-X data/X_toy_test.csv --filename-y data/y_toy_test.csv --param_combo 1 --hypertune`
- Python deps (one-time): `pip install pyomo numpy pandas scikit-learn gurobipy`
- Python example: `python python_code/main.py --method lad_lasso --path_X data/X_test_adaptive.csv --path_y data/y_test.csv`

## Coding Style & Naming Conventions
- Julia: functions and files snake_case; types/modules CamelCase. Match existing patterns in `src/algos/` (keep acronyms like `OLS.jl`).
- Python: snake_case for files/functions; 4-space indentation.
- Prefer small, focused functions with short docstrings. Avoid hardcoded absolute paths; use `data/` relative paths.

## Testing Guidelines
- No formal unit test suite yet. Validate changes by running representative CLI jobs above.
- Include a small CSV in `data/` (or synthetic) and verify printed metrics (MAE, MAPE, RMSE, R2/CVaR when enabled).
- For notebooks, run top-to-bottom and clear outputs before committing.

## Commit & Pull Request Guidelines
- Commits: short imperative summaries (history uses messages like “Update README.md”); prefer descriptive scopes: `feat:`, `fix:`, `docs:` when helpful.
- PRs must include:
  - What/why, linked issues, and impact on results
  - Exact commands used (e.g., the `julia ...` or `python ...` line)
  - Dataset/source paths touched and sample output metrics

## Security & Configuration Tips
- Gurobi is required (Julia and Pyomo). Ensure a local license; do not commit license files or keys.
- Do not commit large raw data or secrets. Prefer `.gitignore`d local paths.
- Keep results reproducible by specifying seeds/params in your PR description.

