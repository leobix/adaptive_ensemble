function _source_audit_table()
    rows = [
        ("critical", "src/utils.jl", "12-19",
         "prepare_data_from_y overwrites the first selected forecast column with ones in both train and adaptive blocks.",
         "One actual ensemble member is silently discarded while the reported member count is unchanged.",
         "The publication loader removes only explicit CSV index columns and retains every forecast column."),
        ("critical", "src/algos/adaptive_linear_decision_rule.jl", "12-33",
         "The legacy history has m*tau+tau columns and fills the final tau columns with ones; the manuscript defines exactly m*tau lagged member errors.",
         "The implemented decision rule has undocumented features and a different parameter dimension.",
         "build_error_histories constructs exactly m*tau causal error features and exports history counts."),
        ("high", "src/algos/adaptive_linear_decision_rule.jl", "24-28",
         "The normalized-history expression is evaluated but not assigned.",
         "The err_rule_norm option has no effect.",
         "Normalization is implemented in-place and tested."),
        ("critical", "src/algos/adaptive_linear_decision_rule.jl", "75-104",
         "The principal historical fit minimizes squared loss plus separate squared penalties on beta_t, beta0, and V0.",
         "This is neither the paper's nonsquared norm-plus-norm robust counterpart nor the two-hyperparameter method described in the manuscript.",
         "The new code labels and separates quadratic_stacked and exact norm_plus_norm objectives; both are reported."),
        ("high", "src/main_hypertune.jl", "174-199",
         "The historical search is 5x5x5x4=500 combinations of three penalties and a window.",
         "The paper understates both model-selection degrees of freedom and total search cost.",
         "Every new method uses an explicit, exported chronological validation grid; full-search runtime is benchmarked."),
        ("high", "src/algos/multi_armed_bandits.jl", "1-40",
         "The method called Exp3 observes every member loss and produces a deterministic weighted forecast.",
         "It is a full-information Hedge-style forecaster, not a bandit-feedback Exp3 experiment.",
         "The baseline is named Hedge and a separate Fixed Share tracker is included."),
        ("high", "src/algos/multi_armed_bandits.jl", "initialization",
         "The historical expert weights are initialized using the number of observations rather than the number of members.",
         "Weights need not sum to one and the baseline can be numerically distorted.",
         "All expert trackers initialize to 1/m and use stable log/softmax updates."),
        ("medium", "src/eval2.jl", "best-model evaluation",
         "Best Model in Hindsight is selected on the test segment and presented beside deployable methods.",
         "It is an oracle diagnostic rather than a usable forecasting policy.",
         "The new outputs label oracle_best_member as diagnostic_only and add validation_best_member as a deployable comparator."),
        ("high", "src/metrics.jl", "MAPE implementation",
         "MAPE divides directly by y on a synthetic target that crosses zero.",
         "Synthetic MAPE is undefined or dominated by values arbitrarily close to zero.",
         "Primary synthetic metrics are MAE, RMSE, exact CVaR, sMAPE, and WAPE; safe MAPE reports its valid fraction."),
        ("medium", "src/metrics.jl", "CVaR implementation",
         "Empirical CVaR is formulated as a Gurobi model although the finite-sample value is an exact upper-tail average.",
         "This adds avoidable solver cost and leaves the fractional boundary convention implicit.",
         "empirical_cvar implements the exact finite-sample tail mass, including fractional boundary weight."),
        ("high", "src/main_hypertune.jl", "212-215",
         "energy_y_test_val.csv is headerless but is read with header inference.",
         "The first target can be interpreted as a column name and lost, shifting forecast/target alignment.",
         "load_energy reads the target with header=false and verifies exact 4929/1973/2956 split counts."),
        ("high", "data/energy_predictions_test_val*.csv", "column identities",
         "The stored energy forecast columns do not exactly match the ten member identities listed in the manuscript.",
         "The source of the reported energy table is ambiguous.",
         "Both stored member labels and manuscript labels are exported in the dataset manifest with an explicit warning."),
        ("high", "src/algos/passive_agressive.jl", "entire file",
         "The historical Passive-Aggressive update has fragile dimensions and insufficient zero-denominator protection.",
         "Very poor PA results can be an implementation or tuning artifact.",
         "The new PA-I predicts before updates, guards norms, and tunes C and epsilon on the same validation segment."),
        ("medium", "src/algos/rls.jl", "entire file",
         "The historical RLS covariance update lacks symmetry and numerical safeguards and has no common tuning protocol.",
         "Instability or unfair defaults can affect the dynamic-baseline comparison.",
         "The new RLS symmetrizes P, protects denominators, and tunes forgetting and ridge values chronologically."),
        ("high", "src/main_synthetic*.jl", "driver behavior",
         "Historical synthetic defaults differ from the manuscript's 4000-point design and some driver failures are caught without failing the run.",
         "A run can finish without producing the intended experiment or the paper's data-generating process.",
         "The new generator is manuscript-faithful by default; task failures are collected and make the stage fail."),
        ("high", "data/", "hurricane matrices",
         "The North Atlantic and Eastern Pacific ATCF/Hurricast matrices referenced by the Julia scripts are absent.",
         "The hurricane table cannot be independently regenerated from the supplied archive.",
         "The new loader fails explicitly and documents the required filenames/schema rather than substituting data."),
        ("high", "deployment protocol", "initial and event-boundary rows",
         "The handling of insufficient error history and new-storm boundaries is not fully specified in the legacy code.",
         "Cold-start behavior can leak prior-event outcomes or vary across implementations.",
         "The new causal history zero-pads unavailable slots (beta0 fallback), enforces verification delay, and supports reset/carry stress tests."),
    ]
    return DataFrame(
        severity=[row[1] for row in rows],
        file=[row[2] for row in rows],
        lines=[row[3] for row in rows],
        finding=[row[4] for row in rows],
        consequence=[row[5] for row in rows],
        remediation=[row[6] for row in rows],
    )
end

function _data_inventory(root::AbstractString)
    files = [
        "X_test_speed_adaptive_out1.csv", "X_test_adaptive.csv",
        "y_test_speed_adaptive_out1.csv", "energy_predictions_test_val2.csv",
        "energy_predictions_test_val.csv", "energy_y_test_val.csv",
        "NA_ARO_Intensity_2014_clean_v2.csv", "EP_ARO_Intensity_2014_clean_v2.csv",
    ]
    rows = DataFrame()
    for filename in files
        path = joinpath(root, "data", filename)
        if isfile(path)
            table = _numeric_dataframe(path; header=filename != "energy_y_test_val.csv")
            append!(rows, DataFrame(
                file=[filename], present=[true], rows=[nrow(table)], columns=[ncol(table)],
                bytes=[filesize(path)], sha256=[sha256_file(path)],
            ); cols=:union)
        else
            append!(rows, DataFrame(
                file=[filename], present=[false], rows=[missing], columns=[missing],
                bytes=[missing], sha256=[missing],
            ); cols=:union)
        end
    end
    return rows
end

function _loader_semantics_audit(root::AbstractString)
    output = DataFrame()
    for (name, bundle) in (("safi", load_safi(root)), ("energy", load_energy(root)))
        X = bundle.X
        y = bundle.y
        m = size(X, 2)
        test = collect(bundle.split.test)
        clean_rmse = rmse(y[test], ensemble_mean_predictions(X, test))
        overwritten = copy(X)
        overwritten[:, 1] .= 1.0
        overwritten_rmse = rmse(y[test], ensemble_mean_predictions(overwritten, test))
        append!(output, DataFrame(
            dataset=[name], observations=[length(y)], forecast_columns=[m],
            effective_forecast_columns_after_legacy_overwrite=[m - 1],
            clean_mean_rmse=[clean_rmse],
            legacy_overwrite_mean_rmse=[overwritten_rmse],
            rmse_change=[overwritten_rmse - clean_rmse],
        ); cols=:union)
    end
    return output
end

function _legacy_command_table()
    return DataFrame(
        experiment=["energy", "safi", "hurricane_na", "synthetic"],
        command=[
            "julia --project=. src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5",
            "julia --project=. src/main_hypertune.jl --data safi_speed --begin-id 1 --end-id 8 --val 1699 --train_test_split 0.5 --num-past 5000 --param_combo 1",
            "julia --project=. src/main.jl --data hurricane_NA --end-id 17 --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1",
            "julia --project=. src/main_synthetic_parallel.jl --past 5 --num-past 10 --train_test_split 0.75 --period 4 --val 1000 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 2000 --seed 1 --N_models 10 --bias_drift 0.5 --std_drift 0.5 --CVAR --rho_beta 0.001 --rho 0.001 --rho_V 0.001",
        ],
        status_note=[
            "Runnable with the original supplied energy matrices and a working Gurobi license; preserves legacy semantics.",
            "Runnable with the supplied Safi matrices and a working Gurobi license; preserves legacy semantics.",
            "Not runnable from this archive because the historical hurricane matrix is absent.",
            "Runnable but uses historical driver defaults/semantics rather than the corrected publication suite.",
        ],
    )
end

function run_legacy_audit_publication(config::AbstractDict, paths::RunPaths;
                                      root::AbstractString=".", resume::Bool=true)
    if resume && stage_complete(paths, "legacy_audit")
        return DataFrame(CSV.File(joinpath(paths.tables, "julia_source_audit.csv")))
    end
    audit = _source_audit_table()
    inventory = _data_inventory(root)
    semantics = _loader_semantics_audit(root)
    commands = _legacy_command_table()
    output_paths = [
        joinpath(paths.tables, "julia_source_audit.csv"),
        joinpath(paths.tables, "data_inventory.csv"),
        joinpath(paths.tables, "legacy_loader_semantics.csv"),
        joinpath(paths.tables, "legacy_commands.csv"),
    ]
    atomic_csv_write(output_paths[1], audit)
    atomic_csv_write(output_paths[2], inventory)
    atomic_csv_write(output_paths[3], semantics)
    atomic_csv_write(output_paths[4], commands)
    update_progress(paths, "legacy_audit"; completed=1, total=1,
        message="source/data audit complete")
    mark_stage_complete(paths, "legacy_audit"; extras=Dict(
        "findings" => nrow(audit)), outputs=output_paths)
    return audit
end

function run_original_legacy_command(dataset::AbstractString)
    normalized = lowercase(string(dataset))
    table = _legacy_command_table()
    key = normalized in ("hurricane", "hurricane_na") ? "hurricane_na" : normalized
    match = table[table.experiment .== key, :]
    nrow(match) == 1 || throw(ArgumentError(
        "legacy dataset must be synthetic, safi, energy, or hurricane_na"))
    return string(match.command[1])
end
