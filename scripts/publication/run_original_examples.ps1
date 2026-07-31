param(
    [ValidateSet("synthetic", "safi", "energy", "hurricane_na")]
    [string]$Dataset = "synthetic",
    [string]$Julia = "julia"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo

switch ($Dataset) {
    "energy" {
        & $Julia --startup-file=no --project=. src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5
    }
    "safi" {
        & $Julia --startup-file=no --project=. src/main_hypertune.jl --data safi_speed --begin-id 1 --end-id 8 --val 1699 --train_test_split 0.5 --num-past 5000 --param_combo 1
    }
    "hurricane_na" {
        & $Julia --startup-file=no --project=. src/main.jl --data hurricane_NA --end-id 17 --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1
    }
    "synthetic" {
        & $Julia --startup-file=no --project=. src/main_synthetic_parallel.jl --past 5 --num-past 10 --train_test_split 0.75 --period 4 --val 1000 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 2000 --seed 1 --N_models 10 --bias_drift 0.5 --std_drift 0.5 --CVAR --rho_beta 0.001 --rho 0.001 --rho_V 0.001
    }
}
if ($LASTEXITCODE -ne 0) { throw "Legacy Julia command failed with exit code $LASTEXITCODE" }
