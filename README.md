# Ensemble Modeling: An Adaptive Robust Optimization Approach

/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5

Comprises all data:
/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main.jl --data safi --end-id 8 --val 4245 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5

module load julia 1.4
module load gurobi-811
julia src/main.jl --end-id 22 --val 8 --past 3 --num-past 3 --rho 0.1 --train_test_split 0.45 --data M3F1493

Hurricane:


/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main.jl --data hurricane_NA  --end-id 17  --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1

Synthetic:

/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main_synthetic.jl --past 5 --num-past 200 --val 1000 --total_drift_additive --bias_range 0.25 --std_range 0.5 --T 1000 --num_exp 10 --seed 1

/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main_synthetic.jl --past 3 --num-past 73 --val 362 --total_drift_additive --bias_range 0.25 --std_range 0.5

/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main_synthetic_parallel.jl --past 5 --num-past 10 --val 500 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 300 --seed 30 --N_models 15 --bias_drift 0.1 --std_drift 0.1 --CVAR --end-id 20

/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main_synthetic_parallel.jl --past 5 --num-past 600 --train_test_split 0.75 --period 4 --val 1000 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 2000 --seed 400 --N_models 10 --bias_drift 0.5 --std_drift 0.5 --CVAR --rho_beta 0.1 --rho 0.1 --rho_V 0.1
