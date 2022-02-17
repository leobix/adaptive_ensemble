# Ensemble Modeling: An Adaptive Robust Optimization Approach

/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5

Comprises all data:
/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main.jl --data safi --end-id 8 --val 4245 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5

module load julia 1.4
module load gurobi-811
julia src/main.jl --end-id 22 --val 8 --past 3 --num-past 3 --rho 0.1 --train_test_split 0.45 --data M3F1493



/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia src/main.jl --data hurricane_NA  --end-id 17  --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1
