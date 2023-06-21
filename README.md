# Ensemble Modeling: An Adaptive Robust Optimization Approach

Accurate time series forecasting is essential for a wide range of problems that involve temporal data. However, the performance of a single predictive model can be highly variable due to shifts in the underlying data distribution. Ensemble modeling is a well-established technique for leveraging multiple models to improve accuracy and robustness. This paper proposes a new methodology for building robust ensembles of time series forecasting models.  Our approach utilizes Adaptive Robust Optimization (ARO) to build a linear regression ensemble in which the models' weights can adapt over time. We demonstrate the effectiveness of our method through a series of synthetic experiments and real-world applications, including air pollution management, energy consumption forecasting, and tropical cyclone intensity forecasting. Our results show that our adaptive ensemble outperforms the best ensemble member in hindsight by 16-26\% in root mean square error and 14-28\% in conditional value at risk.

The following repository hosts the code to perform ensemble modeling in the context of time series forecasting.

The following methods are benchmarked:
XXX complete with paper
- Ensemble Mean

The repository is organized as follows:


- ```/data``` contains all the data for the different use cases
- ```/src``` contains all the core Julia code: 
  - ```/algos``` contains the different ensemble models' algorithms 
  - ```/synthetic_experiment``` contains helpers to perform the synthetic experiments
  - ```eval2.jl``` contains the core function that trains and test the different models (discard eval.jl)
  - ```main.jl``` is the executable file to launch an experiment and specify all hyperparameters
  - ```main_hypertune.jl``` is the executable file to launch an experiment and conduct a hyperparameter search conveniently when a cluster is available
  - ```main_synthetic.jl``` is the executable file to launch all synthetic experiments using a cluster
  - ```metrics.jl``` contains the different metrics to compare the performance, and add the results to a dataframe
  - ```utils.jl``` contains helpers functions
  - ```utils_hurricane.jl``` contains helpers functions for the hurricane forecasting use case

Here are a few examples of jobs to execute (assuming you can use julia from your terminal, otherwise replace with something like ```/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia``` depending on your OS and version):

**Energy dataset**:
```julia src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5```

Wind speed forecasting dataset:
```julia src/main.jl --data safi --end-id 8 --val 4245 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5```

Hurricane dataset:

```julia src/main.jl --data hurricane_NA  --end-id 17  --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1```

Synthetic data experiments:

```julia src/main_synthetic.jl --past 5 --num-past 200 --val 1000 --total_drift_additive --bias_range 0.25 --std_range 0.5 --T 1000 --num_exp 10 --seed 1```

```julia src/main_synthetic.jl --past 3 --num-past 73 --val 362 --total_drift_additive --bias_range 0.25 --std_range 0.5```

```julia src/main_synthetic_parallel.jl --past 5 --num-past 10 --val 500 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 300 --seed 30 --N_models 15 --bias_drift 0.1 --std_drift 0.1 --CVAR --end-id 20```

```julia src/main_synthetic_parallel.jl --past 5 --num-past 600 --train_test_split 0.75 --period 4 --val 1000 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 2000 --seed 400 --N_models 10 --bias_drift 0.5 --std_drift 0.5 --CVAR --rho_beta 0.1 --rho 0.1 --rho_V 0.1```
