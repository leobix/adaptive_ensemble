# Ensemble Modeling: An Adaptive Robust Optimization Approach

Accurate time series forecasting is essential for a wide range of problems that involve temporal data. However, the performance of a single predictive model can be highly variable due to shifts in the underlying data distribution. Ensemble modeling is a well-established technique for leveraging multiple models to improve accuracy and robustness. This paper proposes a new methodology for building robust ensembles of time series forecasting models.  Our approach utilizes Adaptive Robust Optimization (ARO) to build a linear regression ensemble in which the models' weights can adapt over time. We demonstrate the effectiveness of our method through a series of synthetic experiments and real-world applications, including air pollution management, energy consumption forecasting, and tropical cyclone intensity forecasting. Our results show that our adaptive ensemble outperforms the best ensemble member in hindsight by 16-26\% in root mean square error and 14-28\% in conditional value at risk.

The following repository hosts the code to perform ensemble modeling in the context of time series forecasting.

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

## Methods benchmarked

We benchmarked the following ensembles on the same data:

- **Best Model in Hindsight**: we determine in hindsight what was the best ensemble member on the test data with respect to the MAPE and report its performance for all metrics. Notice that in real-time, it is impossible to know which model would be the best on the overall test set, which means the best model in hindsight is a competitive benchmark.

- **Ensemble Mean**: consists of weighing each model equally, predicting the average of all ensemble members at each time step.

- **Exp3**: under the multi-armed bandit setting, Exp3 weighs the different models to minimize the regret compared to the best model so far. The update rule is given by:

```math
\begin{align*}
\boldsymbol{\beta}_{t+1}^i &= \exp\left(\frac{-\eta_t \cdot Regret_t^i}{\sum_{i=1}^m \exp(-\eta\cdot Regret_t^i)}\right),  \text{ with} \\ \quad Regret_t^i &= \sum_{s=t-t_0}^{t}(y_s-X_s^i)^2, \quad \forall i\in[1,m], \ \text{and} \\
\eta_t &= \sqrt{\frac{8\log(m)}{t_0}},
\end{align*}
```
where the window size $t_0$ considered to determine the regularized leader is tuned.

- **Passive-Aggressive**: A well-known margin-based online learning algorithm that updates the weights of its linear model based on the following equation:

    ```math
    \boldsymbol{\beta}_{t+1}=\boldsymbol{\beta}_{t}+sign\left(y_{t}\mathbf{e}-\mathbf{X}_t^\top\boldsymbol{\beta}_{t}\right) \tau_{t} \mathbf{X}_{t}, \quad \tau_t = \frac{\max(0, |\mathbf{X}_t^\top\boldsymbol{\beta}_t  - y_t|-\epsilon)}{\|\mathbf{X}_t\|_2^2},
    ```
    where $\epsilon$ is a margin parameter to be tuned.

- **Ridge**: Consists in learning the best linear combination of ensemble members by solving a ridge problem on the forecasts $\mathbf{X}_{t}$.



## How to use the code:

Note: currently working on documenting the code further, but it is ready for use.

Here are a few examples of jobs to execute (assuming you can use julia from your terminal, otherwise replace with something like ```/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia``` depending on your OS and version):

**Energy dataset**:
- ```julia src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5```

Wind speed forecasting dataset:
- ```julia src/main.jl --data safi --end-id 8 --val 4245 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5```

Hurricane dataset:

- ```julia src/main.jl --data hurricane_NA  --end-id 17  --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1```

Synthetic data experiments:

- ```julia src/main_synthetic.jl --past 5 --num-past 200 --val 1000 --total_drift_additive --bias_range 0.25 --std_range 0.5 --T 1000 --num_exp 10 --seed 1```

- ```julia src/main_synthetic.jl --past 3 --num-past 73 --val 362 --total_drift_additive --bias_range 0.25 --std_range 0.5```

- ```julia src/main_synthetic_parallel.jl --past 5 --num-past 10 --val 500 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 300 --seed 30 --N_models 15 --bias_drift 0.1 --std_drift 0.1 --CVAR --end-id 20```

- ```julia src/main_synthetic_parallel.jl --past 5 --num-past 600 --train_test_split 0.75 --period 4 --val 1000 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 2000 --seed 400 --N_models 10 --bias_drift 0.5 --std_drift 0.5 --CVAR --rho_beta 0.1 --rho 0.1 --rho_V 0.1```
