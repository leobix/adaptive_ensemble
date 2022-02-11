#!/bin/bash
  
#SBATCH --job-name=M3F
#SBATCH --output=M3F_outputs/out_%a.txt
#SBATCH --error=M3F_errors/err_%a.txt
#SBATCH -p normal
#SBATCH --constraint=xeon-g6
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:04:00
#SBATCH --mem-per-cpu=4G

#SBATCH --array=2340-2639%4

source /etc/profile ; 
module load julia/1.4.2
module load gurobi/gurobi-811

julia src/main.jl --end-id 23 --val 9 --past 3 --num-past 3 --rho 0.1 --train_test_split 0.45 --rho_V 1 --data M3F${SLURM_ARRAY_TASK_ID} 
