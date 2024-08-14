#!/bin/bash
#SBATCH --job-name=Naive_Bayes
#SBATCH --error=Naive_Bayes.err
#SBATCH --output=Naive_Bayes.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12

module purge
module load Python

source deactivate
source activate env_humor_roberta

srun python -u part1_train_and_calc_metrics_NB.py