#!/bin/bash
#SBATCH --job-name=TASK_4_PART_4
#SBATCH --error=TASK_4_PART_4.err
#SBATCH --output=TASK_4_PART_4.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

module purge
module load Python

source deactivate
source activate env_humor_roberta

srun python3 -u part4_print_metrics.py
