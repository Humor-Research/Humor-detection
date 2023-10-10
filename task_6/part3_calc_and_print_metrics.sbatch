#!/bin/bash
#SBATCH --job-name=TASK_6_PART_3
#SBATCH --error=TASK_6_PART_3.err
#SBATCH --output=TASK_6_PART_3.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

module purge
module load Python

source deactivate
source activate env_humor_roberta

srun python3 -u part3_calc_and_print_metrics.py