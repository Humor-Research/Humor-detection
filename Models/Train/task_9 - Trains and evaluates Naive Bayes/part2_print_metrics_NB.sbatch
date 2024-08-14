#!/bin/bash
#SBATCH --job-name=TASK_9_PART_2
#SBATCH --error=TASK_9_PART_2.err
#SBATCH --output=TASK_9_PART_2.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12

module purge
module load Python

source deactivate
source activate env_humor_roberta

srun python -u part2_print_metrics_NB.py