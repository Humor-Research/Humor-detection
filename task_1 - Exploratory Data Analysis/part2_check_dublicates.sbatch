#!/bin/bash
#SBATCH --job-name=TASK_1_PART_2
#SBATCH --error=TASK_1_PART_2.err
#SBATCH --output=TASK_1_PART_2.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

module purge
module load Python

source deactivate
source activate env_humor_roberta

srun python3 -u part2_check_dublicates.py
