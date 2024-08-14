#!/bin/bash
#SBATCH --job-name=TASK_5_PART_2
#SBATCH --error=TASK_5_PART_2.err
#SBATCH --output=TASK_5_PART_2.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

module purge
module load Python

source deactivate
source activate env_humor_roberta

srun python3 -u part2_create_image_and_statistics_other_variant.py