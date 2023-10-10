#!/bin/bash
#SBATCH --job-name=TASK_6_PART_1
#SBATCH --error=TASK_6_PART_1.err
#SBATCH --output=TASK_6_PART_1.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="type_e"
#SBATCH --array=0

module purge
module load Python

source deactivate
source activate env_humor_roberta

array=(onion_or_not)

hd_idx=$SLURM_ARRAY_TASK_ID

srun python3 -u part1_predict_hd_flan.py ${array[$hd_idx]}