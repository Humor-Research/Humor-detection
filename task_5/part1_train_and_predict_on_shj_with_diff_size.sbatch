#!/bin/bash
#SBATCH --job-name=humor_roberta_partitial_base
#SBATCH --error=humor_roberta_partitial_base.err
#SBATCH --output=humor_roberta_partitial_base.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --array=0-49

module purge
module load Python

source deactivate
source activate env_humor_roberta

array=(0.25-23 0.25-47 0.25-453 0.25-693 0.25-977 0.5-23 0.5-47 0.5-453 0.5-693 0.5-977 1-23 1-47 1-453 1-693 1-977 3-23 3-47 3-453 3-693 3-977 5-23 5-47 5-453 5-693 5-977 10-23 10-47 10-453 10-693 10-977 15-23 15-47 15-453 15-693 15-977 30-23 30-47 30-453 30-693 30-977 50-23 50-47 50-453 50-693 50-977 100-23 100-47 100-453 100-693 100-977)

hd_idx=$SLURM_ARRAY_TASK_ID

export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun python3 -u part1_train_and_predict_on_shj_with_diff_size.py ${array[$hd_idx]}
