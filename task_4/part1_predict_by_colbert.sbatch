#!/bin/bash
#SBATCH --job-name=colbert_task
#SBATCH --error=colbert_task.err
#SBATCH --output=colbert_task.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=0

module purge
module load CUDA/11.2 Python/Anaconda_v10.2019

source deactivate
source activate my_py_env 

srun python part1_predict_by_colbert.py