#!/bin/bash
#SBATCH --job-name=HR-CD
#SBATCH --error=./logs/HR-CD-%A-%a.err
#SBATCH --output=./logs/HR-CD-%A-%a.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-49

module purge
module load Python

source deactivate
source activate env_humor_roberta

array=(one_liners-23 one_liners-47 one_liners-453 one_liners-693 one_liners-977 pun_of_the_day-23 pun_of_the_day-47 pun_of_the_day-453 pun_of_the_day-693 pun_of_the_day-977 semeval_2017_task_7-23 semeval_2017_task_7-47 semeval_2017_task_7-453 semeval_2017_task_7-693 semeval_2017_task_7-977 short_jokes-23 short_jokes-47 short_jokes-453 short_jokes-693 short_jokes-977 reddit_jokes_last_laught-23 reddit_jokes_last_laught-47 reddit_jokes_last_laught-453 reddit_jokes_last_laught-693 reddit_jokes_last_laught-977 semeval_2021_task_7-23 semeval_2021_task_7-47 semeval_2021_task_7-453 semeval_2021_task_7-693 semeval_2021_task_7-977 funlines_and_human_microedit_paper_2023-23 funlines_and_human_microedit_paper_2023-47 funlines_and_human_microedit_paper_2023-453 funlines_and_human_microedit_paper_2023-693 funlines_and_human_microedit_paper_2023-977 unfun_me-23 unfun_me-47 unfun_me-453 unfun_me-693 unfun_me-977 the_naughtyformer-23 the_naughtyformer-47 the_naughtyformer-453 the_naughtyformer-693 the_naughtyformer-977 comb-23 comb-47 comb-453 comb-693 comb-977)

hd_idx=$SLURM_ARRAY_TASK_ID

export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun python3 -u part3_predict_by_models_cd.py ${array[$hd_idx]}
