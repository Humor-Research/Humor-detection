#!/bin/bash
#SBATCH --job-name=WS_task
#SBATCH --error=WS.err
#SBATCH --output=WS.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-10

module purge
module load CUDA/10.0 Python/Anaconda_v10.2019

source deactivate
source activate my_py_env 

export CUDA_VISIBLE_DEVICES=0

array=(comb funlines_and_human_microedit_paper_2023 one_liners onion_or_not pun_of_the_day reddit_jokes_last_laught semeval_2017_task_7 semeval_2021_task_7 unfun_me short_jokes the_naughtyformer)
hd_idx=$SLURM_ARRAY_TASK_ID

srun python part2_predict_by_ws_hd.py \
--task_name=cola \
--do_eval \
--data_dir=./data \
--max_seq_length=128 \
--train_batch_size=256 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--do_lower_case \
--gradient_accumulation_steps 24 \
--bert_model=bert-base-uncased \
--output_dir=./reports_WS_hd \
--humor_dataset=${array[$hd_idx]}
