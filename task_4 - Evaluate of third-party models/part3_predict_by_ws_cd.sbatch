#!/bin/bash
#SBATCH --job-name=WS_task_cd
#SBATCH --error=WS_cd.err
#SBATCH --output=WS_cd.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-7

module purge
module load CUDA/10.0 Python/Anaconda_v10.2019

source deactivate
source activate my_py_env 

export CUDA_VISIBLE_DEVICES=0

array=(alice curiousity fig_qa_end fig_qa_start friends irony three_men walking_dead)
hd_idx=$SLURM_ARRAY_TASK_ID

srun python part3_predict_by_ws_сd.py \
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
--output_dir=./reports_WS_cd \
--humor_dataset=${array[$hd_idx]}
