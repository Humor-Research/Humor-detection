#!/bin/bash

declare -d datasets=("alice" "curiousity" "fig_qa_end" "fig_qa_start" "friends" "irony" "three_men" "walking_dead")

for dataset in "${datasets[@]}"
do
   echo $dataset
   python part2_predict_by_chatgpt_cd.py $dataset
done