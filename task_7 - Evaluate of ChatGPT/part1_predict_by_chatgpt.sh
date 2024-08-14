#!/bin/bash

declare -d datasets=("the_naughtyformer")

for dataset in "${datasets[@]}"
do
   echo $dataset
   python part1_predict_by_chatgpt.py $dataset
done