# Task 7

This task evaluates the performance of the ChatGPT LLM on both humor and conversational datasets.

**Part 1**

This subtask uses the ChatGPT LLM  to calculate the label predictions for humor datasets.

For run on SLURM server use:

```
sbatch part1_predict_hd_flan.sbatch
```

Or use python file with configs:

```
python part1_predict_hd_flan.py $TEST_DATASET
```

Here, one has to explicitly mention the humor dataset for which labels will be calculated.

**Part 2**

This subtask uses the ChatGPT LLM  to calculate the label predictions for conversational datasets.

For run on SLURM server use:

```
sbatch part2_predict_cd_flan.sbatch
```

Or use python file with configs:
```
python part2_predict_cd_flan.py $TEST_DATASET
```

The possible test datasets are: 

```
fig_qa_start fig_qa_end irony alice three_men curiousity friends walking_dead
```

**Part 3**

This subtask evaluates the labels acquired from subtasks 1 and 2. The evaluation metrics for humor datasets include F1 score and recall, and for the conversational – the fraction of humorous (positive class) labels.

For run on SLURM server use:

```
sbatch part3_calc_and_print_metrics.sbatch
```

Or use python file with configs:

```
python part2_calc_and_print_metrics.py
```
