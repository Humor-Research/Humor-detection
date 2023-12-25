# Task 4

This task evaluates the performance of third-party humor detection models on both humor and conversational datasets. For the former the evaluation metrics include F1 score and recall, and for the latter the fraction of humorous (positive class) labels.

**Part 1**

This subtask uses ColBERT model to calculate the label predictions and evaluation metrics for both humor and conversational datasets.

For run on SLURM server use:

```
sbatch part1_predict_by_colbert.sbatch
```

Or use python file with configs:

```
python part1_predict_by_colbert.py
```

**Part 2**

This subtask uses the BERT-based humor classifier by Weller and Seppi to calculate the label predictions and evaluation metrics for the humor datasets.

For run on SLURM server use:

```
sbatch part2_predict_by_ws_hd.sbatch
```

Or use python file with configs:

```
python part2_predict_by_ws_hd.py
```

**Part 3**

This subtask uses the BERT-based humor classifier by Weller and Seppi to calculate the label predictions and evaluation metrics for conversational datasets.

For run on SLURM server use:

```
sbatch part3_predict_by_ws_cd.sbatch
```

Or use python file with configs:

```
python part3_predict_by_ws_cd.py
```

**Part 4**

This subtask prints out the evaluation metrics calculated in subtask 1, 2 and 3.

For run on SLURM server use:

```
sbatch part4_print_metrics.sbatch
```

Or use python file with configs:

```
python part4_print_metrics.py
```
