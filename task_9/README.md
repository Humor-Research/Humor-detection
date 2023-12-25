# Task 9

This task trains and evaluates Naive Bayes models on the existing humor datasets. The evaluation metrics F1 score, recall, precision and accuracy.

**Part 1**

This subtask trains Naive Bayes models on humor datasets and calculates the label predictions and evaluation metrics for them.

For run on SLURM server use:

```
sbatch part1_train_and_calc_metrics_NB.sbatch
```

Or use python file with configs:

```
python part1_train_and_calc_metrics_NB.py
```

**Part 2**

This subtask prints out the evaluation metrics calculated in subtask 1.

For run on SLURM server use:

```
sbatch part2_print_metrics_NB.sbatch
```

Or use python file with configs:

```
python part2_print_metrics_NB.py
```
