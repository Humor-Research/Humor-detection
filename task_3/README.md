# Task 3

This task evaluates the models trained in the task 2 on both humor and conversational datasets.

**Part 1**

This subtask calculates the label predictions for the humor datasets, used in task 2.

For run on SLURM server use:

```
sbatch part1_predict_by_models_hd.sbatch
```

Or use python file with configs:

```
python part1_predict_by_models_hd.py $TRAIN_DATASET-$RANDOM_SEED
```

Here one has to choose the model for evaluation, by explicitly mentioning the training dataset and the random seed which had been used for training in task 2.

**Part 2**

This subtask evaluates the predicted labels from subtask 1 by calculating F1 score and recall.

For run on SLURM server use:

```
sbatch part2_calc_metrics.sbatch
```

Or use python file with configs:

```
python part2_calc_metrics.py
```

**Part 3**

This subtask calculates the label predictions for the conversational datasets.

For run on SLURM server use:

```
sbatch part3_predict_by_models_cd.sbatch
```

Or use python file with configs:

```
python part3_predict_by_models_cd.py $TRAIN_DATASET-$RANDOM_SEED
```

Here one has to choose the model for evaluation, following the same pattern as in subtask 1.

**Part 4**

This subtask evaluates the predicted labels from subtask 3 by calculating the ratio of humorous (positive class) predictions.

For run on SLURM server use:

```
sbatch part4_calc_metrics_cd.sbatch
```

Or use python file with configs:

```
python part4_calc_metrics_cd.py
```
