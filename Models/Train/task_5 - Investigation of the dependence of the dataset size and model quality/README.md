# Task 5

This task trains RoBERTa models on subsets of the Short Jokes dataset and evaluates how this limitation influences the result.

**Part 1**

This subtask trains a RoBERTa model on a subset from the Short Jokes dataset and evaluates it on several humor datasets.

For run on SLURM server use:

```
sbatch part1_train_and_predict_on_shj_with_diff_size.sbatch
```

Or use python file with configs:

```
python part1_train_and_predict_on_shj_with_diff_size.py $PERCENT-$RANDOM_SEED
```

Here, one has to explicitly mention the percent of the original dataset to use for training and the random seed.

**Part 2**

This subtask creates a plot showing the relation between the size of the subset used for training and the resulting F1 score in evaluation. It has two variations which are alternative ways to plot the legend, in a simpler or more sophisticated way.

For run on SLURM server use one of the following:

```
sbatch part2_create_image_and_statistics.sbatch
sbatch part2_create_image_and_statistics_other_variant.sbatch
```

Or use one of the python files:

```
python part2_create_image_and_statistics.py
python part2_create_image_and_statistics_other_variant.py
```
