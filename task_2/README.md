# Task 2

This task trains RoBERTa models on the existing humor datasets and creates a file with paths to the trained models.

**Part 1**

This subtask involves training RoBERTa models on the humor datasets.

For run on SLURM server use:

```
sbatch part1_train_models.sbatch 
```

Or use python file with configs:

```
python part1_train_models.py $TRAIN_DATASET-$RANDOM_SEED
```

Here, one has to explicitly mention the training dataset and random seed. The supported dataset names can be found after running the following code in python console:

```
from hri_tools import SUPPORTED_DATASETS

print(SUPPORTED_DATASETS)
```

**Part 2**

This subtask generates a configuration file that stores the paths to all the trained models. 

For run on SLURM server use:

```
sbatch part2_generate_config.sbatch
```

Or use python file with configs:

```
python part2_generate_config.py
```
