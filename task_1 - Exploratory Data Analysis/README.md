# Task 1

This assignment calculates all the basic statistics for the tables in the paper.

This task is divided into several parts.

**Part 1** is statistics on the number of examples in the humor datasets, these are statistics on the use of obscene words, these are statistics on the length of texts, and kl divergence counts.

For run on SLURM server use:
```
sbatch part1_basic_statistics.sbatch
```

Or use python file with configs:
```
python part1_basic_statistics.py
```


**Part 2** is statistics on data intersection between datasets, as well as statistics on duplicates in the data.

For run on SLURM server use:
```
sbatch part2_check_dublicates.sbatch
```

Or use python file with configs:
```
python part2_check_dublicates.py
```

**Part 3** is statistics on the number of examples in the conversational datasets.

For run on SLURM server use:
```
sbatch part3_conversational_statistics.sbatch
```

Or use python file with configs:
```
python part3_conversational_statistics.py
```