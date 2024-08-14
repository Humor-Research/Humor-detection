import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "./results/"

files = os.listdir(data_path)
files = sorted(files)
res_df = list()

for f in files:
    tmp_df = pd.read_csv(os.path.join(data_path, f))
    res_df.append(tmp_df.copy())

df = pd.concat(res_df)
df_stat = df.groupby(["test_on", "train_percent"]).agg(
    {
        "f1":[np.median, np.std]
    }
).reset_index(drop=False)

df_stat.to_csv("./metrics_and_reports/metrics.csv")

tests_datasets = pd.unique(df_stat["test_on"])

fig = plt.figure(figsize=(10,6))
ax = fig.subplots(nrows=1, ncols=1)

markers_dict = {
    "short_jokes": "-o",
    "comb": "-v",
    "reddit_jokes_last_laught":"-s",
    "the_naughtyformer":"-p",
    "one_liners":"-P"
}

short_names = {
    "short_jokes": "ShJ",
    "comb": "COMB",
    "reddit_jokes_last_laught":"ReJ",
    "the_naughtyformer": "NF",
    "one_liners": "16kOL"

}

for d_name in tests_datasets:
    tmp_df = df_stat[df_stat["test_on"] == f"{d_name}"]
    ax.plot(np.log(tmp_df["train_percent"]), tmp_df["f1"]["median"], markers_dict[d_name], label=f"{short_names[d_name]}")
    
ax.set_xticks(np.log(tmp_df["train_percent"]))
ax.set_xticklabels(["0.25%", "0.5%", "1%", "3%", "5%", "10%", "15%", "30%", "50%", "100%"])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax.set_xlabel("Percentage of Short Jokes dataset used for training", fontsize="16")
ax.set_ylabel("F1-score", fontsize="16")

ax.legend(loc='lower right', fontsize="16")
plt.savefig("./metrics_and_reports/shj_diff_size.png")