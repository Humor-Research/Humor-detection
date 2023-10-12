import pandas as pd


df = pd.read_csv("../task_3/metrics/all_metrics_mean.csv")

mapping_names = {
    "one_liners": "16kOL",
    "pun_of_the_day": "PotD",
    "semeval_2017_task_7": "EnPuns",
    "short_jokes": "ShJ",
    "reddit_jokes_last_laught": "ReJ",
    "semeval_2021_task_7": "Haha",
    "funlines_and_human_microedit_paper_2023": "FL+HME",
    "unfun_me": "Uf.me",
    "the_naughtyformer": "NF",
    "comb": "COMB",
    "onion_or_not": "TheO"
}

df["trained_"] = df["trained_"].apply(lambda x: mapping_names[x])
df["tested_"] = df["tested_"].apply(lambda x: mapping_names[x])

df = df.pivot_table(values="f1_median", index="trained_", columns="tested_")
df = df.rename_axis(None, axis=1)
df = df.rename_axis("model", axis=0)

df.to_csv("roberta_metrics.csv")