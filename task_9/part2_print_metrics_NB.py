from collections import OrderedDict
import copy

import pandas as pd
import numpy as np

all_metrics_mean = pd.read_csv("./reports/result_NB.csv")

order_datasets = ["one_liners", "pun_of_the_day", "semeval_2017_task_7", "short_jokes", "reddit_jokes_last_laught",
                  "semeval_2021_task_7", "funlines_and_human_microedit_paper_2023", "unfun_me", "the_naughtyformer",
                  "comb"
                  ]



order_datasets_test = ["one_liners", "pun_of_the_day", "semeval_2017_task_7", "short_jokes", "reddit_jokes_last_laught",
                  "semeval_2021_task_7", "funlines_and_human_microedit_paper_2023", "unfun_me", "onion_or_not", "the_naughtyformer", "comb"
                  ]

results = OrderedDict()

for ds in order_datasets:
    tmp_df = all_metrics_mean[all_metrics_mean["train_data"] == ds]
    results[ds] = OrderedDict()
    for dst in order_datasets_test:
        ttmp_df = tmp_df[tmp_df["test_data"] == dst]
        if dst == "onion_or_not":
            results[ds][dst] = ttmp_df["recall_score"].iloc[0]
        else:
            results[ds][dst] = ttmp_df["f1_score"].iloc[0]


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

results_f1_only = copy.deepcopy(results)

for k in results_f1_only:
    results_f1_only[k].pop("onion_or_not")


for k in results:
        print(
            f"""{mapping_names[k]}   & {round(results[k]["one_liners"], 2):.2f} & {round(results[k]["pun_of_the_day"], 2):.2f} & {round(results[k]["semeval_2017_task_7"], 2):.2f} & {round(results[k]["short_jokes"], 2):.2f} & {round(results[k]["reddit_jokes_last_laught"], 2):.2f} & {round(results[k]["semeval_2021_task_7"], 2):.2f} & {round(results[k]["funlines_and_human_microedit_paper_2023"], 2):.2f} & {round(results[k]["unfun_me"], 2):.2f} & {round(results[k]["the_naughtyformer"], 2):.2f} & {round(results[k]["comb"], 2):.2f} & {round(np.mean(list(results_f1_only[k].values())), 2):.2f} & {round(results[k]["onion_or_not"], 2):.2f}"""
        )