import pandas as pd
import numpy as np

import os
import copy

from collections import OrderedDict

from sklearn.metrics import f1_score, recall_score


##### Humor dataset metrics
files = os.listdir("./reports_hd/")

all_df = list()

for file in files:

    all_df.append(
        pd.read_csv(f"./reports_hd/{file}")
    )

df = pd.concat(all_df)


def normalize_label(raw_label):

    if "1" in raw_label:
        return 1
    else:
        return 0
    

def normalize_preds(raw_pred):

    raw_pred = str(raw_pred).lower()

    if "unfunny" in raw_pred:
        return 0
    else:
        return 1
    

df["predicted"] = df["predicted"].apply(normalize_preds)

df["label"] = df["label"].apply(normalize_label)

metrics_dict = dict()
metrics_index = 0

all_datasets = pd.unique(df["dataset"])

for d_name in all_datasets:

    tmp_df = df[df["dataset"] == d_name]

    metrics_dict[metrics_index] = {
        "dataset": d_name,
        "model": "flan",
        "f1_score": f1_score(tmp_df["label"], tmp_df["predicted"]),
        "recall_score": recall_score(tmp_df["label"], tmp_df["predicted"])
    }

    metrics_index += 1

df_metrics = pd.DataFrame().from_dict(metrics_dict, orient="index")

order_datasets = ["flan"]

order_datasets_test = ["one_liners", "pun_of_the_day", "semeval_2017_task_7", "short_jokes", "reddit_jokes_last_laught",
                  "semeval_2021_task_7", "funlines_and_human_microedit_paper_2023", "unfun_me", "onion_or_not", "the_naughtyformer", "comb"
                  ]

results = OrderedDict()

for ds in order_datasets:
    tmp_df = df_metrics[df_metrics["model"] == ds]
    results[ds] = OrderedDict()
    for dst in order_datasets_test:
        ttmp_df = tmp_df[tmp_df["dataset"] == dst]
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
    "onion_or_not": "TheO",
    "WS": "W&S",
    "Colbert": "ColBERT",
    "flan": "FLAN"
}

results_f1_only = copy.deepcopy(results)

for k in results_f1_only:
    results_f1_only[k].pop("onion_or_not")

print("--------------------------------------")

for k in results:
        print(
            f"""{mapping_names[k]}   & {round(results[k]["one_liners"], 2):.2f} & {round(results[k]["pun_of_the_day"], 2):.2f} & {round(results[k]["semeval_2017_task_7"], 2):.2f} & {round(results[k]["short_jokes"], 2):.2f} & {round(results[k]["reddit_jokes_last_laught"], 2):.2f} & {round(results[k]["semeval_2021_task_7"], 2):.2f} & {round(results[k]["funlines_and_human_microedit_paper_2023"], 2):.2f} & {round(results[k]["unfun_me"], 2):.2f} & {round(results[k]["the_naughtyformer"], 2):.2f} & {round(results[k]["comb"], 2):.2f} & {round(np.mean(list(results_f1_only[k].values())), 2):.2f} & {round(results[k]["onion_or_not"], 2):.2f}"""
        )

print("--------------------------------------")




################# Conversational Dataset

##### Humor dataset metrics
files = os.listdir("./reports_cd/")

all_df = list()

for file in files:

    all_df.append(
        pd.read_csv(f"./reports_cd/{file}")
    )

df = pd.concat(all_df)


def normalize_label(raw_label):

    if "1" in raw_label:
        return 1
    else:
        return 0
    

def normalize_preds(raw_pred):

    raw_pred = str(raw_pred).lower()

    if "unfunny" in raw_pred:
        return 0
    else:
        return 1
    

df["predicted"] = df["predicted"].apply(normalize_preds)

metrics_dict = dict()
metrics_index = 0

all_datasets = pd.unique(df["dataset"])

for d_name in all_datasets:

    tmp_df = df[df["dataset"] == d_name]

    metrics_dict[metrics_index] = {
        "dataset": d_name,
        "model": "flan",
        "ratio_score": tmp_df["predicted"].sum() / len(tmp_df),
    }

    metrics_index += 1

df_metrics = pd.DataFrame().from_dict(metrics_dict, orient="index")

order_datasets = ["flan"]

order_datasets_test = ['fig_qa_start', 'fig_qa_end', 'irony', 'alice', 'three_men', 'curiousity', 'friends',  'walking_dead']

for ds in order_datasets:
    tmp_df = df_metrics[df_metrics["model"] == ds]
    results[ds] = OrderedDict()
    for dst in order_datasets_test:
        ttmp_df = tmp_df[tmp_df["dataset"] == dst]
        results[ds][dst] = ttmp_df["ratio_score"].iloc[0]

print("--------------------------------------")

for k in results:
        print(
            f"""{mapping_names[k]}   & {round(results[k]["fig_qa_start"], 2):.2f} & {round(results[k]["fig_qa_end"], 2):.2f} & {round(results[k]["irony"], 2):.2f} & {round(results[k]["alice"], 2):.2f} & {round(results[k]["three_men"], 2):.2f} & {round(results[k]["curiousity"], 2):.2f} & {round(results[k]["friends"], 2):.2f} & {round(results[k]["walking_dead"], 2):.2f}"""
        )

print("--------------------------------------")
