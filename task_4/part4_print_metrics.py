import os
from collections import OrderedDict
import copy

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

from hri_tools import HRI_PAPER_2023_DATASETS
from hri_tools. utils import SUPPORTED_CONVERSATIONAL_DATASETS


######### Print Humor Dataset Metrics

df_colbert = pd.read_csv("./reports_colbert/colbert_pred_hd.csv")
df_colbert["model"] = "Colbert"

ws_results = dict()

for i, d_name in enumerate(HRI_PAPER_2023_DATASETS):

    tmp_texts = open(f"./reports_WS_hd/{d_name}/eval_results.txt").readlines()
    
    row = tmp_texts[1].strip()
    f1_score_f = float(row[row.find("=")+2:])

    row = tmp_texts[4]
    recall_score_f = float(row[row.find("=")+2:])
    
    ws_results[i] = {
        "model" : "WS",
        "f1_score": f1_score_f,
        "recall": recall_score_f,
        "dataset": d_name
    }

df_ws = pd.DataFrame().from_dict(ws_results, orient="index")

all_metrics_mean = pd.concat([df_colbert, df_ws])

order_datasets = ["WS", "Colbert"]

order_datasets_test = ["one_liners", "pun_of_the_day", "semeval_2017_task_7", "short_jokes", "reddit_jokes_last_laught",
                  "semeval_2021_task_7", "funlines_and_human_microedit_paper_2023", "unfun_me", "onion_or_not", "the_naughtyformer", "comb"
                  ]

results = OrderedDict()

for ds in order_datasets:
    tmp_df = all_metrics_mean[all_metrics_mean["model"] == ds]
    results[ds] = OrderedDict()
    for dst in order_datasets_test:
        ttmp_df = tmp_df[tmp_df["dataset"] == dst]
        if dst == "onion_or_not":
            results[ds][dst] = ttmp_df["recall"].iloc[0]
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
    "Colbert": "ColBERT"
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




######### Print Conversational Dataset Metrics


df_colbert = pd.read_csv("./reports_colbert/colbert_pred_cd.csv")
df_colbert["model"] = "Colbert"
df_colbert["ratio"] = df_colbert["ratio"].apply(lambda x: eval(x)[0])


ws_results = dict()

for i, d_name in enumerate(SUPPORTED_CONVERSATIONAL_DATASETS):

    tmp_texts = open(f"./reports_WS_cd/{d_name}/eval_results.txt").readlines()
    
    row = tmp_texts[1].strip()
    f1_score_f = float(row[row.find("=")+2:])
    
    ws_results[i] = {
        "model" : "WS",
        "ratio": f1_score_f,
        "dataset": d_name
    }

df_ws = pd.DataFrame().from_dict(ws_results, orient="index")

all_metrics_mean = pd.concat([df_colbert, df_ws])

order_datasets = ["WS", "Colbert"]

order_datasets_test = ['fig_qa_start', 'fig_qa_end', 'irony', 'alice', 'three_men', 'curiousity', 'friends',  'walking_dead']


results = OrderedDict()

for ds in order_datasets:
    tmp_df = all_metrics_mean[all_metrics_mean["model"] == ds]
    results[ds] = OrderedDict()
    for dst in order_datasets_test:
        ttmp_df = tmp_df[tmp_df["dataset"] == dst]
        results[ds][dst] = ttmp_df["ratio"].iloc[0]



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
    "Colbert": "ColBERT"
}

print("--------------------------------------")

for k in results:
        print(
            f"""{mapping_names[k]}   & {round(results[k]["fig_qa_start"], 2):.2f} & {round(results[k]["fig_qa_end"], 2):.2f} & {round(results[k]["irony"], 2):.2f} & {round(results[k]["alice"], 2):.2f} & {round(results[k]["three_men"], 2):.2f} & {round(results[k]["curiousity"], 2):.2f} & {round(results[k]["friends"], 2):.2f} & {round(results[k]["walking_dead"], 2):.2f}"""
        )

print("--------------------------------------")