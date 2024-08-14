import os
from collections import OrderedDict
import copy

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score


files_path = "./reports_cd/"

files = os.listdir(files_path)
all_dfs = list()
for f in files:
    tmp_df = pd.read_csv(os.path.join(files_path, f))
    pred_position = f.find("_pred_")
    name_with_rs = f[:pred_position]
    trained_name = name_with_rs.split("-")[0]
    trained_rs = name_with_rs.split("-")[1]
    tmp_df["rs"] = trained_rs
    tmp_df["trained_on"] = trained_name
    all_dfs.append(
        tmp_df.copy()
    )

df = pd.concat(all_dfs)


def normalize_predict(p):
    p = eval(p)
    if "LABEL_1" in p["label"]:
        return 1
    else:
        return 0

all_trained = pd.unique(df["trained_on"])
all_tested = pd.unique(df["test_on"])
all_rs = pd.unique(df["rs"])

df["preds"] = df["preds"].apply(normalize_predict)

all_res = dict()
all_res_idx = 0

for tr in tqdm(all_trained):
    for ts in all_tested:
        for rs in all_rs:
            tmp_df = df.query(
                "(trained_on == @tr) & (test_on == @ts) & (rs == @rs)"
            )

            if len(tmp_df) > 0:

                all_res[all_res_idx] = {
                    "trained": tr,
                    "tested": ts,
                    "rs": rs,
                    "ratio": tmp_df["preds"].sum() / len(tmp_df)
                }
                all_res_idx += 1
            
            else:
                print(tr, ts, rs)


all_metrics = pd.DataFrame().from_dict(all_res, orient="index")
all_metrics.to_csv("./metrics/all_metrics_cd.csv")

all_metrics_mean = all_metrics.groupby(["trained", "tested"]).agg({
    "ratio": [np.median, np.std, np.mean]
})

all_metrics_mean = all_metrics_mean.reset_index(drop=False)
all_metrics_mean.columns = ["_".join(pair) for pair in all_metrics_mean.columns]
all_metrics_mean.to_csv("./metrics/all_metrics_mean_cd.csv")


order_datasets = ["one_liners", "pun_of_the_day", "semeval_2017_task_7", "short_jokes", "reddit_jokes_last_laught",
                  "semeval_2021_task_7", "funlines_and_human_microedit_paper_2023", "unfun_me", "the_naughtyformer",
                  "comb"
                  ]



order_datasets_test = ['fig_qa_start', 'fig_qa_end', 'irony', 'alice', 'three_men', 'curiousity', 'friends',  'walking_dead']

eps = [23, 47, 453, 693, 977]

results = OrderedDict()

for ds in order_datasets:
    tmp_df = all_metrics_mean[all_metrics_mean["trained_"] == ds]
    results[ds] = OrderedDict()
    for dst in order_datasets_test:
        ttmp_df = tmp_df[tmp_df["tested_"] == dst]
        results[ds][dst] = ttmp_df["ratio_median"].iloc[0]


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

for k in results:
        print(
            f"""{mapping_names[k]}   & {round(results[k]["fig_qa_start"], 2):.2f} & {round(results[k]["fig_qa_end"], 2):.2f} & {round(results[k]["irony"], 2):.2f} & {round(results[k]["alice"], 2):.2f} & {round(results[k]["three_men"], 2):.2f} & {round(results[k]["curiousity"], 2):.2f} & {round(results[k]["friends"], 2):.2f} & {round(results[k]["walking_dead"], 2):.2f}"""
        )