import os
import json

import pandas as pd
import numpy as np

files_path = "./best_models_paths/"

files = os.listdir(files_path)

all_dfs = list()

for f in files:
    all_dfs.append(
        pd.read_csv(os.path.join(files_path, f))
    )

df = pd.concat(all_dfs)

base_path = "/home/ambaranov/You_Told_Me_That_Joke_Twice/task_2/"

df["best_model_path"] = df["best_model_path"].apply(
    lambda x: os.path.join(base_path, x[2:])
)

df["new_key"] = df["dataset"] + '-' + df["rs"].astype(str)
df.index = df["new_key"].tolist()

need_dict = df.to_dict()['best_model_path']

with open("best_model_path.json", "w") as f:
    json.dump(need_dict, f)