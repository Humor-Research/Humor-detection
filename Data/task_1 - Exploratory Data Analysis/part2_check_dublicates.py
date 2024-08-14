import re
import unicodedata
import json
from itertools import combinations

import pandas as pd

from hri_tools import HumorDataset, HRI_PAPER_2023_DATASETS

original_split_dataset = ["semeval_2021_task_7", "funlines_and_human_microedit_paper_2023", "short_jokes", "comb"]

result_df = list()

result_df_original_split = list()

for df_name in HRI_PAPER_2023_DATASETS:
    if df_name in original_split_dataset:
        df1 = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/train.csv')
        df2 = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/valid.csv')
        df3 = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/test.csv')
        df1["df_name"] = df_name
        df2["df_name"] = df_name
        df3["df_name"] = df_name
        result_df.append(df1)
        result_df.append(df2)
        result_df.append(df3)

        df1 = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/train.csv')
        df2 = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/valid.csv')
        df3 = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/test.csv')

        df1["df_name"] = f"{df_name}_train"
        df2["df_name"] = f"{df_name}_valid"
        df3["df_name"] = f"{df_name}_test"
        result_df_original_split.append(df1)
        result_df_original_split.append(df2)
        result_df_original_split.append(df3)
        
    else:
        df = pd.read_csv(f'~/hri_tools_data/datasets/{df_name}/files/data.csv')
        df["df_name"] = df_name
        result_df.append(df)

result_df = pd.concat(result_df)

print(len(result_df))

result_df["text_pr"] = result_df["text"].apply(lambda row: unicodedata.normalize('NFKD', row).encode('ascii', 'ignore').decode())
result_df["text_pr"] = result_df["text_pr"].apply(lambda row: str(row).lower())
result_df["text_pr"] = result_df["text_pr"].apply(lambda row: re.sub(pattern='[^\w ]', repl='', string=row))

dft = pd.DataFrame(result_df.groupby('text_pr').agg({"text":list, "df_name":list, "label":list}))
dft["df_name_len"] = dft["df_name"].apply(lambda n: len(set(n)))

all_results = dict()
all_pairs = list(combinations(HRI_PAPER_2023_DATASETS, 2))
for pair in all_pairs:
    sample_dft = dft.copy()
    sample_dft["contain_intersection"] = sample_dft["df_name"].apply(lambda x: True if len(set(pair).intersection(set(x))) == 2 else False)
    all_results[str(pair)] = sample_dft["contain_intersection"].sum()


pd.DataFrame().from_dict(all_results, orient="index").sort_values(0, ascending=False).to_csv("./reports/task_1_part_2_report_dups_cross_datasets.csv")


specific_results = dict()
for name in HRI_PAPER_2023_DATASETS:
    sample_dft = dft.copy()
    sample_dft["count_name"] = sample_dft["df_name"].apply(lambda x: list(x).count(name))
    sample_dft["count_name_flag"] = sample_dft["count_name"].apply(lambda x: True if x > 1 else False)
    specific_results[name] = sample_dft["count_name_flag"].sum()

pd.DataFrame().from_dict(specific_results, orient="index").sort_values(0, ascending=False).to_csv("./reports/task_1_part_2_report_dups_in_dataset.csv")


# Check dubs between diff parts

result_df = pd.concat(result_df_original_split)

print(len(result_df))

result_df["text_pr"] = result_df["text"].apply(lambda row: unicodedata.normalize('NFKD', row).encode('ascii', 'ignore').decode())
result_df["text_pr"] = result_df["text_pr"].apply(lambda row: str(row).lower())
result_df["text_pr"] = result_df["text_pr"].apply(lambda row: re.sub(pattern='[^\w ]', repl='', string=row))

dft = pd.DataFrame(result_df.groupby('text_pr').agg({"text":list, "df_name":list, "label":list}))
dft["df_name_len"] = dft["df_name"].apply(lambda n: len(set(n)))

all_results = dict()
all_pairs = list(combinations(result_df["df_name"].unique(), 2))
for pair in all_pairs:
    sample_dft = dft.copy()
    sample_dft["contain_intersection"] = sample_dft["df_name"].apply(lambda x: True if len(set(pair).intersection(set(x))) == 2 else False)
    all_results[str(pair)] = sample_dft["contain_intersection"].sum()


pd.DataFrame().from_dict(all_results, orient="index").sort_values(0, ascending=False).to_csv("./reports/task_1_part_2_report_dups_cross_parts_of_datasets.csv")
