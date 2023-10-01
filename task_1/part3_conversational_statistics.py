import re
import unicodedata
import json
from itertools import combinations

import pandas as pd

from hri_tools import HumorDataset, HRI_PAPER_2023_DATASETS
from hri_tools.dataset_loader import ConversationalDataset
from hri_tools.utils import SUPPORTED_CONVERSATIONAL_DATASETS


results = dict()

for name in SUPPORTED_CONVERSATIONAL_DATASETS:
    chd = ConversationalDataset(name)
    results[name] = len(chd.get_data())

pd.DataFrame().from_dict(results, orient="index").to_csv("./reports/task_1_part_3_report.csv")



