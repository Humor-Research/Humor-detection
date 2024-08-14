import os
import sys
import json
import random
random.seed(0)

import pandas as pd
from tqdm import tqdm
import numpy as np
np.random.seed(0)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

import datasets
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import TextClassificationPipeline

from hri_tools import HumorDataset, HRI_PAPER_2023_DATASETS
from hri_tools.utils import SUPPORTED_CONVERSATIONAL_DATASETS
from hri_tools.dataset_loader import ConversationalDataset

TRAIN_DATASET = str(sys.argv[1])

with open("../task_2/best_model_path.json") as f:
    best_model_path = json.load(f)

model_path = best_model_path[TRAIN_DATASET]
tokenizer_path = '/home/ambaranov/roberta-base/'

model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_length = 512, truncation=True)

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0, max_length = 512, truncation=True)


for d_name in SUPPORTED_CONVERSATIONAL_DATASETS:
    hd = ConversationalDataset(d_name)
    df = hd.get_data()
    preds = pipe(df["text"].tolist())

    res_df = pd.DataFrame()
    res_df["preds"] = preds
    res_df["trained_on"] = TRAIN_DATASET
    res_df["test_on"] = d_name

    res_df.to_csv(f"./reports_cd/{TRAIN_DATASET}_pred_{d_name}.csv")