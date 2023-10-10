import os
import sys
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

DATASETS_FOR_TEST = ["short_jokes", "comb", "reddit_jokes_last_laught", "the_naughtyformer", "one_liners"]

def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding = True, truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def convert_humor_dataset_for_hf(data_name, percent_df, rs):

    hd = HumorDataset(data_name)
    hd.load()
    train_data = hd.get_train()
    train_data = train_data.sample(frac=percent_df/100, random_state=rs)

    train_data = datasets.Dataset.from_pandas(train_data)
    train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    valid_data = hd.get_valid()
    valid_data = datasets.Dataset.from_pandas(valid_data)
    valid_data = valid_data.map(tokenization, batched = True, batch_size = len(valid_data))
    valid_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    test_data = hd.get_test()
    test_data = datasets.Dataset.from_pandas(test_data)
    test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_data, valid_data, test_data


TRAIN_DATASET = "short_jokes"

TRAIN_DATASET_PERCENT = float(sys.argv[1].split('-')[0])

rs = int(sys.argv[1].split('-')[1])

model = RobertaForSequenceClassification.from_pretrained('roberta-base')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)

train_data, valid_data, _ = convert_humor_dataset_for_hf(TRAIN_DATASET, TRAIN_DATASET_PERCENT, rs)

training_args = TrainingArguments(
    output_dir = f'./models/results_{TRAIN_DATASET}_{TRAIN_DATASET_PERCENT}_{rs}',
    num_train_epochs=1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 16,    
    per_device_eval_batch_size= 8,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm = False, 
    warmup_steps=0.1,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps = 8,
    fp16 = True,
    logging_dir=f'./models/logs_{TRAIN_DATASET}_{TRAIN_DATASET_PERCENT}_{rs}',
    dataloader_num_workers = 8,
    run_name = f'roberta-classification_{TRAIN_DATASET}_{TRAIN_DATASET_PERCENT}_{rs}',
    seed=rs
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=valid_data
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer.train()

trainer.evaluate()

results = list()

for hd_dataset in DATASETS_FOR_TEST:
    _, _, test_dataset = convert_humor_dataset_for_hf(hd_dataset, 100, rs)
    preds = trainer.predict(test_dataset)
    results.append({
        "trained_on": TRAIN_DATASET,
        "train_percent": TRAIN_DATASET_PERCENT,
        "random_state": rs,
        "test_on": hd_dataset,
        "f1": preds[2]["test_f1"]
    })

results_dict = dict()

for i in range(len(results)):
    results_dict[str(i)] = results[i]

res_df = pd.DataFrame().from_dict(results_dict, orient="index")

res_df.to_csv(f"./results/result_mart_{TRAIN_DATASET}_{TRAIN_DATASET_PERCENT}_{rs}.csv")