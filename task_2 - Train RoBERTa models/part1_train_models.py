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


def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding = True, truncation=True, max_length = 512)


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


def convert_humor_dataset_for_hf(data_name):
    hd = HumorDataset(data_name)
    hd.load()
    train_data = hd.get_train()
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


def get_test_texts_and_labels(data_name):
    hd = HumorDataset(data_name)
    hd.load()
    test_data = hd.get_test()
    return test_data["text"].tolist(), test_data["label"].tolist()


model = RobertaForSequenceClassification.from_pretrained('/home/ambaranov/roberta-base/')
tokenizer = RobertaTokenizerFast.from_pretrained('/home/ambaranov/roberta-base/', max_length = 512, truncation=True)


TRAIN_DATASET = str(sys.argv[1]).split('-')[0]
rs = int(str(sys.argv[1]).split('-')[1])

train_data, valid_data, _ = convert_humor_dataset_for_hf(TRAIN_DATASET)


if ("semeval_2017_task_7" in TRAIN_DATASET) or ("pun_of_the_day" in TRAIN_DATASET) or ("unfun_me" in TRAIN_DATASET):
    training_args = TrainingArguments(
        output_dir = f'./models/results_{TRAIN_DATASET}_{rs}',
        num_train_epochs=1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 16,    
        per_device_eval_batch_size= 8,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps = 5,
        disable_tqdm = False, 
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_steps = 8,
        fp16 = True,
        logging_dir=f'./logs/logs_{TRAIN_DATASET}_{rs}',
        dataloader_num_workers = 8,
        run_name = f'roberta-classification_{TRAIN_DATASET}_{rs}',
        seed=rs,
        save_total_limit=2,
        save_strategy="steps",
        save_steps = 5
    )
elif ("short_jokes" in TRAIN_DATASET):
    training_args = TrainingArguments(
        output_dir = f'./models/results_{TRAIN_DATASET}_{rs}',
        num_train_epochs=1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 16,    
        per_device_eval_batch_size= 8,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps = 100,
        disable_tqdm = False, 
        weight_decay=0.01,
        warmup_steps=500,
        learning_rate=5e-5,
        logging_steps = 8,
        fp16 = True,
        logging_dir=f'./logs/logs_{TRAIN_DATASET}_{rs}',
        dataloader_num_workers = 8,
        run_name = f'roberta-classification_{TRAIN_DATASET}_{rs}',
        seed=rs,
        save_total_limit=2,
        save_strategy="steps",
        save_steps = 100
    )
else:
    training_args = TrainingArguments(
        output_dir = f'./models/results_{TRAIN_DATASET}_{rs}',
        num_train_epochs=1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 16,    
        per_device_eval_batch_size= 8,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps = 25,
        disable_tqdm = False, 
        weight_decay=0.01,
        warmup_steps=100,
        learning_rate=5e-5,
        logging_steps = 8,
        fp16 = True,
        logging_dir=f'./logs/logs_{TRAIN_DATASET}_{rs}',
        dataloader_num_workers = 8,
        run_name = f'roberta-classification_{TRAIN_DATASET}_{rs}',
        seed=rs,
        save_total_limit=2,
        save_strategy="steps",
        save_steps = 25
    )

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=valid_data,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer.train()

df_best_model = pd.DataFrame()
df_best_model["dataset"] = [TRAIN_DATASET]
df_best_model["rs"] = [rs]
df_best_model["best_model_path"] = [trainer.state.best_model_checkpoint]

df_best_model.to_csv(f"./best_models_paths/best_model_path_{TRAIN_DATASET}_{rs}.csv")