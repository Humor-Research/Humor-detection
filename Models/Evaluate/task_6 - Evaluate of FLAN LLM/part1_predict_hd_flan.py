import sys

import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import DataLoader

from hri_tools import HumorDataset, HRI_PAPER_2023_DATASETS

def predict_by_model(promt_text, model, tokenizer):
    inputs = tokenizer(promt_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=200)
    return tokenizer.decode(outputs[0])

def generate_promt_text(text):
    promt = f"""
    Classify the text into funny or unfunny. 
    Text: {text}
    Label: 
    """
    return promt

def run_llm_humor_test(name_dataset_test):
    model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", torch_dtype=torch.bfloat16, device_map="auto")                                                                 
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    results = dict()
    result_index = 0
    
    print(f"curr {name_dataset_test}")
    
    dataset_test = HumorDataset(name_dataset_test)
    dataset_test.load()

    dataset_test = dataset_test.get_test()

    loader = DataLoader(
        list(zip(dataset_test["text"], dataset_test["label"])),
        shuffle=True,
        batch_size=32
    )

    for X_batch, y_batch in loader:
        batch_promts = list()
        for text in X_batch:
            batch_promts.append(
                generate_promt_text(text)
            )

        inputs = tokenizer(batch_promts, return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = model.generate(inputs, max_length=200)
        preds = tokenizer.batch_decode(outputs)
        
        for i in range(len(preds)):
            results[result_index] = {
                "dataset": name_dataset_test,
                "label": y_batch[i],
                "text": X_batch[i],
                "predicted": preds[i]
            }
            result_index += 1

    df = pd.DataFrame().from_dict(results, orient="index")
    df = df[["dataset", "label", "predicted"]]
    df.to_csv(f"./reports_hd/{name_dataset_test}_preds_flan.csv")

def run_llm_humor_test_big(name_dataset_test):
    model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", torch_dtype=torch.bfloat16, device_map="auto")                                                                 
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    results = dict()
    result_index = 0
    
    dataset_test = HumorDataset(name_dataset_test)
    dataset_test.load()

    dataset_test = dataset_test.get_test()
    dataset_test["text_list"] = dataset_test["text"].apply(lambda x: word_tokenize(x))
    dataset_test["text_len"] = dataset_test["text_list"].apply(lambda x: len(x))

    dataset_test_small = dataset_test[dataset_test["text_len"] <= 165]

    dataset_test_big = dataset_test[dataset_test["text_len"] > 165]    

    loader_small = DataLoader(
        list(zip(dataset_test_small["text"], dataset_test_small["label"])),
        shuffle=True,
        batch_size=32
    )

    loader_big = DataLoader(
        list(zip(dataset_test_big["text"], dataset_test_big["label"])),
        shuffle=True,
        batch_size=1
    )

    for X_batch, y_batch in loader_big:
        batch_promts = list()
        for text in X_batch:
            batch_promts.append(
                generate_promt_text(text)
            )

        inputs = tokenizer(batch_promts, return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = model.generate(inputs, max_length=200)
        preds = tokenizer.batch_decode(outputs)
        
        for i in range(len(preds)):
            results[result_index] = {
                "dataset": name_dataset_test,
                "label": y_batch[i],
                "text": X_batch[i],
                "predicted": preds[i]
            }
            result_index += 1
    
    for X_batch, y_batch in loader_small:
        batch_promts = list()
        for text in X_batch:
            batch_promts.append(
                generate_promt_text(text)
            )

        inputs = tokenizer(batch_promts, return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = model.generate(inputs, max_length=200)
        preds = tokenizer.batch_decode(outputs)
        
        for i in range(len(preds)):
            results[result_index] = {
                "dataset": name_dataset_test,
                "label": y_batch[i],
                "text": X_batch[i],
                "predicted": preds[i]
            }
            result_index += 1

    df = pd.DataFrame().from_dict(results, orient="index")
    df = df[["dataset", "label", "predicted"]]
    df.to_csv(f"./reports_hd/{name_dataset_test}_preds_flan.csv")


if __name__ == "__main__":

    name_dataset_test = str(sys.argv[1])

    if name_dataset_test not in ["short_jokes", "the_naughtyformer"]:
        run_llm_humor_test(name_dataset_test)
    else:
        run_llm_humor_test_big(name_dataset_test)
