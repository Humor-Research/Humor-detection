import sys

import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import DataLoader

from hri_tools.dataset_loader import ConversationalDataset

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
    
    dataset_test = ConversationalDataset(name_dataset_test)

    dataset_test = dataset_test.get_data()

    loader = DataLoader(
        list(zip(dataset_test["text"], [1] * len(dataset_test))),
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
    df.to_csv(f"./reports_cd/{name_dataset_test}_preds_flan.csv")


if __name__ == "__main__":

    name_dataset_test = str(sys.argv[1])

    run_llm_humor_test(name_dataset_test)
    