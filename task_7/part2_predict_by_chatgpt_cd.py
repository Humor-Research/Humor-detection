import os
import json
import sys
import time
from datetime import datetime


import grequests
import pandas as pd
from tqdm import tqdm
from requests_ratelimiter import LimiterSession


from hri_tools.dataset_loader import ConversationalDataset

dataset = str(sys.argv[1])

session = LimiterSession(per_minute=3400)

with open("chatgpt_config.json") as f:
    configs = json.load(f)

proxies=dict(
    http=f"socks5://{configs['proxy_login']}:{configs['proxy_password']}@{configs['proxy_ip']}",
    https=f"socks5://{configs['proxy_login']}:{configs['proxy_password']}@{configs['proxy_ip']}"
)


hd = ConversationalDataset(dataset)


test_data = hd.get_data()

test_data["chatgpt_preds"] = None
test_data["dataset"] = dataset


def generate_request_to_chatgpt(prompt, proxies, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "model": "gpt-3.5-turbo-0301",
        "temperature": 0.2
    }
    req = grequests.post(
        url,
        headers=headers,
        json=data,
        proxies=proxies,
        session=session,
        timeout=5
    )
    return req

def process_answer(request_answer):

    if request_answer is None:
        return None

    if request_answer.status_code == 200:
        return request_answer.text
    else:
        return None



while test_data["chatgpt_preds"].isna().sum() != 0:

    df_to_predict = test_data[test_data["chatgpt_preds"].isna()]
    df_predicted = test_data[~test_data["chatgpt_preds"].isna()]

    print(datetime.now(), "|", dataset, "| Need to predict: |", len(df_to_predict))

    requests_to_api = list()

    for index, row in df_to_predict.iterrows():
        requests_to_api.append(
            generate_request_to_chatgpt(
                prompt=f"""
                Classify the text into funny or unfunny.
                Text: {row["text"]}
                Label: 
                """,
                proxies=proxies,
                api_key=configs["openai_token"]
            )
        )
    
    request_ans = grequests.map(requests_to_api, size=8)
    df_to_predict["chatgpt_preds"] = request_ans
    df_to_predict["chatgpt_preds"] = df_to_predict["chatgpt_preds"].apply(process_answer)

    test_data = pd.concat([df_to_predict, df_predicted])

    test_data[["dataset", "chatgpt_preds"]].to_csv(f"./reports_cd/{dataset}_pred_chatgpt.csv")

    time.sleep(1)

test_data = test_data[["dataset", "chatgpt_preds"]]
test_data.to_csv(f"./reports_cd/{dataset}_pred_chatgpt.csv")

