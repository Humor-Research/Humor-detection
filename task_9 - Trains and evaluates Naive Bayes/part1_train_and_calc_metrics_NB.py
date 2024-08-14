import pickle
import os

import pandas as pd
from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from hri_tools import HRI_PAPER_2023_DATASETS, HumorDataset

all_data = list()
for name in HRI_PAPER_2023_DATASETS:
    all_data.append(
        HumorDataset(
            name=name
        )
    )

for i in range(len(all_data)):
    all_data[i].load()

for i in range(len(all_data)):
    all_data[i].run_preprocessing()

for i in range(len(all_data)):
    all_data[i].build_vocab()


all_idx = 0
all_results = dict()
for i in tqdm(range(len(all_data))):

    train = all_data[i]
    print(f'Start train on {train.name}')

    def return_self(x):
        return x

    if train.name == "short_jokes":
        cv = CountVectorizer(analyzer=return_self, max_features=50000)
    else:
        cv = CountVectorizer(analyzer=return_self)

    train_data = cv.fit_transform(train.get_train()['text_preprocessed'])

    model = MultinomialNB(alpha=0.5)
    model.fit(train_data.toarray(), train.get_train()['label'])

    for j in range(len(all_data)):
        test = all_data[j]

        test_data = cv.transform(test.get_test()['text_preprocessed'])
        test_data = test_data.toarray()
        predicted = model.predict(test_data)
        all_results[all_idx] = {
            'train_data': train.name,
            'test_data': test.name,
            'accuracy_score': accuracy_score(test.get_test()['label'], predicted),
            'precision_score' : precision_score(test.get_test()['label'], predicted),
            'recall_score': recall_score(test.get_test()['label'], predicted),
            'f1_score': f1_score(test.get_test()['label'], predicted)
        }

        all_idx += 1
    
    
    os.mkdir(f"./models/{train.name}/")

    with open(f"./models/{train.name}/model.pickle", "wb") as f:
        pickle.dump(model, f)

    with open(f"./models/{train.name}/vectorizer.pickle", "wb") as f:
        pickle.dump(cv, f)
    
    del model
    del cv


res_df = pd.DataFrame().from_dict(all_results, orient='index')
res_df.to_csv('./reports/result_NB.csv')
