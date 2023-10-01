import random
random.seed(0)

import numpy as np
np.random.seed(0)

import os
import copy
import pandas as pd
from tqdm import tqdm
from better_profanity import profanity
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=4, progress_bar=False)

from hri_tools import HumorDataset, HRI_PAPER_2023_DATASETS
from hri_tools import calc_divergence_between_target

SAVE_RESULT_PATH = os.path.dirname(__file__)
SAVE_RESULT_PATH = os.path.join(SAVE_RESULT_PATH, "reports", "task_1_part_1_report.csv")


def save_results(results):
    df = pd.DataFrame().from_dict(results, orient="index")
    df.to_csv(SAVE_RESULT_PATH)


def calc_positive_negative_samples(results):

    for name in HRI_PAPER_2023_DATASETS:
        hd = HumorDataset(name)
        hd.load()

        train_data = hd.get_train()
        valid_data = hd.get_valid()
        test_data = hd.get_test()
        all_data = pd.concat([train_data, valid_data, test_data])
        number_of_positive_samples = len(all_data[all_data["label"] == 1])
        number_of_negative_samples = len(all_data[all_data["label"] == 0])
        
        assert number_of_negative_samples + number_of_positive_samples == len(all_data)

        results[name]['number_of_positive_samples'] = copy.copy(number_of_positive_samples)
        results[name]['number_of_negative_samples'] = copy.copy(number_of_negative_samples)

    return results


def calc_average_text_length_in_words(results):

    for name in tqdm(HRI_PAPER_2023_DATASETS):

        hd = HumorDataset(name)
        hd.load()
        hd.run_preprocessing()
        hd.calc_statistics()

        results[name]['mean_word_len_positive_samples'] = hd.mean_word_length_pos
        results[name]['mean_word_len_negative_samples'] = hd.mean_word_length_neg


    return results


def calc_obscene_words(results):

    custom_bad_words = set(open("bad_words.txt").read().split("\n")[:-1])
    profanity.add_censor_words(custom_bad_words)

    for name in tqdm(HRI_PAPER_2023_DATASETS):

        hd = HumorDataset(name)
        hd.load()
        hd.run_preprocessing()

        train_data = hd.get_train()
        valid_data = hd.get_valid()
        test_data = hd.get_test()
        all_data = pd.concat([train_data, valid_data, test_data])

        all_data["is_contain_bad_words"] = all_data["text"].parallel_apply(lambda text: profanity.contains_profanity(text))

        number_of_obscene_in_positive_samples = len(all_data[(all_data["label"] == 1)&(all_data["is_contain_bad_words"])])
        number_of_obscene_in_negative_samples = len(all_data[(all_data["label"] == 0)&(all_data["is_contain_bad_words"])])

        results[name]['number_of_obscene_in_positive_samples'] = number_of_obscene_in_positive_samples
        results[name]['number_of_obscene_in_negative_samples'] = number_of_obscene_in_negative_samples

    return results


def calc_kl(results):

    for name in tqdm(HRI_PAPER_2023_DATASETS):
        hd = HumorDataset(name)
        hd.load()
        kl = calc_divergence_between_target(hd)
        results[name]['kl'] = kl

    return results


def calc_part_size(results):

    for name in tqdm(HRI_PAPER_2023_DATASETS):
        hd = HumorDataset(name)
        hd.load()
        train_size = len(hd.get_train())
        valid_size = len(hd.get_valid())
        test_size = len(hd.get_test())
        results[name]['part_size'] = f"{train_size} / {valid_size} / {test_size}"

    return results


def main():

    results = dict()
    for name in HRI_PAPER_2023_DATASETS:
        results[name] = dict()

    results = calc_positive_negative_samples(results)
    results = calc_obscene_words(results)
    results = calc_average_text_length_in_words(results)
    results = calc_kl(results)
    results = calc_part_size(results)

    save_results(results)
    

if __name__ == "__main__":
    main()