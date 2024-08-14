# Humor detection  

[ Paper ](https://aclanthology.org/2023.emnlp-main.845/)|[ Slides ](https://docs.google.com/presentation/d/1zZhq-fJfDYL0FLRjqL5AADexz9WFauM2ng6K7eFhzXw/edit?usp=sharing)|[ Datasets ](https://github.com/Humor-Research/hri_tools)|[ Huggingface ](https://huggingface.co/Humor-Research)

This repository contains the code for the article "You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models"


## Updates
- Changing folder names for easy navigation 14/08/2024
- Update README.md 21/12/2023
- Update HuggingFace links 18/12/2023
- Fixing links 15/12/2023


## Data
Access to data and processing functions is available through our library [hri_tools](https://github.com/Humor-Research/hri_tools).


## Models
Our models are available at [HuggingFace](https://huggingface.co/Humor-Research). Our project has published all 50 trained models. If you require a rapid solution for humor classification, please refer to the example provided below. 


### Quick start and example of using the best model:

```
from transformers import RobertaTokenizerFast
from transformers import RobertaForSequenceClassification
from transformers import TextClassificationPipeline

model = RobertaForSequenceClassification.from_pretrained("Humor-Research/humor-detection-comb-23")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", max_length=512, truncation=True)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True)
print(pipe(["That joke so funny"]))
```


## Citation
Please cite our article as follows:
```bibtex
@inproceedings{baranov-etal-2023-told,
    title = "You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models",
    author = "Baranov, Alexander  and
      Kniazhevsky, Vladimir  and
      Braslavski, Pavel",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.845",
    doi = "10.18653/v1/2023.emnlp-main.845",
    pages = "13701--13715",
}

