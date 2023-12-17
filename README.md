# Humor detection  

This repository contains the code for the article "You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models"


## Updates
- Update HuggingFace links 18/12/2023
- Fixing links 15/12/2023


## Data
Access to data and processing functions is available through our library [hri_tools](https://github.com/Humor-Research/hri_tools).


## Models
Our models are available at [HuggingFace](https://huggingface.co/Humor-Research).
Example of usage:
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
@inproceedings{JokeTwice2023,
   title={You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models},
   author={Alexander Baranov, Vladimir Kniazhevsky and Pavel Braslavski},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2023}
}
