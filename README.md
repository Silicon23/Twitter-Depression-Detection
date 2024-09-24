# Twitter-Depression-Detection
Working repository for Group 5 of Wharton Global Youth Summer Program 2024 Data Science Academy.

## Notebooks

[Project R Notebook](https://colab.research.google.com/drive/1L7-CchELRZ9HN55tqHn2PSFzPdDOW6XX?usp=sharing)

[BERT Training](https://colab.research.google.com/drive/1poW0MLgpWgeIcbzgUIl5BhtFiR3z7pRo?usp=sharing)

[BELT Training](https://colab.research.google.com/drive/1h8CW6SC9VFHFfn6Oe4E6Rc2xyIYJjKwv?usp=sharing)

## Datasets

[Twitter 2015](https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media)

[Harnessing the Power of Hugging Face
Transformers for Predicting Mental
Health Disorders in Social
Networks](https://arxiv.org/abs/2306.16891)

## Models

[2015 BERT](https://huggingface.co/Silicon23/BERTForDetectingDepression-Twitter2015)

[2020 BERT](https://huggingface.co/Silicon23/BERTForDetectingDepression-Twitter2020)

[BELT](./BELT/)

For more information see [BERT for longer texts](https://github.com/mim-solutions/bert_for_longer_texts)

To use this model, clone the above repo and do:
```py
from belt_nlp.bert_classifier_with_pooling import BERTClassifierWithPooling
model = BERTClassifierWithPooling().load(‘path/to/model/directory’)

```