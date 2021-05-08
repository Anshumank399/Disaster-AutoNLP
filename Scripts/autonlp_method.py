# -*- coding: utf-8 -*-
"""
Created on Sat May  8 00:38:44 2021

@author: anshu
"""

import pandas as pd
from autonlp import AutoNLP

client = AutoNLP()
client.login(token="api_XVqCYoASjgfNPruEsQAVSWTHaBiyOvJODu")


# Read Data
data = pd.read_csv("Data/train.csv")
# 70-30 Split
train = data[:5329]
test = data[5329:]

train.to_csv("Data/train_labelled.csv", index=False)
test.to_csv("Data/test_labelled.csv", index=False)

project = client.create_project(
    name="disaster_detection", task="binary_classification", language="en", max_models=5
)

project.upload(
    filepaths=["Data/train_labelled.csv"],
    split="train",
    col_mapping={
        "text": "text",
        "target": "target",
    },
)

project.upload(
    filepaths=["Data/test_labelled.csv"],
    split="valid",
    col_mapping={
        "text": "text",
        "target": "target",
    },
)
