# -*- coding: utf-8 -*-
"""
Created on Sat May  8 00:38:44 2021

@author: anshuman.kirty@gmail.com
"""

import pandas as pd
from autonlp import AutoNLP

client = AutoNLP()
client.login(token="")


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

# Start Training
project.train()

# Check Model Status
project.refresh()
print(project)

predict = []
result = []
# The unlabelled Test Dataset
test_unlabelled = pd.read_csv("Data/test.csv")
for i in range(0, len(test_unlabelled)):
    result.append(
        client.predict(
            project="disaster_detection",
            model_id=157464,
            input_text=test_unlabelled["text"][i],
        )[0]
    )


def get_prediction_value(result):
    result = pd.DataFrame(result)
    zero_score = float(result[result["label"] == "0"]["score"])
    one_score = float(result[result["label"] == "1"]["score"])
    if one_score >= zero_score:
        return 1
    else:
        return 0


# test_unlabelled = test_unlabelled[:306]
test_unlabelled["result"] = result
test_unlabelled["Predicted"] = test_unlabelled["result"].apply(get_prediction_value)
