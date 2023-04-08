import matplotlib.pyplot as plt
import numpy as np
from transformers import ViltProcessor, ViltForImagesAndTextClassification, AutoImageProcessor, ViTModel, ViTMAEModel, ViltModel
from models import *
from dataset import *
from tqdm import tqdm   
import ipdb
import csv
import ast
import json
import random
import copy
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
unimodal_models = ["roberta", "mae", "resnet50","roberta_mae","vilt"]
models = ["vilt_pretrained", "vlmo", "x-vlm" ]
num_fails = 2


fails = {}
for model in unimodal_models + models:
    fails[model] = []
model_data = []

for model in unimodal_models + models:
    model_path = f"./models/{model}/model_0/"
    data = []
    with open(model_path+"results.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0][:3]!="dev":
                continue
            data.append([row[0], int(row[1]), int(row[2])])
            if row[1]!=row[2]:
                fails[model].append(row[0])
    model_data.append(data)


id_sentence_mapping = {}
with open("./nlvr/nlvr2/data/dev.json") as f:
    for line in f:
        line_dict = json.loads(line)
        identifier = line_dict['identifier']
        sentence = line_dict['sentence']
        id_sentence_mapping[identifier] = sentence

# for model in unimodal_models + models:
#     df = pd.read_csv(f"./models/{model}/model_0/results.csv", header=None)
#     labels = df[2].to_numpy()
#     preds = df[1].to_numpy()
#     precision, recall, f1, _ = precision_recall_fscore_support(preds, labels, average='binary')
#     print("model : ", model)
#     print("Accuracy : ", (preds==labels).mean())
#     print("Precision : ", precision)
#     print("Recall : ", recall)
#     print("F1 : ", f1)
#     print(labels.mean())



# print("Multimodal models")
# for model in fails:
#     print(model + " fails")
#     shuffled_i = copy.deepcopy(fails[model])
#     random.shuffle(shuffled_i)
#     print(shuffled_i[:num_fails])
# for i in range(len(models)):
#     for j in range(len(models)):
#         if i==j:
#             continue
#         shuffled_i = copy.deepcopy(fails[models[i]])
#         shuffled_j = copy.deepcopy(fails[models[j]])
#         random.shuffle(shuffled_i)
#         random.shuffle(shuffled_j)
#         for k in shuffled_i:
#             if k not in shuffled_j:
#                 print(f"Fail in {models[i]} but success in {models[j]}")
#                 print(k)
#                 break
# for i in range(len(models)):
#     j = (i+1)%len(models)
#     k = (i+2)%len(models)
#     shuffled_i = copy.deepcopy(fails[models[i]])
#     shuffled_j = copy.deepcopy(fails[models[j]])
#     shuffled_k = copy.deepcopy(fails[models[k]])
#     random.shuffle(shuffled_i)
#     random.shuffle(shuffled_j)
#     random.shuffle(shuffled_k)
#     for l in shuffled_i:
#         if l in shuffled_j and l not in shuffled_k:
#             print(f"Fail in {models[i]} and {models[j]} but success in {models[k]}")
#             print(l)
#             break
# for i in range(len(models)):
#     j = (i+1)%len(models)
#     k = (i+2)%len(models)
#     shuffled_i = copy.deepcopy(fails[models[i]])
#     shuffled_j = copy.deepcopy(fails[models[j]])
#     shuffled_k = copy.deepcopy(fails[models[k]])
#     random.shuffle(shuffled_i)
#     random.shuffle(shuffled_j)
#     random.shuffle(shuffled_k)
#     for l in shuffled_i:
#         if l not in shuffled_j and l not in shuffled_k:
#             print(f"Fail in {models[i]} but success in {models[k]} and {models[j]}")
#             print(l)
#             break
# for i in range(len(models)):
#     j = (i+1)%len(models)
#     k = (i+2)%len(models)
#     shuffled_i = copy.deepcopy(fails[models[i]])
#     shuffled_j = copy.deepcopy(fails[models[j]])
#     shuffled_k = copy.deepcopy(fails[models[k]])
#     random.shuffle(shuffled_i)
#     random.shuffle(shuffled_j)
#     random.shuffle(shuffled_k)
#     for l in shuffled_i:
#         if l in shuffled_j and l in shuffled_k:
#             print(f"Fail in {models[i]}, {models[k]} and {models[j]}")
#             print(l)
#             break



import spacy

nlp = spacy.load('en_core_web_sm')  # Load the pre-trained model

def contains_number(statement):
    doc = nlp(statement)  # Process the statement with the model
    for token in doc:
        if token.like_num:  # Check if the token represents a numeric value
            return True
    return False

numeric_ids = []
for id in id_sentence_mapping:
    if contains_number(id_sentence_mapping[id]):
        numeric_ids.append(id)

big_model = ["vilt_pretrained", "vlmo", "x-vlm" ]
numeric_success = {"vilt_pretrained":0, "vlmo":0, "x-vlm":0}
non_numeric_success = {"vilt_pretrained":0, "vlmo":0, "x-vlm":0}
numeric_counts = {"vilt_pretrained":0, "vlmo":0, "x-vlm":0}
non_numeric_counts = {"vilt_pretrained":0, "vlmo":0, "x-vlm":0}



for id in id_sentence_mapping:
    for model in big_model:
        
        if id in numeric_ids:
            numeric_counts[model] += 1
            if id not in fails[model]:
                numeric_success[model] += 1
        else:
            non_numeric_counts[model] += 1
            if id not in fails[model]:
                non_numeric_success[model] += 1


numeric_acc = [numeric_success[model]/numeric_counts[model] for model in big_model]
non_numeric_acc = [non_numeric_success[model]/non_numeric_counts[model] for model in big_model]
print(numeric_acc)
print(non_numeric_acc)
# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
index = range(len(models))

# Plot the numeric accuracies
plt.bar(index, numeric_acc, bar_width, label='Numeric')

# Plot the non-numeric accuracies
plt.bar([i + bar_width for i in index], non_numeric_acc, bar_width, label='Non-Numeric')

# Add labels, title, and legend
plt.xlabel('Models')
plt.ylabel('Accuracy')

plt.title('Model Accuracies by Category')
plt.xticks([i + bar_width/2 for i in index], models)
plt.legend()
plt.ylim([0.65, 0.9])

plt.savefig("numeric_acc.png")
# Show the plot
plt.show()





