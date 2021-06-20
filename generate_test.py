#from data_create import TextKVQADataSet
import random
import json
import numpy as np

with open("data/QA-scene.json") as f:
    question_json = json.load(f)

combo_keys = list(question_json.keys())
ids = set()
for key in combo_keys:
    ids.add(key.split('_')[0])
ids = list(ids)
np.random.shuffle(ids)
length = len(ids)
train_data = ids[:int(0.8 * length)]
ids = ids[int(0.8 * length):]
val_data = ids[:int(0.1 * length)]
test_data = ids[int(0.1 * length):]
print(train_data, len(train_data))
print(val_data, len(val_data))
print(test_data, len(test_data))
qids_train, qids_val, qids_test = [], [], []
for key in question_json:
    if key.split('_')[0] in train_data:
        for j in range(len(question_json[key]["questions"])):
            value = key + "_" + str(j)
            qids_train.append(value)
    elif key.split('_')[0] in val_data:
        for j in range(len(question_json[key]["questions"])):
            value = key + "_" + str(j)
            qids_val.append(value)
    if key.split('_')[0] in test_data:
        for j in range(len(question_json[key]["questions"])):
            value = key + "_" + str(j)
            qids_test.append(value)


f = open('test_indices.txt', 'w')
for id in qids_test:
    f.write(id + '\n')
f.close()

f = open('train_indices.txt', 'w')
for id in qids_train:
    f.write(id + '\n')
f.close()

f = open('val_indices.txt', 'w')
for id in qids_val:
    f.write(id + '\n')
f.close()