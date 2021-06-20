import json
import nltk
import numpy as np

word_list = set()
word_list.add("yes")
word_list.add("no")
word_list.add("<s>")
word_list.add("<pad>")
word_list.add("<s>")
word_list.add("</s>")
word_list.add("<unk>")
'''with open('ocr_with_bbox.json') as f:
    text = json.load(f)

for key in text:
    for dic in text[key]:
        word = dic['word']
        tokens = nltk.wordpunct_tokenize(word)
        for t in tokens:
            word_list.add(t)'''

with open('KB-business.json') as f:
    kb = json.load(f)
    
for key in kb:
    kb_dict = kb[key]
    values = list(kb_dict.values())
    keys = list(kb_dict.keys())
    for v in values:
        if isinstance(v, float) and np.isnan(v):
            continue
        v = v.lower()
        tokens = nltk.wordpunct_tokenize(v)
        for t in tokens:
            word_list.add(t)
    for k in keys:
        k = k.lower()
        tokens = nltk.wordpunct_tokenize(k)
        for t in tokens:
            word_list.add(t)

with open('data/QA-scene.json') as f:
    qa = json.load(f)

for key in qa:
    qa_dic = qa[key]
    for q in qa_dic["questions"]:
        q = q.lower()
        tokens = nltk.wordpunct_tokenize(q)
        for t in tokens:
            word_list.add(t)
    for a in qa_dic["answers"]:
        if isinstance(a, float) and np.isnan(a):
            continue
        a = a.lower()
        tokens = nltk.wordpunct_tokenize(a)
        for t in tokens:
            word_list.add(t)

with open('final_questions.json') as f:
    qa = json.load(f)

for key in qa:
    qa_dic = qa[key]
    for q in qa_dic["questions"]:
        q = q.lower()
        tokens = nltk.wordpunct_tokenize(q)
        for t in tokens:
            word_list.add(t)
    for a in qa_dic["answers"]:
        if isinstance(a, float) and np.isnan(a):
            continue
        a = a.lower()
        tokens = nltk.wordpunct_tokenize(a)
        for t in tokens:
            word_list.add(t)

print(len(word_list))

file = open('vocab_new.txt', 'w')
for ele in word_list:
    file.write(ele + "\n")
file.close()