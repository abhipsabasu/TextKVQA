import sys
#sys.path.append(r'/sdc1/abhipsa/TextKVQA/detector')
import os
import torch
import numpy as np
import logging
import json
from bisect import bisect
from torchviz import make_dot
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from detector.text_detect_and_recognize import recognize_and_detect
from extract_FRCNN_feats import do_frcnn
from scene_recognizer.run_placesCNN_unified import scene_recognizer
from data_create import TextKVQADataSetKey, TextKVQADataSet
from string_ocr_similarity import get_title_score
from mmt import MultiModalTransformer
from combine_image_info import combine
from data_create import create_data


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k, lower_keys(v)) for k, v in x.items())
    elif isinstance(x, str):
        return x.lower()
    else:
        return x

def accuracy(truth, pred, vocab, batch_size, id_list, reverse_vocab):
    acc = 0
    for i in range(batch_size):
        truth_i = truth[i]
        pred_i = pred[i]
        truth_idx = truth_i.index(vocab['</s>'])
        if vocab['</s>'] not in pred_i:
            pred_idx = len(pred_i)
        else:
            pred_idx = pred_i.index(vocab['</s>'])
        true = truth_i[:truth_idx]
        prediction = pred_i[:pred_idx]
        if true == prediction:
            acc = acc + 1
        else:
            true_ans = ' '.join([reverse_vocab[c] for c in true])
            pred_ans = ' '.join([reverse_vocab[c] for c in prediction])
            print(id_list[i], true_ans, "_________", pred_ans)
    return acc / batch_size


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    with open('ocr_with_bbox.json') as f:
        text = json.load(f)
    with open("combined_questions.json") as f:
        question_json = json.load(f)
    question_json = lower_keys(question_json)
    with open("KB-business.json") as f:
        kb_json = json.load(f)
    kb_json = lower_keys(kb_json)
    with open("knowledge_similarity.json") as f:
        kb_sim_json = json.load(f)
    with open('scene_dets.json') as f:
        scene = json.load(f)
    base_lr = 0.0001
    scene = lower_keys(scene)
    vocab = {}
    i = 0
    f = open('vocab_new.txt', 'r')
    for line in f:
        vocab[line.strip()] = i
        i = i + 1
    f.close()
    dataset = TextKVQADataSetKey(question_json, "new_indices.txt", mode="test")
    dataset_size = len(dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    data_sampler = SubsetRandomSampler(indices)
    val_loader = DataLoader(
                dataset,
                sampler=data_sampler,
                batch_size=32
            )
    len_vocab = len(list(vocab.keys()))
    model = MultiModalTransformer(len_vocab, vocab)
    model.to(device)
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
    optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    checkpoint = torch.load("best_model_13.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch_id']
    loss = checkpoint['current_val_loss']
    model.eval()
    reverse_vocab = {v:k for (k,v) in vocab.items()}
    acc_list = []
    for step, dic in enumerate(val_loader):
        data_dict = {}
        i = 0
        for combo_key in dic['data']:
            split_keys = combo_key.split('_')
            j = int(split_keys[-1])
            key = '_'.join(split_keys[:-1])
            if key not in text:
                continue
            if key not in scene:
                continue
            images = combine(key, text[key], scene[key])
            if images is None:
                continue
            kb_sim = kb_sim_json[key]
            obj_features = images[-2]
            bbox_dic = images[-1].item()
            scene_dict = images[-3]
            # print(obj_features)
            # print(type(bbox_dic))
            # print(scene_dict)
            if not bool(scene_dict):
                continue
            question = question_json[key]["questions"][j]
            answer = question_json[key]["answers"][j]
            if isinstance(answer, float) and np.isnan(answer):
                continue
            data_dict[i] = create_data(kb_sim, kb_json, obj_features, bbox_dic, scene_dict, question,
                                       answer, vocab)
            data_dict[i]['id'] = combo_key
            i = i + 1
        batch_dataset = TextKVQADataSet(data_dict)
        batch_loader = DataLoader(
            batch_dataset,
            batch_size=len(batch_dataset)
        )
        batch_dict = None
        for _, data in enumerate(batch_loader):
            batch_dict = data['data']
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.cuda(device=device, non_blocking=True)
        model(batch_dict)
        argmax_inds = batch_dict["scores"].argmax(dim=-1)
        acc = accuracy(batch_dict["answer"].tolist(), argmax_inds.tolist(), vocab, argmax_inds.size(0),
                       batch_dict["id"], reverse_vocab)
        acc_list.append(acc)

    print("Acc", float(sum(acc_list) / len(acc_list)))

eval()
