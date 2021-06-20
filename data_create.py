import json
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import nltk
import math
from word_embeddings import gen_question_embeddings
from fasttext_vec import get_fasttext_vec, get_fasttext_kb
data_dict = {}


class TextKVQADataSet(Dataset):

    def __init__(self, data_dict):
        super().__init__()
        self.data_dict = data_dict

    def __getitem__(self, index):
        x = self.data_dict[index]
        return {'data': x}

    def __len__(self):
        return len(list(self.data_dict.keys()))


class TextKVQADataSetKey(Dataset):

    def __init__(self, question_json, test_indices, mode):
        super().__init__()
        self.data_dict = {}
        i = 0
        if mode == 'test':
            f = open(test_indices, 'r')
            for line in f:
                line = line.strip()
                self.data_dict[i] = line
                i = i + 1
            f.close()
        else:
            test_idx = []
            f = open(test_indices, 'r')
            for line in f:
                line = line.strip()
                test_idx.append(line)
            f.close()
            combo_keys = list(question_json.keys())
            for key in question_json:
                if key not in combo_keys:
                    continue
                for j in range(len(question_json[key]["questions"])):
                    value = key + "_" + str(j)
                    if value in test_idx:
                        continue
                    self.data_dict[i] = value
                    i = i + 1

    def __getitem__(self, index):
        x = self.data_dict[index]
        return {'data': x}

    def __len__(self):
        return len(list(self.data_dict.keys()))


def _pad_image_features(features, bboxes, num_boxes, max_feat_num, tensorize=True):
    mix_num_boxes = min(int(num_boxes), max_feat_num)
    mask = [1] * (int(mix_num_boxes))
    while len(mask) < max_feat_num:
        mask.append(0)

    mix_boxes_pad = np.zeros((max_feat_num, 4))
    mix_boxes_pad[:mix_num_boxes] = bboxes[:mix_num_boxes]

    mix_features_pad = np.zeros((max_feat_num, 2048))
    mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

    #mix_relations_pad = np.zeros((max_feat_num, num_boxes))
    #mix_relations_pad[:mix_num_boxes] = relations[:mix_num_boxes]

    if not tensorize:
        return mix_features_pad, mask, mix_boxes_pad

    # tensorize
    pad_features = torch.tensor(mix_features_pad).float()
    mask_features = torch.tensor(mask).long()
    pad_bboxes = torch.tensor(mix_boxes_pad).float()
    #pad_relations = torch.tensor(mix_relations_pad).float()
    return pad_features, mask_features, pad_bboxes #, pad_relations


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    assert (boxBArea + boxAArea - interArea) != 0

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def add_spatial_relations(bboxes, num_box):
    distance_threshold = 0.5
    relations = np.zeros((num_box, num_box))
    xmin, ymin, xmax, ymax = np.split(bboxes, 4, axis=1)
    image_h = 1.0
    image_w = 1.0
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    image_diag = math.sqrt(image_h ** 2 + image_w ** 2)
    for i in range(num_box):
        bbA = bboxes[i]
        # (YK): Padded bbox
        if sum(bbA) == 0:
            continue
        relations[i, i] = 12
        for j in range(i + 1, num_box):
            bbB = bboxes[j]
            # (YK): Padded bbox
            if sum(bbB) == 0:
                continue
            if (
                xmin[i] < xmin[j]
                and xmax[i] > xmax[j]
                and ymin[i] < ymin[j]
                and ymax[i] > ymax[j]
            ):
                relations[i, j] = 1  # covers
                relations[j, i] = 2  # inside
            elif (
                xmin[j] < xmin[i]
                and xmax[j] > xmax[i]
                and ymin[j] < ymin[i]
                and ymax[j] > ymax[i]
            ):
                relations[i, j] = 2
                relations[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)

                # class 3: i and j overlap
                if ioU >= 0.5:
                    relations[i, j] = 3
                    relations[j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt((y_diff) ** 2 + (x_diff) ** 2)
                    if diag < distance_threshold * image_diag:
                        sin_ij = y_diff / diag
                        cos_ij = x_diff / diag
                        # first quadrant
                        if sin_ij >= 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)
                            label_j = math.pi + label_i
                        # fourth quadrant
                        elif sin_ij < 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij) + 2 * math.pi
                            label_j = label_i - math.pi
                        # second quadrant
                        elif sin_ij >= 0 and cos_ij < 0:
                            label_i = np.arccos(cos_ij)
                            label_j = label_i + math.pi
                        # third quadrant
                        else:
                            label_i = 2 * math.pi - np.arccos(cos_ij)
                            label_j = label_i - math.pi
                        # goes from [1-8] + 3 -> [4-11]
                        # if (adj_matrix_shared["1"][i, j] > 0):
                        relations[i, j] = (
                            int(np.ceil(label_i / (math.pi / 4))) + 3
                        )
                        relations[j, i] = (
                            int(np.ceil(label_j / (math.pi / 4))) + 3
                        )
    return relations


def create_data(kb_sim, kb_json, obj_features, bbox_dic, scene_dict, question, answer, vocab):
    max_obj_num = 50

    #print("KB reading started")
    kb_dict = get_fasttext_kb(kb_json, kb_sim)
    #print("KB reading ended")
    obj_num_boxes = bbox_dic['num_boxes']
    obj_bboxes = bbox_dic['bbox']
    image_width = bbox_dic['image_width']
    image_height = bbox_dic['image_height']
    obj_bboxes[:, 0] = obj_bboxes[:, 0] / image_width
    obj_bboxes[:, 1] = obj_bboxes[:, 1] / image_height
    obj_bboxes[:, 2] = obj_bboxes[:, 2] / image_width
    obj_bboxes[:, 3] = obj_bboxes[:, 3] / image_height

    #obj_spatial_graph = add_spatial_relations(obj_bboxes, obj_num_boxes)
    pad_obj_features, pad_obj_mask, pad_obj_bboxes = _pad_image_features(
        obj_features, obj_bboxes, obj_num_boxes, max_obj_num
    )
    del obj_bboxes, obj_features
    scene_dict = get_fasttext_vec(scene_dict)
    # print(key)
    processed_question = gen_question_embeddings(question, tokenizer)
    # new_key = key + "_" + str(i)
    # print(new_key)
    #one_hot_vectors = []
    answer_tokens = nltk.wordpunct_tokenize(answer)
    num_tokens = len(answer_tokens)
    answer_tensor = [vocab["<s>"]]
    target = []
    for w in answer_tokens:
        #one_hot = torch.zeros(len(list(vocab.keys())))
        #one_hot[vocab[w]] = 1
        #one_hot_vectors.append(one_hot)
        answer_tensor.append(vocab[w])
        target.append(vocab[w])
    #one_hot = torch.zeros(len(list(vocab.keys())))
    #one_hot[vocab["</s>"]] = 1
    #one_hot_vectors.append(one_hot)
    target.append(vocab["</s>"])
    answer_tensor.append(vocab["</s>"])
    rem_len = 12 - len(answer_tensor)
    for i in range(rem_len):
        answer_tensor.append(vocab["<pad>"])
    for i in range(12 - len(target)):
        target.append(vocab["<pad>"])
    #print(len(answer_tensor))
    #one_hot_vectors = torch.stack(one_hot_vectors)
    target = torch.tensor(target).long()
    answer_tensor = torch.tensor(answer_tensor).long()
    #print(one_hot_vectors.size())
    train_loss_mask = torch.zeros(12, dtype=torch.float)
    dec_step_num = min(1 + num_tokens, 12)
    train_loss_mask[:dec_step_num] = 1.0
    #print("loss_mask", train_loss_mask.size())
    data = {"answer": target, "question_indices": processed_question["token_inds"],
            "num_question_tokens": processed_question["token_num"], "question_mask": processed_question["tokens_mask"],
            "pad_obj_features": pad_obj_features, "pad_obj_mask": pad_obj_mask, "pad_obj_bboxes": pad_obj_bboxes,
            "num_boxes": obj_num_boxes,
            "padded_scene_indices": scene_dict["padded_token_indices"],
            "padded_scene_tokens": scene_dict["padded_tokens"], "padded_scene_length": scene_dict["length"],
            "train_prev_inds": answer_tensor, "train_loss_mask": train_loss_mask} #torch.zeros(12, dtype=torch.long)}
    data.update(kb_dict)
    #print("Questions complete")
    #print("Image Complete")
    return data


'''
class TextKVQADataSetKey(Dataset):

    def __init__(self, combo, question_json, kb_sim_file, kb_file):
        super().__init__()
        self.data_dict = {}
        self.images = combo
        self.question_json = question_json
        self.kb_sim_file = kb_sim_file
        self.kb_file = kb_file
        i = 0
        combo_keys = list(self.images.keys())
        for key in question_json:
            if key not in combo_keys:
                continue
            for j in range(len(question_json[key]["questions"])):
                value = key + "_" + str(j)
                self.data_dict[i] = value
                i = i + 1

    def __getitem__(self, index):
        combo_key = self.data_dict[index]
        split_keys = combo_key.split('_')
        j = int(split_keys[-1])
        key = '_'.join(split_keys[:-1])
        kb_sim = self.kb_sim_file[key]
        obj_features = self.images[key][-2]
        bbox_dic = self.images[key][-1].item()
        scene_dict = self.images[key][-3]
        if not bool(scene_dict):
            return None
        question = self.question_json[key]["questions"][j]
        answer = self.question_json[key]["answers"][j]
        x = create_data(kb_sim, self.kb_file, obj_features, bbox_dic, scene_dict, question, answer)
        return {'data': x}
        
        
class TextKVQADataSet(Dataset):

    def __init__(self, image_feat_file, question_file, kb_sim_file, kb_file):
        super().__init__()
        self.image_file = image_feat_file
        self.question_file = question_file
        self.trainX, self.trainY = self._create_data(self.image_file, self.question_file, kb_sim_file, kb_file)

    def __getitem__(self, index):
        x = self.trainX[index]
        y = self.trainY[index]
        return {'data': x, 'target': y}

    def __len__(self):
        return len(list(self.trainX.keys()))

    def _pad_image_features(self, features, bboxes, num_boxes, max_feat_num, tensorize=True):
        mix_num_boxes = min(int(num_boxes), max_feat_num)
        mask = [1] * (int(mix_num_boxes))
        while len(mask) < max_feat_num:
            mask.append(0)

        mix_boxes_pad = np.zeros((max_feat_num, 4))
        mix_boxes_pad[:mix_num_boxes] = bboxes[:mix_num_boxes]

        mix_features_pad = np.zeros((max_feat_num, 2048))
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        if not tensorize:
            return mix_features_pad, mask, mix_boxes_pad

        # tensorize
        pad_features = torch.tensor(mix_features_pad).float()
        mask_features = torch.tensor(mask).long()
        pad_bboxes = torch.tensor(mix_boxes_pad).float()

        return pad_features, mask_features, pad_bboxes

    def _create_data(self, image_feat_file, question_file, kb_sim_file, kb_file):
        max_obj_num = 50
        images = np.load(image_feat_file, allow_pickle=True)
        images = images.item()
        trainY = {}
        with open(question_file) as f:
            question_json = json.load(f)
        with open(kb_file) as f:
            kb_json = json.load(f)
        with open(kb_sim_file) as f:
            kb_sim_json = json.load(f)
        print("KB reading started")
        kb_dict = get_fasttext_kb(kb_json, kb_sim_json)
        print("KB reading ended")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        i = 0
        print(len(list(images.keys())))
        for key in question_json:
            if key not in images:
                continue

            obj_features = images[key][-2]
            bbox_dic = images[key][-1].item()
            obj_num_boxes = bbox_dic['num_boxes']
            obj_bboxes = bbox_dic['bbox']
            image_width = bbox_dic['image_width']
            image_height = bbox_dic['image_height']
            obj_bboxes[:, 0] = obj_bboxes[:, 0] / image_width
            obj_bboxes[:, 1] = obj_bboxes[:, 1] / image_height
            obj_bboxes[:, 2] = obj_bboxes[:, 2] / image_width
            obj_bboxes[:, 3] = obj_bboxes[:, 3] / image_height
            pad_obj_features, pad_obj_mask, pad_obj_bboxes = self._pad_image_features(
                obj_features, obj_bboxes, obj_num_boxes, max_obj_num
            )
            scene_dets = images[key][-3].item()
            scene_dict = get_fasttext_vec(scene_dict)
            #print(key)
            for j in range(len(question_json[key]["questions"])):
                question = question_json[key]["questions"][j]
                processed_question = gen_question_embeddings(question, tokenizer)
                #new_key = key + "_" + str(i)
                new_key = i
                #print(new_key)
                trainY[new_key] = 0 #question_json[key]["answers"]
                data_dict[new_key] = {}
                data_dict[new_key]["question_indices"] = processed_question["token_inds"]
                data_dict[new_key]["num_question_tokens"] = processed_question["token_num"]
                data_dict[new_key]["question_mask"] = processed_question["tokens_mask"]
                data_dict[new_key]["pad_obj_features"] = pad_obj_features
                data_dict[new_key]["pad_obj_mask"] = pad_obj_mask
                data_dict[new_key]["pad_obj_bboxes"] = pad_obj_bboxes
                data_dict[new_key]["padded_scene_indices"] = scene_dict["padded_token_indices"]
                data_dict[new_key]["padded_scene_tokens"] = scene_dict["padded_tokens"]
                data_dict[new_key]["padded_scene_length"] = scene_dict["length"]
                data_dict[new_key]["knowledge_triplets"] = kb_dict[key]
                i = i + 1
        print("Questions complete")
        print("Image Complete")
        return data_dict, trainY'''

