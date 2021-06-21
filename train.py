import sys
#sys.path.append(r'/sdc1/abhipsa/TextKVQA/detector')
import os
import torch
import numpy as np
import logging
import json
from bisect import bisect

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
#from evaluate import accuracy


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def accuracy(truth, pred, vocab, batch_size):
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
    return acc / batch_size


def clip_gradients(model, max_grad_l2_norm):
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)


def compute_loss(scores, targets, loss_mask):
    one = torch.Tensor([1.0])
    assert scores.dim() == 3 and loss_mask.dim() == 2
    scores = torch.transpose(scores, 1, 2)
    losses = nn.CrossEntropyLoss(reduction='none')(scores, targets)
    losses *= loss_mask
    count = torch.max(torch.sum(loss_mask), one.to(losses.device))
    loss = torch.sum(losses) / count
    return loss


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k, lower_keys(v)) for k, v in x.items())
    elif isinstance(x, str):
        return x.lower()
    else:
        return x


def get_optim_scheduler(
    optimizer_grouped_parameters,
    base_lr,
):
    optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)
    warmup_iters = 1000
    warmup_factor = 0.2
    lr_decay_iters = [14000, 19000]
    lr_decay = 0.1

    def lr_update(_iter):
        if _iter <= warmup_iters:
            alpha = float(_iter / warmup_iters)
            return (warmup_factor * (1.0 - alpha)) + alpha
        else:
            idx = bisect(lr_decay_iters, _iter)
            return pow(lr_decay, idx)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, warmup_scheduler


def main(create_ocr_feats=False, create_obj_feats=False, recognize_scenes=False, combine_features=False,
         find_string_ocr_simi=False, train=True):
    #Text recognition and detection
    image_dir = 'data/images'
    feats_out_dir = 'data/frcnn'
    ocr_out_dir = 'result'
    scene_out_dir = 'data/scene_recognizer'
    if not os.path.exists(feats_out_dir):
        os.makedirs(feats_out_dir)
    if not os.path.exists(ocr_out_dir):
        os.makedirs(ocr_out_dir)
    if not os.path.exists(scene_out_dir):
        os.makedirs(scene_out_dir)
    if create_ocr_feats:
        recognize_and_detect(image_dir, ocr_out_dir)
    if create_obj_feats:
        do_frcnn(image_dir, feats_out_dir)
    if recognize_scenes:
        scene_recognizer(image_dir, scene_out_dir)
    if find_string_ocr_simi:
        title_score_dict = {}
        with open("KB-business.json") as f:
            kb_json = json.load(f)
        with open('ocr_with_bbox.json') as f:
            text = json.load(f)
        max_score = -1
        for key in text:
            #image_key = key.split("_")[0]
            title_score_dict[key] = {}
            for knowledge_key in kb_json:
                title = kb_json[knowledge_key]["has title"]
                title_score, ocr_word_info_with_max_score = get_title_score(title, text[key])
                if title_score == 0.0:
                    continue
                title_score_dict[key][knowledge_key] = title_score
            title_score_dict[key] = {k: v for k, v in sorted(title_score_dict[key].items(), key=lambda item: item[1], reverse=True)}
        with open("knowledge_similarity_wo_area.json", "w") as f:
            json.dump(title_score_dict, f)
        #get_fasttext_kb(kb_json, title_score_dict)

    if combine_features:
        combine()
    if train:
        print("Enter training phase")
        base_lr = 0.0001
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info(f"Device: {device}, Numer of GPUs: {n_gpu}")
        #images = np.load("combo.npy", allow_pickle=True)
        #images = images.item()
        with open('ocr_with_bbox.json') as f:
            text = json.load(f)
        with open("data/QA-scene.json") as f:
            question_json = json.load(f)
        question_json = lower_keys(question_json)
        with open("KB-business.json") as f:
            kb_json = json.load(f)
        kb_json = lower_keys(kb_json)
        with open("knowledge_similarity.json") as f:
            kb_sim_json = json.load(f)
        with open('scene_dets.json') as f:
            scene = json.load(f)
        scene = lower_keys(scene)
        vocab = {}
        i = 0
        f = open('vocab.txt', 'r')
        for line in f:
            vocab[line.strip()] = i
            i = i + 1
        #print(vocab)
        f.close()
        train_dataset = TextKVQADataSetKey(question_json, 'test_indices1.txt', mode="train")
        #val_dataset = TextKVQADataSetKey(question_json, 'val_indices.txt', mode="val")
        #print("dataset-len", len(dataset))
        #print("num_questions", len(list(text.keys())))
        #print(dataset.trainX)
        batch_size = 32
        num_workers = 4
        validation_split = 0.1
        test_split = 0.1
        shuffle_dataset = True
        random_seed = 42
        dataset_size = len(train_dataset)
        #print(dataset_size, "dataset_size")
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        #train_indices = [ind for ind in indices if ind not in test_indices]
        train_indices, val_indices = indices[split:], indices[:split]
        
        #val_split = int(np.floor(validation_split * len(train_indices)))
        #train_indices, val_indices = train_indices[val_split:], train_indices[:val_split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        #test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            #num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            train_dataset,
            sampler=valid_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        print(len(train_loader), len(val_loader))
        len_vocab = len(list(vocab.keys()))
        model = MultiModalTransformer(len_vocab, vocab)
        model.to(device)
        checkpoint_path = 'best_model_17.tar'

        optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
        optimizer, warmup_scheduler = get_optim_scheduler(
            optimizer_grouped_parameters, base_lr
        )
        #print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        #checkpoint_train = torch.load("best_model_13.tar")
        #model.load_state_dict(checkpoint_train['model_state_dict'])
        #optimizer.load_state_dict(checkpoint_train['optimizer_state_dict'])
        model.train()

        #for name, param in model.named_parameters():
        #    if not name.startswith('classifier'):
        #        param.requires_grad = False
        #    #if param.requires_grad:
        #    print(name)
        trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        logger.info(f"Training Parameters: {trainable_params}")
        best_val_loss = 10000
        logger.info(f"Num batches: {len(train_loader)}")
        for e in range(10):
            loss_values = []
            epoch_acc = []
            for step, dic in enumerate(train_loader):
                logger.info(f"Epoch {e}, Iteration {step}")
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
                    #print(obj_features)
                    #print(type(bbox_dic))
                    #print(scene_dict)
                    if not bool(scene_dict):
                        continue
                    question = question_json[key]["questions"][j]
                    answer = question_json[key]["answers"][j]
                    if isinstance(answer, float) and np.isnan(answer):
                        continue
                    data_dict[i] = create_data(kb_sim, kb_json, obj_features, bbox_dic, scene_dict, question, answer, vocab)
                    i = i+1

                batch_dataset = TextKVQADataSet(data_dict)
                if len(batch_dataset) == 0:
                    print(dic['data'])
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

                loss_batch = compute_loss(batch_dict["scores"], batch_dict["answer"], batch_dict["train_loss_mask"])
                logger.info(f"Iteration {step}, Loss {loss_batch.item()}")
                loss_batch.backward(retain_graph=True)
                clip_gradients(model, 0.25)
                optimizer.step()
                warmup_scheduler.step()
                model.zero_grad()
                argmax_inds = batch_dict["scores"].argmax(dim=-1)
                acc = accuracy(batch_dict["answer"].tolist(), argmax_inds.tolist(), vocab, argmax_inds.size(0))
                epoch_acc.append(acc)
                loss_values.append(loss_batch.item())
            acc_avg = float(sum(epoch_acc) / len(epoch_acc))
            loss_avg = float(sum(loss_values) / len(loss_values))
            logger.info(f"Epoch {e}, Epoch-Loss {loss_avg}, Epoch-Acc {acc_avg}")
            val_loss = []
            val_acc = []
            model.eval()
            with torch.no_grad():
                print(len(val_loader))
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
                    loss_batch = compute_loss(batch_dict["scores"], batch_dict["answer"], batch_dict["train_loss_mask"])
                    val_loss.append(loss_batch.item())
                    argmax_inds = batch_dict["scores"].argmax(dim=-1)
                    acc = accuracy(batch_dict["answer"].tolist(), argmax_inds.tolist(), vocab, argmax_inds.size(0))
                    val_acc.append(acc)
                    #print(val_loss)
            val_acc_avg = float(sum(val_acc) / len(val_acc))
            val_loss_avg = float(sum(val_loss) / len(val_loss))
            logger.info(f"Epoch {e}, Val-Epoch-Loss {val_loss_avg}, Val-Epoch-Acc {val_acc_avg}")
            if val_loss_avg < best_val_loss:
                logger.info(f"Saving Checkpoint: {checkpoint_path}, acc: {val_acc_avg}")
                model_to_save = model.module if hasattr(model, "module") else model
                best_val_loss = val_loss_avg
                torch.save(
                    {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                        #"global_step": global_step,
                        "current_val_loss": val_loss_avg,
                        "epoch_id": e,
                    },
                    checkpoint_path,
                )
            model.train()
                    

if __name__ == "__main__":
    create_ocr_feats = False
    create_obj_feats = False
    recognize_scenes = False
    combine_features = False
    find_string_ocr_simi = False
    train = True
    main(create_ocr_feats, create_obj_feats, recognize_scenes, combine_features, find_string_ocr_simi, train)
