import torch
from easydict import EasyDict as edict
import numpy as np
import re
from fasttext import load_model
import json


class WordToVectorDict:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        # Check if mean for word split needs to be done here
        return np.mean([self.model.get_word_vector(w) for w in re.split('_|/| ',word)], axis=0)


def _pad_tokens(tokens, PAD_TOKEN, max_length):
    padded_tokens = [PAD_TOKEN] * max_length
    token_length = min(len(tokens), max_length)
    padded_tokens[:token_length] = tokens[:token_length]
    token_length = torch.tensor(token_length, dtype=torch.long)
    return padded_tokens, token_length


class FastTextProcessor:
    """FastText processor, similar to GloVe processor but returns FastText vectors.

    Args:
        config (ConfigNode): Configuration values for the processor.

    """

    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length
        self._load_fasttext_model("wiki.en.bin")
        self.PAD_INDEX = 0
        self.PAD_TOKEN = "<pad>"

    def _load_fasttext_model(self, model_file):

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(
            (self.max_length, self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output

    def __call__(self, item):
        # indices are padded
        indices = self._map_strings_to_indices(item["tokens"])
        # pad tokens
        tokens, length = _pad_tokens(item["tokens"], self.PAD_TOKEN, self.max_length)
        return {
            "padded_token_indices": indices,
            "padded_tokens": tokens,
            "length": length,
        }
    

config = edict()
config.max_length = 5
fasttext_processor = FastTextProcessor(config)
#config = edict()
#config.max_length = 24
#fasttext_processor_ocr = FastTextProcessor(config)
no_triplets = 40


def get_fasttext_vec(dic):
    tokens = list(dic["Scene Categories"].keys())
    ft_processed_tokens = fasttext_processor(
        {"tokens": tokens}
    )
    entry = {"padded_token_indices": ft_processed_tokens["padded_token_indices"],
             "padded_tokens": ft_processed_tokens["padded_tokens"], "length": ft_processed_tokens["length"]}
    return entry



def get_fasttext_kb(kb_json, kb_sim):
    
    kb_list = list(kb_sim.keys())
    id_list = []
    for id in kb_list:
        triplets = kb_json[id]
        title = triplets["has title"]
        for k in triplets:
            if k == "has title":
                continue
            if isinstance(triplets[k], float) and np.isnan(triplets[k]):
                continue
            tokens = [title, k, triplets[k]]
            ft_processed_tokens = fasttext_processor(
                {"tokens": tokens}
            )
            entry = {"padded_token_indices": ft_processed_tokens["padded_token_indices"],
                     "padded_tokens": ft_processed_tokens["padded_tokens"], "length": ft_processed_tokens["length"]}
            id_list.append(entry)
    if len(id_list) >= no_triplets:
        id_list = id_list[:no_triplets]
    else:
        remaining = no_triplets - len(id_list)
        for i in range(remaining):
            ft_processed_tokens = fasttext_processor(
                {"tokens": []}
            )
            entry = {"padded_token_indices": ft_processed_tokens["padded_token_indices"],
                     "padded_tokens": ft_processed_tokens["padded_tokens"], "length": ft_processed_tokens["length"]}
            id_list.append(entry)
    id_dic = {}
    i = 0
    for item in id_list:
        id_dic["padded_kb_indices_" + str(i)] = item["padded_token_indices"]
        id_dic["padded_kb_tokens_" + str(i)] = item["padded_tokens"]
        id_dic["padded_kb_length_" + str(i)] = item["length"]
        i = i + 1
    #print(len(id_list))
    return id_dic
