#from transformers import BertTokenizer, BertModel, BertConfig
#from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings,
#                                                BertEncoder,
#                                                BertPreTrainedModel)
import torch
from easydict import EasyDict as edict


class BertTokenizerProcessor:
    """
    Tokenize a text string with BERT tokenizer, using Tokenizer passed to the dataset.
    """

    def __init__(self, config, tokenizer):
        self.max_length = config.max_length
        self.bert_tokenizer = tokenizer
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #assert self.bert_tokenizer.encode(self.bert_tokenizer.pad_token) == [0]

    def get_vocab_size(self):
        return self.bert_tokenizer.vocab_size

    def __call__(self, item):
        # [PAD] in self.bert_tokenizer is zero (as checked in assert above)
        token_inds = torch.zeros(self.max_length, dtype=torch.long)

        indices = self.bert_tokenizer.encode(item["text"], add_special_tokens=True)
        indices = indices[: self.max_length]
        token_inds[: len(indices)] = torch.tensor(indices)
        token_num = torch.tensor(len(indices), dtype=torch.long)

        tokens_mask = torch.zeros(self.max_length, dtype=torch.long)
        tokens_mask[: len(indices)] = 1

        results = {
            "token_inds": token_inds,
            "token_num": token_num,
            "tokens_mask": tokens_mask,
        }
        return results


def gen_question_embeddings(text, tokenizer):

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #text = "I am a girl"
    question_config = edict()
    question_config.max_length = 20
    bert_processor = BertTokenizerProcessor(question_config, tokenizer)
    processed_text = bert_processor(
                    {"text": text}
    )
    return processed_text


'''print(processed_text)
processed_dic = processed_text
embeddings = BertEmbeddings(config)
ei = embeddings(processed_dic["token_inds"].reshape(-1, question_config.max_length))
attention_masks = processed_dic["tokens_mask"].reshape(-1, question_config.max_length)
extended_attention_mask = attention_masks.unsqueeze(1).unsqueeze(2)
extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
encoder = BertEncoder(config)
assert not extended_attention_mask.requires_grad
head_mask = [None] * config.num_hidden_layers
encoder_outputs = encoder(
            ei, extended_attention_mask, head_mask=head_mask, output_hidden_states=True
        )
hidden_states = encoder_outputs[1]
seq_output = torch.stack(hidden_states[-4:]).sum(0)
return seq_output
bert = BertModel.from_pretrained('bert-base-uncased', config=config)
print(bert.parameters())
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_text = tokenizer(text, add_special_tokens=True, return_tensors='pt', return_attention_mask=True)
print(tokenized_text)
bert.eval()
with torch.no_grad():
    output = bert(**tokenized_text)
hidden_states = output[2]
embeddings = torch.stack(hidden_states[-4:]).sum(0)
print(embeddings.shape)

bert_embeddings = BertEmbeddings(config)
inputs = bert_embeddings.word_embeddings
print(inputs)'''