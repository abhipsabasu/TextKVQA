import torch
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertEncoder, BertLayerNorm,
                                                BertPreTrainedModel)
from torch import nn


class MMT1(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_weights)  # old versions of pytorch_transformers
        #self.init_weights()

    def forward(self, batch_dict, no_triplets, fixed_ans_emb):
        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary

        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, batch_dict["train_prev_inds"])
        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
        )
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        #print(no_triplets)
        #print(dec_emb.size(), dec_mask.size())
        encoder_inputs = torch.cat(
            [
                batch_dict["text_bert_emb"],
                batch_dict["obj_mmt_in"],
                batch_dict["scene_mmt_in"]
            ] +
            [batch_dict["kb_mmt_in_" + str(i)] for i in range(no_triplets)],
            dim=1,
        )
        #print(encoder_inputs.size())
        encoder_inputs = torch.cat(
            [
                encoder_inputs,
                dec_emb,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                batch_dict["question_mask"],
                batch_dict["pad_obj_mask"],
                batch_dict["scene_mask"].cuda(device=dec_emb.device, non_blocking=True),

            ] +
            [batch_dict["kb_mask_" + str(i)].cuda(device=dec_emb.device, non_blocking=True)
             for i in range(no_triplets)],
            dim=1,
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                dec_mask
            ],
            dim=1,
        )
        # offsets of each modality in the joint embedding space
        txt_max_num = batch_dict["question_mask"].size(-1)
        obj_max_num = batch_dict["pad_obj_mask"].size(-1)
        dec_max_num = dec_mask.size(-1)

        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length
        #print("attention_mask", attention_mask, attention_mask.size())
        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #print("extended_attention_mask_size", extended_attention_mask.size())
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        #print("Next extended mask size", extended_attention_mask.size())
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(
            dec_max_num, encoder_inputs.device
        )
        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        #print("Num_hidden", self.config.num_hidden_layers)
        #print(encoder_inputs.size(), extended_attention_mask.size())
        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        #mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        #mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            #"mmt_seq_output": mmt_seq_output,
            #"mmt_txt_output": mmt_txt_output,
            #"mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_weights)  # old versions of pytorch_transformers
        #self.init_weights(self)

    def forward(self, batch_dict):
        encoder_inputs = self.embeddings(batch_dict["question_indices"])
        attention_mask = batch_dict["question_mask"]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask1 = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask1
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MultiModalTransformer(nn.Module):
    """
    MultiModalTransformer has two transfomers MMT and TextBert.
    """

    def __init__(self, vocab, vocabulary):
        super().__init__()
        self.finetune_modules = []
        self.normalize = True
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.config.num_hidden_layers = 3
        self.lr_scale_text_bert = 0.1
        self.no_triplets = 20
        self.lr_scale_mmt = 1.0
        self.scene_size = 300
        self.kb_vector = 300
        self.vocabulary = vocabulary
        self.hidden_size = 768
        self.obj_dropout = 0.1
        self.num_boxes = 50
        self.obj_feature_size = 2048
        self.vocab = vocab
        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_scene_encoding()
        self._build_kb_encoding()
        self._build_mmt()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768
        # self.text_bert_config = BertConfig(**self.config.text_bert)

        self.text_bert = TextBert.from_pretrained(
            "bert-base-uncased", config=self.config
        )
        # Use a smaller learning rate on text bert when initializing
        # from BERT_BASE
        self.finetune_modules.append(
            {
                "module": self.text_bert,
                "lr_scale": self.lr_scale_text_bert,
            }
        )

    def _build_obj_encoding(self):
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.hidden_size)# - self.num_boxes)
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.obj_feature_size, self.hidden_size # - self.num_boxes
        )
        self.obj_feat_layer_norm = BertLayerNorm(self.hidden_size) # - self.num_boxes)
        self.obj_bbox_layer_norm = BertLayerNorm(self.hidden_size) # - self.num_boxes)
        #self.obj_relation_layer_norm = BertLayerNorm(self.num_boxes)
        self.obj_drop = nn.Dropout(self.obj_dropout)

    def _build_scene_encoding(self):
        self.linear_scene_to_mmt_in = nn.Linear(
            self.scene_size, self.hidden_size
        )

    def _build_kb_encoding(self):
        self.linear_kb_to_mmt_in = nn.Linear(
            self.kb_vector, self.hidden_size
        )

    def _build_mmt(self):
        self.mmt = MMT1(self.config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append(
            {
                "module": self.mmt,
                "lr_scale": self.lr_scale_mmt,
            }
        )

    def _build_output(self):

        self.classifier = nn.Linear(self.hidden_size, self.vocab)
        
    def _forward_scene_encoding(self, batch_dict):
        self.scene_output = self.linear_scene_to_mmt_in(batch_dict["padded_scene_indices"])
        batch_dict["scene_mmt_in"] = self.scene_output
        batch_size = batch_dict["padded_scene_indices"].size(0)
        mask = [1] * 5
        mask = torch.tensor(mask).long()
        mask = mask.unsqueeze(0).expand(batch_size, 5)
        #print("mask_scene", mask)
        batch_dict["scene_mask"] = mask

    def _forward_kb_encoding(self, batch_dict):
        batch_size = len(batch_dict["question_indices"])
        mask = [1, 1, 1, 0, 0]
        mask = torch.tensor(mask).long()
        mask = mask.unsqueeze(0).expand(batch_size, 5)
        #print("mask_kb", mask)
        for i in range(self.no_triplets):
            batch_dict["kb_mmt_in_" + str(i)] = self.linear_kb_to_mmt_in(batch_dict['padded_kb_indices_'+str(i)])
            batch_dict["kb_mask_" + str(i)] = mask

    def forward(self, batch_dict):
        """Main forward method"""
        self._forward_obj_encoding(batch_dict)
        self._forward_scene_encoding(batch_dict)
        self._forward_kb_encoding(batch_dict)
        self._forward_mmt_and_output(batch_dict)
        results_dict = {
            "textvqa_scores": batch_dict["scores"],
            # "spatial_scores": None if not self.use_aux_heads else batch_dict["spatial_head_out"]
        }

        '''if "complete_seqs" in batch_dict:
            results_dict["complete_seqs"] = batch_dict["complete_seqs"].squeeze()
            results_dict["topkscores"] = batch_dict["topkscores"].squeeze()
            results_dict["question_id"] = batch_dict["question_id"].squeeze()'''

        return results_dict

    def _forward_obj_encoding(self, batch_dict):
        # object appearance feature: Faster R-CNN fc7
        #obj_fc7 = self.obj_faster_rcnn_fc7(batch_dict["pad_obj_features"])
        obj_fc7 = batch_dict["pad_obj_features"]
        if self.normalize:
            obj_fc7 = F.normalize(batch_dict["pad_obj_features"], dim=-1)

        obj_feat = obj_fc7

        # remove bbox-area
        obj_bbox = batch_dict["pad_obj_bboxes"]
        #relations = batch_dict["pad_obj_relations"]
        #print(obj_bbox.shape)
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        #obj_mmt_in = torch.cat(
        #    (obj_mmt_in,
        #    self.obj_relation_layer_norm(relations)),
        #    dim=-1
        #)
        batch_dict["obj_mmt_in"] = obj_mmt_in

    def _forward_mmt(self, batch_dict):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(batch_dict)
        batch_dict["text_bert_emb"] = text_bert_out

        mmt_results = self.mmt(batch_dict, self.no_triplets, fixed_ans_emb=self.classifier.weight)
        batch_dict.update(mmt_results)

    def _forward_output(self, batch_dict):
        mmt_dec_output = batch_dict["mmt_dec_output"]
        fixed_scores = self.classifier(mmt_dec_output)
        batch_dict["scores"] = fixed_scores
        #print("fixed_scores", fixed_scores.size())

    def _forward_mmt_and_output(self, batch_dict):
        if self.training:
            #print("Reached here")
            # fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(batch_dict)
            self._forward_output(batch_dict)
        else:
            dec_step_num = batch_dict["train_prev_inds"].size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            batch_dict["train_prev_inds"] = torch.zeros_like(
                batch_dict["train_prev_inds"]
            )
            batch_dict["train_prev_inds"][:, 0] = self.vocabulary["<s>"]

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(batch_dict)
                self._forward_output(batch_dict)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = batch_dict["scores"].argmax(dim=-1)
                batch_dict["train_prev_inds"][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, base_lr):
        optimizer_param_groups = []

        # base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        #self.ocr_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)
        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        #print("ans_emb", ans_emb.size())
        #ocr_emb = self.ocr_layer_norm(ocr_emb)
        #assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        #print("ans_emb expanded", ans_emb.size())
        #ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_emb, prev_inds)
        #print("raw_dec_emb", raw_dec_emb.size())
        # Add position and type embedding for previous predictions
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ans_emb.device)
        #print("position_ids", position_ids)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        #print("position_ids_expand", position_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #print("position_embeddings", position_embeddings.size())
        # Token type ids: 0 -- vocab; 1 -- OCR
        #token_type_ids = prev_inds.ge(ans_num).long()
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    #print("Causal:", mask)
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.reshape(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    #print("batch_offsets_unsqueezed", batch_offsets.size())
    assert batch_offsets.dim() == inds.dim()
    #print("inds", inds.size())
    inds_flat = batch_offsets + inds
    #print("inds_flat", inds_flat, inds_flat.size())
    #print("x_flat", x_flat.size())
    results = F.embedding(inds_flat, x_flat)
    return results






