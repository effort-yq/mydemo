#!/usr/bin/python3
# -*- coding: utf-8 -*-
# time    : 2023/4/29 16:30
# author  : huangyq
# filename: predict.py
# software: PyCharm
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import json
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModel, BertPreTrainedModel, \
    BertModel
from modeling_nezha.modeling_nezha import NeZhaModel, NeZhaConfig, NeZhaPreTrainedModel
import torch.nn.functional as F

max_seq_len = 512
bert_path = r'./nezha-cn-base'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
# class BertSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertSequenceClassification, self).__init__(config)
#         self.num_labels = 36
#         config.output_hidden_states = True
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(p=0.2)
#         self.high_dropout = nn.Dropout(p=0.5)
#         n_weights = config.num_hidden_layers + 1  # 13
#         weights_init = torch.zeros(n_weights).float()   # torch，tensor向量，
#         weights_init.data[:-1] = -3
#
#         self.layer_weights = torch.nn.Parameter(weights_init)   # 可学习
#         self.classifier = nn.Linear(config.hidden_size, self.num_labels)
#
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids)  # outputs会输出3个向量，第一个是句向量，第2个是pool向量（取cls位置向量），第3个所有层的向量（embedding和12层encoder的向量）
#         hidden_layers = outputs[2]
#
#
#         cls_outputs = torch.stack(
#             [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2   # 取每一层的cls向量进行拼接
#         )
#
#         cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)   # self.layer_weights可以更新，表示学习到的不同层的级别不一样
#
#
#         logits = torch.mean(
#             torch.stack(
#                 [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
#                 dim=0,
#             ),
#             dim=0,
#         )  # high dropout，优化机制
#
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#
#         if labels is not None:
#
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算loss
#             outputs = (loss.mean(),) + outputs
#
#         return outputs  # (loss), scores, (hidden_states), (attentions)

class NezhaSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config):
        super(NezhaSequenceClassification, self).__init__(config)
        self.num_labels = 36
        config.output_hidden_states = True
        self.bert = NeZhaModel(config)
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropout = nn.Dropout(0.1)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])
        self.classifier = nn.Linear(768 * 2, self.num_labels)
        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        all_hidden = outputs[2]

        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden)[:, :1, :].view(13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        f = torch.mean(sequence_output, 1)
        feature = torch.cat((feature, f), 1)
        logit = self.classifier(self.dropout(feature))
        # for i, dropout in enumerate(self.dropouts):
        #     if i == 0:
        #         h = self.fc(dropout(feature))
        #         if loss_fn is not None:
        #             loss = loss_fn(h, y)
        #     else:
        #         hi = self.fc(dropout(feature))
        #         h = h + hi
        #         if loss_fn is not None:
        #             loss = loss + loss_fn(hi, y)
        # if loss_fn is not None:
        #     return h / len(self.dropouts), loss / len(self.dropouts)
        # return h / len(self.dropouts)

        # out = sequence_output.unsqueeze(1)
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        # logit = self.classifier(out)
        outputs = (logit, ) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logit.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs
        return outputs



def get_data(path):
    text = []
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            idx = line['id']
            title = line['title']
            assignee = line['assignee']
            abstract = line['abstract']
            a = title + '。' + assignee + '。' + abstract
            text.append(a)
            ids.append(idx)
    df = pd.DataFrame({'id': ids, 'text': text})
    df['text'] = df['text'].apply(str)
    return df


class Mydatasets(object):
    def __init__(self, text, tokenizer, max_seq_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenizer_result = self.tokenizer.encode_plus(self.text[idx],
                                                      max_length=self.max_seq_len,
                                                      truncation=True,
                                                      truncation_strategy='longest_first')
        return {
            'input_ids': tokenizer_result['input_ids'],
            'attention_mask': tokenizer_result['attention_mask'],
            'token_type_ids': tokenizer_result['token_type_ids'],
        }

def dynamic_batch(batch):
    input_ids, attention_mask, token_type_ids = [], [], []
    collate_max_len = 0
    for sample in batch:
        collate_max_len = max(collate_max_len, len(sample['input_ids']))
    for sample in batch:
        length = len(sample['input_ids'])
        input_ids.append(sample['input_ids'] + [0] * (collate_max_len - length))
        attention_mask.append(sample['attention_mask'] + [0] * (collate_max_len - length))
        token_type_ids.append(sample['token_type_ids'] + [0] * (collate_max_len - length))
    input_ids = torch.tensor(input_ids).long()
    attention_mask = torch.tensor(attention_mask).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
    }



def predict(model, predict_dataloader):
    pre_label_all = []
    for batch in tqdm(predict_dataloader, desc='infer....', total=len(predict_dataloader)):
        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch['input_ids'].to(device),
                "attention_mask": batch['attention_mask'].to(device),
                "token_type_ids": batch['token_type_ids'].to(device),
            }
            outputs = model(**inputs)
            pre_label_all.append(outputs[0].softmax(-1).detach().cpu().numpy())
    pre_label_all = np.concatenate(pre_label_all)
    pre_label_all = np.argmax(pre_label_all, axis=-1)
    return pre_label_all

if __name__ == '__main__':
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = NezhaSequenceClassification.from_pretrained(bert_path).to(device)
    model.load_state_dict(torch.load('./pytorch_model.bin'))
    predict_data = get_data('./testA.json')
    predict_dataset = Mydatasets(predict_data['text'], tokenizer, max_seq_len=max_seq_len)
    predict_dataloader = DataLoader(predict_dataset,  batch_size=32, collate_fn=dynamic_batch)
    result = predict(model, predict_dataloader)
    submit = []
    idx_ = []
    labels = []

    output_file = open('submission.json', 'w', encoding='utf-8')
    for idx, item in predict_data.iterrows():
        id_ = item.id
        idx_.append(id_)
        label = result[idx]
        labels.append(label)
    sub = pd.DataFrame({'id': idx_, 'label': labels})
    sub.to_csv('./result.csv', encoding='utf-8', index=False)
