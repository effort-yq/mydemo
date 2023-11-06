# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2023/4/20 19:47
# software: PyCharm

"""
文件说明：

"""
import torch
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModel, BertPreTrainedModel, \
    BertModel
from sklearn.model_selection import KFold
import torch.nn.functional as F
from modeling_nezha.modeling_nezha import NeZhaModel, NeZhaConfig, NeZhaPreTrainedModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 超参数
model_path = r'./nezha-cn-base'
epochs = 5
learning_rate = 3e-5
max_seq_len = 512    
batch_size = 8   # 8
warmup_ratio = 0.1
eps = 1e-8
linear_learning_rate = 3e-3
weight_decay = 0.01
seed = 2023


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
        outputs = (logit, ) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logit.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs
        return outputs


class EMA(object):
    """
    Maintains (exponential) moving average of a set of parameters.
    使用ema累积模型参数
    Args:
        parameters (:obj:`list`): 需要训练的模型参数
        decay (:obj:`float`): 指数衰减率
        use_num_updates (:obj:`bool`, optional, defaults to True): Whether to use number of updates when computing averages
    Examples::
        >>> ema = EMA(module.parameters(), decay=0.995)
        >>> # Train for a few epochs
        >>> for _ in range(epochs):
        >>>     # 训练过程中，更新完参数后，同步update shadow weights
        >>>     optimizer.step()
        >>>     ema.update(module.parameters())
        >>> # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
        >>> ema.store(module.parameters())
        >>> ema.copy_to(module.parameters())
        >>> # evaluate
        >>> ema.restore(module.parameters())
    Reference:
        [1]  https://github.com/fadel/pytorch_ema
    """  # noqa: ignore flake8"

    def __init__(
            self,
            parameters,
            decay,
            use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed)


def get_data(path):
    text = []
    label = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            idx = line['id']
            title = line['title']
            assignee = line['assignee']
            abstract = line['abstract']
            label_id = line['label_id']
            a = title + '。' + assignee + '。' + abstract
            text.append(a)
            label.append(label_id)

    df = pd.DataFrame({'text': text, 'label': label})
    df['text'] = df['text'].apply(str)
    return df


class Mydatasets(object):
    def __init__(self, text, label, tokenizer, max_seq_len):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenizer_result = self.tokenizer.encode_plus(self.text[idx],
                                                      max_length=self.max_seq_len,
                                                      truncation=True,
                                                      truncation_strategy='longest_first')
        if self.label[idx] is not None:
            return {
                'input_ids': tokenizer_result['input_ids'],
                'attention_mask': tokenizer_result['attention_mask'],
                'token_type_ids': tokenizer_result['token_type_ids'],
                'label': int(self.label[idx]),
            }
        else:
            return {
                'input_ids': tokenizer_result['input_ids'],
                'attention_mask': tokenizer_result['attention_mask'],
                'token_type_ids': tokenizer_result['token_type_ids'],
            }


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def dynamic_batch(batch):
    input_ids, attention_mask, token_type_ids = [], [], []
    label = []
    collate_max_len = 0
    for sample in batch:
        collate_max_len = max(collate_max_len, len(sample['input_ids']))
    for sample in batch:
        length = len(sample['input_ids'])
        input_ids.append(sample['input_ids'] + [0] * (collate_max_len - length))
        attention_mask.append(sample['attention_mask'] + [0] * (collate_max_len - length))
        token_type_ids.append(sample['token_type_ids'] + [0] * (collate_max_len - length))
        if 'label' in sample:
            label.append(int(sample['label']))
    input_ids = torch.tensor(input_ids).long()
    attention_mask = torch.tensor(attention_mask).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    if label:
        label = torch.tensor(label).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label,
        }
    else:
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label,
        }


tokenizer = BertTokenizer.from_pretrained(model_path)
model = NezhaSequenceClassification.from_pretrained(model_path).to(device)


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(model, train_steps):
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in bert_param_optimizer if
                    any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': learning_rate},

        {'params': [p for n, p in linear_param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': linear_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if
                    any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': linear_learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * warmup_ratio,
                                     t_total=train_steps)
    return optimizer, scheduler


def train(train_data, dev_data=None):
    train_dataset = Mydatasets(train_data['text'],
                               train_data['label'],
                               tokenizer,
                               max_seq_len=max_seq_len)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_data), batch_size=batch_size,
                                  collate_fn=dynamic_batch)
    t_total = len(train_dataloader) * epochs
    optimizer, scheduler = build_optimizer(model, t_total)
    global_step = 1
    best_score = 0.0
    model.zero_grad()
    for eo in range(epochs):
        print("----- Epoch {}/{} ;-----".format(eo + 1, epochs))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", )
        epoch_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {
                "input_ids": batch['input_ids'].to(device),
                "attention_mask": batch['attention_mask'].to(device),
                "token_type_ids": batch['token_type_ids'].to(device),
                "labels": batch['label'].to(device),
            }
            output = model(**inputs)
            loss = output[0]
            loss.backward()
            epoch_loss += loss.item()
            epoch_iterator.set_postfix(loss=loss.item())

            output1 = model(**inputs)
            loss_adv = output1[0]
            loss_adv.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
        # model.save_pretrained(os.path.join('save_model', f'epoch_{eo + 1}'))
        val_score = evaluate(dev_data, model)
        print("currently!! best_score is{}, cur_score is "
              "{}".format(best_score, val_score))
        if val_score >= best_score:
            best_score = val_score
            torch.save(model.state_dict(), './pytorch_model.bin')
            tqdm.write('saving_model')


def evaluate(eval_data, model):
    eval_dataset = Mydatasets(eval_data['text'],
                              eval_data['label'],
                              tokenizer, max_seq_len=max_seq_len)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=dynamic_batch)
    eval_iterator = tqdm(eval_dataloader, desc="valid")
    pre_label_all = []
    label_true = []
    for step, batch in enumerate(eval_iterator):
        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch['input_ids'].to(device),
                "attention_mask": batch['attention_mask'].to(device),
                "token_type_ids": batch['token_type_ids'].to(device),
            }
            label_true.extend(batch['label'].detach().cpu().numpy())
            outputs = model(**inputs)

            pre_label_all.append(outputs[0].softmax(-1).detach().cpu().numpy())
    pre_label_all = np.concatenate(pre_label_all)

    return f1_score(label_true, np.argmax(pre_label_all, axis=-1), average='macro')



if __name__ == '__main__':
    set_seed(seed)
    origin_data = get_data('./train.json')
    skf = KFold(n_splits=5, random_state=seed, shuffle=True)
    for idx, (train_index, valid_index) in enumerate(skf.split(origin_data)):
        train_data = origin_data.iloc[train_index]
        train_data.index = range(len(train_data))
        valid_data = origin_data.iloc[valid_index]
        valid_data.index = range(len(valid_data))
        break
    train(origin_data, valid_data)
