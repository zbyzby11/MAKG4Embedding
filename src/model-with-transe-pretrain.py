"""
LINE模型的定义，主要重写nn.Module下的forward方法和实现保存模型参数的方法
"""
import codecs
import json

import torch as t
import numpy as np

from torch import nn
from torch.functional import F

L = ['subtype', '公司', 'type', 'attentionLevel', '下载量', '开发者', '开发者0', '开发者1', '开发者2', '开发者3', '来源']


class LINE(nn.Module):
    def __init__(self, appid2entityids, app_count, entity_count, embedding_dim):
        """
        Line模型的初始化
        :param app_count:app的数量
        :param entity_count:实体的数量
        :param embedding_dim:embedding的维度
        """
        super(LINE, self).__init__()
        appEmb = json.load(open('./app_emb_pre_train_without_textrank.txt', 'r', encoding='utf8'))
        entEmb = json.load(open('./ent_emb_pre_train_without_textrank.txt', 'r', encoding='utf8'))
        self.app_count = app_count  # app数量
        self.entity_count = entity_count  # 实体数量
        self.embedding_dim = embedding_dim
        self.appid2entityids = appid2entityids  # 一个app对应的实体列表，{app_id: [entity_id]}
        self.app_emb = nn.Embedding(app_count, embedding_dim).from_pretrained(t.Tensor(appEmb), freeze=False)
        self.entity_emb = nn.Embedding(entity_count, embedding_dim).from_pretrained(t.Tensor(entEmb), freeze=False)
        self.config = json.load(codecs.open("../config/config.json", "r", encoding="utf8"))
        self.result_dir = self.config["save_dir"]

        # self.init_weight()

    # def init_weight(self):
    #     """
    #     模型参数的初始化
    #     :return: None
    #     """
    #     nn.init.xavier_uniform_(self.app_emb.weight.data)
    #     nn.init.xavier_uniform_(self.entity_emb.weight.data)

    def forward(self, pos_app, pos_entity, neg_app, neg_entity):
        """
        重写forward函数，传入四个Variable
        :param pos_app: 正例app
        :param pos_entity: 正例实体
        :param neg_app: 负例app
        :param neg_entity: 负例实体
        :return: 损失函数的值
        """
        # 正例得分
        pos_emb_app = self.app_emb(pos_app)
        pos_emb_entity = self.entity_emb(pos_entity)
        pos_score = t.mul(pos_emb_app, pos_emb_entity)
        pos_score = pos_score.squeeze()
        pos_score = t.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)
        # 负例得分
        neg_emb_app = self.app_emb(neg_app)
        neg_emb_entity = self.entity_emb(neg_entity)
        neg_score = t.mul(neg_emb_app, neg_emb_entity)
        neg_score = neg_score.squeeze()
        neg_score = t.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        # 总得分
        pos_score_sum = t.sum(pos_score)
        neg_score_sum = t.sum(neg_score)
        all_score = -1 * (pos_score_sum + neg_score_sum)

        return all_score

    def save_para(self, use_cuda):
        # entity embedding
        if use_cuda:
            entity_embedding = self.entity_emb.weight.detach().cpu().numpy().tolist()
        else:
            entity_embedding = self.entity_emb.weight.data.numpy()

        ent_f = open(self.result_dir + 'entityEmbedding.txt', "w")
        ent_f.write(json.dumps(entity_embedding))
        ent_f.close()

        # app embedding
        if use_cuda:
            app_embedding = self.app_emb.weight.detach().cpu().numpy().tolist()
        else:
            app_embedding = self.app_emb.weight.data.numpy()

        app_f = open(self.result_dir + 'appEmbedding.txt', "w")
        app_f.write(json.dumps(app_embedding))
        app_f.close()

        # app embedding, calculate by averaging entity embedding
        app_embedding_average_entity = [0.0] * self.app_count
        entity_embedding = np.array(entity_embedding)
        for appid in range(self.app_count):
            if appid in self.appid2entityids:
                entityids = self.appid2entityids[appid]
                vec = list(np.sum(entity_embedding[entityids], axis=0) / len(entityids))
            else:
                vec = [0.0] * self.embedding_dim
            app_embedding_average_entity[appid] = vec

        app_f = open(self.result_dir + 'appEmbeddingAverageEntity.txt', "w")
        app_f.write(json.dumps(app_embedding_average_entity))
        app_f.close()

    def sava_emb_concat(self, use_cuda):
        if use_cuda:
            entity_embedding = self.entity_emb.weight.detach().cpu().numpy().tolist()
        else:
            entity_embedding = self.entity_emb.weight.data.numpy()

        ent_f = open(self.result_dir + 'entityEmbedding.txt', "w")
        ent_f.write(json.dumps(entity_embedding))
        ent_f.close()

        # app embedding
        if use_cuda:
            app_embedding = self.app_emb.weight.detach().cpu().numpy().tolist()
        else:
            app_embedding = self.app_emb.weight.data.numpy()

        app_f = open(self.result_dir + 'appEmbedding.txt', "w")
        app_f.write(json.dumps(app_embedding))
        app_f.close()

        # concat embedding
        for appid in self.appid2entityids:
            # 每个app对应的实体列表
            entity_list = self.appid2entityids[str(appid)]
            # 遍历实体
            for entityid in entity_list:
                ...

