"""
定义LINE模型与预处理数据的接口，负采样规则和模型的训练
"""
import codecs
import datetime
import random

import torch as t
import numpy as np
import json
import os

from torch import optim
from data_preprocess import get_existed_train_data
from model import LINE


class Net2vec(object):
    def __init__(self, app_entity_pairs, appid2entityids, app_count, entity_count):
        """
        初始化整个网络
        :param app_entity_pairs:app,entity对
        :param app_count: app数量
        :param entity_count: 实体数量
        """
        config = json.load(codecs.open("../config/config.json", "r", encoding="utf8"))
        self.device = t.device("cuda:0,1")
        self.app_entity_pairs = app_entity_pairs
        self.app_entity_pairs_set = set(app_entity_pairs)
        self.appid2entityids = appid2entityids
        self.app_count = app_count
        self.entity_count = entity_count
        self.emb_dimension = config["embedding_dim"]
        self.save_dir = config["save_dir"]
        self.initial_lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.negative_num = config['negative_num']
        self.train_times = config["train_times"]

        self.num_batch = len(self.app_entity_pairs) // self.batch_size + 1
        self.model = LINE(self.appid2entityids, self.app_count, self.entity_count, self.emb_dimension).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)

    def get_pos_pairs(self, idx):
        """
        进行批处理的每批的数据
        :param idx:每次批的间隔
        :return:正例的appid的列表和正例的实体id的列表
        """
        pair_app_entity = self.app_entity_pairs[self.batch_size * idx: self.batch_size * (idx + 1)]
        # p_app保存的是app_entity_pairs的前面一个，即appid
        # p_entity保存的是app_entity_pairs的后面一个，即对应的实体id
        p_app = []
        p_entity = []
        for appid, entityid in pair_app_entity:
            p_app.append(appid)
            p_entity.append(entityid)

        return p_app, p_entity

    def neg_sampling(self, p_app, p_entity):
        """
        负采样
        :param p_app:正例的appid
        :param p_entity: 正例的实体id
        :param num: 负采样个数
        :return: 负例的appid的列表和负例的entityid的列表
        """
        neg_app = []
        neg_entity = []
        for num in range(self.negative_num):
            for i in range(len(p_app)):
                p_appid = p_app[i]
                p_entityid = p_entity[i]
                while True:
                    is_head = random.randint(0, 1)
                    if is_head:
                        e = random.randint(0, self.app_count - 1)
                        if (e, p_entityid) in self.app_entity_pairs_set:
                            continue
                        else:
                            neg_app.append(e)
                            neg_entity.append(p_entityid)
                            break
                    else:
                        e = random.randint(0, self.entity_count - 1)
                        if (p_appid, e) in self.app_entity_pairs_set:
                            continue
                        else:
                            neg_app.append(p_appid)
                            neg_entity.append(e)
                            break

        return neg_app, neg_entity

    def current_time(self):
        '''输出当前时间，打印日志的时候用的'''
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': '

    def train(self):
        """
        模型的训练
        :return: None
        """
        for epoch in range(self.train_times + 1):
            process_bar = range(self.num_batch)
            flag = True
            for i in process_bar:
                pos_appid, pos_entityid = self.get_pos_pairs(i)
                neg_appid, neg_entityid = self.neg_sampling(pos_appid, pos_entityid)
                pos_appid = t.LongTensor(pos_appid).cuda()
                pos_entityid = t.LongTensor(pos_entityid).cuda()
                neg_appid = t.LongTensor(neg_appid).cuda()
                neg_entityid = t.LongTensor(neg_entityid).cuda()
                self.optimizer.zero_grad()
                # print('5 ' + current_time())
                loss = self.model(pos_appid, pos_entityid, neg_appid, neg_entityid)
                # print('6 ' + current_time())
                loss.backward()
                # print('7 ' + current_time())
                self.optimizer.step()
                # print('8 ' + current_time())

                if flag:
                    print(self.current_time() + 'epoch ' + str(epoch + 1) + ' | loss is: ' + str(loss.item()))
                    flag = False
                # print('9 ' + current_time())
                # print('-------------------------------')

        self.model.save_para(t.cuda.is_available())


if __name__ == '__main__':
    app_entity_pairs, appid2entityids, app2id_dict, entity2id_dict = get_existed_train_data()
    print('数据预处理完成')

    app_count = len(list(app2id_dict.keys()))
    entity_count = len(list(entity2id_dict.keys()))

    w2v = Net2vec(app_entity_pairs, appid2entityids, app_count, entity_count)
    w2v.train()
