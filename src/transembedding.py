"""
按照每个app对应的实体将result结果文件夹下的实体拼接起来。
有标签的实体按照顺序拼接
无标签的实体（textrank补充的实体）求平均在加到后面
对测试集和训练集的app都要执行这个操作
顺序为[subtype, 公司, type, attentionLevel, 下载量, 来源]
"""
import json
import time

import numpy as np
import itertools
from sklearn.cluster import k_means
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

DATA_DIR = json.load(open('../config/config.json'))['data_input_dir']
RESULT_DIR = json.load(open('../config/config.json'))["save_dir"]
L = ['公司', '开发者']
ex = ['下载量', 'subtype','type','attentionLevel', '来源']

def create_dict():
    train_app = json.load(open(DATA_DIR + 'trainset.json', 'r', encoding='utf8'))
    test_app = json.load(open(DATA_DIR + 'testset.json', 'r', encoding='utf8'))
    # 这个字典存储每个实体对应的属性
    obj2pred_train = dict()
    obj2pred_test = dict()
    for app in train_app:
        for pred, obj in train_app[app].items():
            if pred == '描述':
                continue
            obj2pred_train[obj] = pred
    f = open('./obj2pred_train.json', 'w', encoding='utf8')
    f.write(json.dumps(obj2pred_train, ensure_ascii=False, indent=4))
    print(len(obj2pred_train))

    for test in test_app:
        for pred, obj in test_app[test].items():
            if pred == '描述':
                continue
            obj2pred_test[obj] = pred
    f = open('./obj2pred_test.json', 'w', encoding='utf8')
    f.write(json.dumps(obj2pred_test, ensure_ascii=False, indent=4))
    print(len(obj2pred_test))


def trans2emb():
    create_dict()
    train_app = json.load(open(DATA_DIR + 'trainapp.json', 'r', encoding='utf8'))
    # train_app = json.load(open(DATA_DIR + 'train_40000.json', 'r', encoding='utf8'))
    obj2pred_train = json.load(open('./obj2pred_train.json', 'r', encoding='utf8'))
    entity_dict = json.load(open(DATA_DIR + 'entity2id_dict.json', 'r', encoding='utf8'))
    entityemb = json.load(open(RESULT_DIR + 'entityEmbedding.txt', 'r', encoding='utf8'))
    # 对每个训练app进行concat向量操作
    app_concat_emb = []
    count = 0
    for train in train_app:
        # print(train)
        count += 1
        if count % 1000 == 0:
            print(count)
        no_count = 1  # 不带标签的实体的个数
        temp = [0] * (len(L))  # 存储每个带标签的app实体向量的列表
        temp_no = np.array([0] * len(entityemb[0]))  # 存储不带标签的app实体向量，最后要求平均值
        entity_list = train_app[train]
        for entityname in entity_list:
            pred = obj2pred_train.get(entityname, None)
            # 带标签的实体
            if pred is not None and pred not in ex:
                entityid = entity_dict[entityname]
                index = L.index(pred)
                temp[index] = entityemb[entityid]
            else:
                # 不带标签的实体数量加1
                no_count += 1
                entityid = entity_dict[entityname]
                temp_no = temp_no + np.array(entityemb[entityid])
        # temp_no = temp_no / no_count
        # temp_no = temp_no.reshape(1,-1)
        # print('temp_no:', temp_no.shape)
        for ix, elem in enumerate(temp):
            # 如果存在不存在的实体，补上0
            if elem == 0:
                pad = [0] * len(entityemb[0])
                temp[ix] = pad
        emb = np.array(temp)
        # print('emb:', emb.shape)
        emb = np.vstack((emb, temp_no))
        emb = emb.tolist()
        emb = list(itertools.chain.from_iterable(emb))
        app_concat_emb.append(emb)
    print(len(app_concat_emb))
    print(len(app_concat_emb[0]))
    f = open(RESULT_DIR + 'concatEmbedding.txt', 'w', encoding='utf8')
    f.write(json.dumps(app_concat_emb, ensure_ascii=False))


if __name__ == '__main__':
    trans2emb()

