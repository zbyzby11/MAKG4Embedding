"""
运用新的算法（拼接向量的算法来测试每个测试app的mrr等评价指标）
"""
import itertools
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = json.load(open('../config/config.json'))['data_input_dir']
RESULT_DIR = json.load(open('../config/config.json'))["save_dir"]
L = ['公司', '开发者']
ex = ['下载量', 'subtype', 'type', 'attentionLevel', '来源']


def predicate_test(k):
    testapp = json.load(open(DATA_DIR + 'testapp.json', 'r', encoding='utf8'))
    obj2pred_test = json.load(open('./obj2pred_test.json', 'r', encoding='utf8'))
    concat_emb = json.load(open(RESULT_DIR + 'concatEmbedding.txt', 'r', encoding='utf8'))
    appid_dict = json.load(open(DATA_DIR + 'app2id_dict.json', 'r', encoding='utf8'))
    entityid_dict = json.load(open(DATA_DIR + 'entity2id_dict.json', 'r', encoding='utf8'))
    entity_emb = json.load(open(RESULT_DIR + 'entityEmbedding.txt', 'r', encoding='utf8'))
    predicate_dict = dict()
    concat_emb = np.array(concat_emb)
    for test in testapp:
        no_count = 1  # 不带标签的实体的个数
        temp_no = np.array([0] * len(entity_emb[0]))  # 存储不带标签的app实体向量，最后要求平均值
        temp = [0] * (len(L))  # 存储每个带标签的app实体向量的列表
        # 取出测试集中每个app对应的实体集合
        test_entity_list = testapp[test]
        for entityname in test_entity_list:
            pred = obj2pred_test.get(entityname, None)
            # 带标签的实体
            if pred is not None and pred not in ex:
                entityid = entityid_dict.get(entityname, None)
                if entityid is None:
                    continue
                index = L.index(pred)
                temp[index] = entity_emb[entityid]
            else:
                # 不带标签的实体数量加1
                no_count += 1
                if entityname in entityid_dict:
                    entityid = entityid_dict[entityname]
                    temp_no = temp_no + np.array(entity_emb[entityid])
        temp_no = temp_no / no_count
        for ix, elem in enumerate(temp):
            # 如果存在不存在的实体，补上0
            if elem == 0:
                pad = [0] * len(entity_emb[0])
                temp[ix] = pad
        emb = np.array(temp)
        # print(emb.shape)
        # print(temp_no.shape)
        emb = np.vstack((emb, temp_no))
        emb = emb.tolist()
        emb = list(itertools.chain.from_iterable(emb))
        emb = np.array(emb).reshape(1, -1)
        # print(emb.shape)
        # print(concat_emb.shape)
        distance = cosine_similarity(emb, concat_emb)
        # distance = np.abs(distance)[0]
        sim_index = np.argsort(distance[0])[-k:]
        # print(sim_index)
        sim_index = list(reversed(sim_index))
        # print(sim_index)
        l = []
        for idx in sim_index:
            for app, appid in appid_dict.items():
                if idx == appid:
                    l.append(app)
        predicate_dict[test] = l
    # print(predicate_dict)
    return predicate_dict


def line_mmr():
    app_si_dict = predicate_test(20)
    test_review = json.load(open('../benchmark/similar.json', 'r', encoding='utf8'))
    mrr_sum = 0
    for testapp in app_si_dict:
        benchmark_app_list = test_review[testapp]
        predicate_list = app_si_dict[testapp]
        mrr = 0
        for i in predicate_list:
            if i in benchmark_app_list:
                mrr += 1 / (predicate_list.index(i) + 1)
        mrr_sum += mrr
        print('{}的MMR为：{}'.format(testapp, round(mrr, 3)))
        # print(round(mrr, 3))
    print('LINE的MRR为：', mrr_sum)
    # print('----------------------------')
    # print('----------------------------')
    # print('----------------------------')
    # print('----------------------------')
    # for testapp in app_si_dict:
    #     benchmark_app_list = test_review[testapp]
    #     predicate_list = app_si_dict[testapp]
    #     count = 0  # 记录不在benchmark列表中的个数
    #     for i in predicate_list:
    #         if i in benchmark_app_list:
    #             print('{}的rank-min的值为：{}'.format(testapp, predicate_list.index(i) + 1))
    #             break
    #         else:
    #             count += 1
    #     if count == len(benchmark_app_list):
    #         print('{}的rank-min的值为：{}'.format(testapp, 21))


def line_rank_min():
    app_si_dict = predicate_test(20)
    test_review = json.load(open('../benchmark/similar.json', 'r', encoding='utf8'))
    for testapp in app_si_dict:
        benchmark_app_list = test_review[testapp]
        predicate_list = app_si_dict[testapp]
        count = 0  # 记录不在benchmark列表中的个数
        for i in predicate_list:
            if i in benchmark_app_list:
                print('{}的rank-min的值为：{}'.format(testapp, predicate_list.index(i) + 1))
                # print(predicate_list.index(i) + 1)
                break
            else:
                count += 1
        if count == len(benchmark_app_list):
            print('{}的rank-min的值为：{}'.format(testapp, 21))
            # print(21)


if __name__ == '__main__':
    print('rr------------')
    print('rr------------')
    print('rr------------')
    line_mmr()
    # predicate_test(20)
    print('rank_min------------------------------------')
    print('rank_min------------------------------------')
    print('rank_min------------------------------------')
    line_rank_min()
