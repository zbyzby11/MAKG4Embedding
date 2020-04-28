"""
用rank_min作为评测指标
"""
import codecs
import json
import random
import time

from src.predicate import predicate_test

random.seed(12)
data_dir = json.load(codecs.open('../config/config.json'))["data_input_dir"]
result_dir = json.load(codecs.open('../config/config.json'))["save_dir"]


def feature_rank_min(app_test_dict):
    """
    对测试集进行rank-min的计算，基于特征工程的rank-min计算
    :param app_test_dict: 输入的测试集，包含app名称和对应的实体的集合
    :return: 每个app的rank-min
    """
    test_review = json.load(open('../benchmark/similar.json', 'r', encoding='utf8'))
    train_data = json.load(open(data_dir + 'trainapp.json', 'r', encoding='utf8'))
    for i in app_test_dict:
        count = 0  # 记录不在benchmark列表中的个数
        review = test_review[i]  # 保存的基准的相似的列表
        result_list = list()
        entity_set = set(app_test_dict[i])
        # 遍历train_data，求交集的长度
        for app in train_data:
            train_entity_set = set(train_data[app])
            result_list.append((app, len(train_entity_set.intersection(entity_set))))
        result_list.sort(key=lambda x: x[1], reverse=True)
        result_list = result_list[:20]
        for appname, j in result_list:
            if appname in review:
                print('{}的rank-min的值为：{}'.format(i, result_list.index((appname, j)) + 1))
                break
            else:
                count += 1
        if count == len(review):
            print('{}的rank-min的值为：{}'.format(i, 21))


def rank_min_line():
    app_si_dict = predicate_test(20)
    test_review = json.load(open('../benchmark/similar_1.json', 'r', encoding='utf8'))
    for testapp in app_si_dict:
        benchmark_app_list = test_review[testapp]
        predicate_list = app_si_dict[testapp]
        count = 0  # 记录不在benchmark列表中的个数
        for i in predicate_list:
            if i in benchmark_app_list:
                print('{}的rank-min的值为：{}'.format(testapp, predicate_list.index(i) + 1))
                break
            else:
                count += 1
        if count == len(benchmark_app_list):
            print('{}的rank-min的值为：{}'.format(testapp, 21))


if __name__ == '__main__':
    app_test_dict = json.load(open(data_dir + 'testapp.json', 'r', encoding='utf8'))
    print('--------line-------')
    t = time.time()
    rank_min_line()
    print('line消耗：', time.time() - t)
    print('------------------')
    print('------------------')
    print('-------特征工程-----------')
    t = time.time()
    feature_rank_min(app_test_dict)
    print('特征工程消耗：', time.time() - t)
