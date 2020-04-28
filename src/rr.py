import codecs
import json
import random
import sys
from src.predicate import predicate_new, predicate_ex, predicate_test

data_dir = json.load(codecs.open('../config/config.json'))["data_input_dir"]
result_dir = json.load(codecs.open('../config/config.json'))["save_dir"]
random.seed(24)

def feature_mmr(app_test_dict):
    """
    对测试集进行mmr的计算，基于特征工程的mmr计算
    :param app_test_dict: 输入的测试集，包含app名称和对应的实体的集合
    :return: 每个app的mmr
    """
    test_review = json.load(open('../benchmark/similar.json', 'r', encoding='utf8'))
    train_data = json.load(open(data_dir + 'trainapp.json', 'r', encoding='utf8'))
    mrr_sum = 0
    for i in app_test_dict:
        mrr = 0
        review = test_review[i]  # 保存的基准的相似的列表
        result_list = list()
        entity_set = set(app_test_dict[i])
        # 遍历train_data，求交集的长度
        for app in train_data:
            train_entity_set = set(train_data[app])
            result_list.append((app, len(train_entity_set.intersection(entity_set))))
        result_list.sort(key=lambda x: x[1], reverse=True)
        result_list = result_list[:20]
        # 计算mmr
        for appname, j in result_list[:20]:
            if appname in review:
                mrr += 1 / (result_list.index((appname, j)) + 1)
        mrr_sum += mrr
        print('{}的RR为：{}'.format(i, round(mrr, 3)))
    print('特征工程RR为：', mrr_sum)


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
        mrr_sum += round(mrr, 3)
        print('{}的RR为：{}'.format(testapp, round(mrr, 3)))
    print('LINE的RR为：', mrr_sum)


if __name__ == '__main__':
    app_test_dict = json.load(open(data_dir + 'testapp.json', 'r', encoding='utf8'))
    print('--------line-------')
    line_mmr()
    print('------------------')
    print('------------------')
    print('-------特征工程-----------')
    feature_mmr(app_test_dict)

