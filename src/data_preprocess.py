import os
import json
import codecs

DATA_INPUT_DIR = json.load(codecs.open('../config/config.json'))["data_input_dir"]

def get_existed_train_data():

    app_entity_pairs_id_innerlist = json.load(open(DATA_INPUT_DIR + 'app_entity_pairs_id.txt','r',encoding='utf8'))
    appid2entityids_ = json.load(open(DATA_INPUT_DIR + 'appid2entity.json','r',encoding='utf8'))
    app2id_dict = json.load(open(DATA_INPUT_DIR + 'app2id_dict.json','r',encoding='utf8'))
    entity2id_dict = json.load(open(DATA_INPUT_DIR + 'entity2id_dict.json','r',encoding='utf8'))
    appid2entityids = dict()
    for appid_, entityids in appid2entityids_.items():
        appid = int(appid_)
        appid2entityids[appid] = entityids

    app_entity_pairs_id = list()
    for app_entity in app_entity_pairs_id_innerlist:
        app_entity_pairs_id.append((app_entity[0], app_entity[1]))

    return app_entity_pairs_id, appid2entityids, app2id_dict, entity2id_dict

if __name__ == '__main__':
    get_existed_train_data()