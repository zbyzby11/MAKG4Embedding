import torch
import os
from net2vec import Net2vec

from data_preprocess import get_existed_train_data

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    app_entity_pairs, appid2entityids, app2id_dict, entity2id_dict = get_existed_train_data()
    print('数据预处理完成')

    app_count = len(list(app2id_dict.keys()))
    entity_count = len(list(entity2id_dict.keys()))
    print("app数量：", app_count)
    print('实体数量：', entity_count)
    print('网络中节点总数量：', app_count + entity_count)
    torch.cuda.manual_seed(32)
    w2v = Net2vec(app_entity_pairs, appid2entityids, app_count, entity_count)
    print('模型的信息：', w2v.model.config)
    w2v.train()
