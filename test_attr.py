import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from model.attr_net import Attr_Net
import bin.attr_test_att as attr_test

# configure
model_data_path = '/hdd/lujunyu/dataset/meituan/'
model_path = '/hdd/lujunyu/model/meituan/'

conf = {
    'train_data_path' : os.path.join(model_data_path, 'train.pkl'),
    'dev_data_path' : os.path.join(model_data_path, 'dev.pkl'),
    'testa_data_path' : os.path.join(model_data_path, 'testa.pkl'),
    'domain_emb_path': os.path.join(model_data_path, 'emb4data.pkl'),
    'fasttext_emb_path': os.path.join(model_data_path, 'fasttext_emb4data.pkl'),
    'tencent_emb_path': os.path.join(model_data_path, 'tencent_emb4data.pkl'),

    "attr_init_model": '/hdd/lujunyu/model/meituan/D_HAN_MC/attr5/', #should be set for test

    "rand_seed": None,
    "learning_rate":3e-4,
    "vocab_size": 212307,    #111695
    "domain_emb_dim": 100,
    "tencent_emb_dim": 200,
    "fasttext_emb_dim": 300,
    "batch_size": 10, #200 for test

    "max_rev_len": 30,
    "max_sent_len": 300,
    'attribute_num': 20,
    'attribute_prototype': 3,

    "max_to_keep": 1,
    "num_scan_data": 10,
    "<PAD>": 0, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising

    "rnn_layers":3,
    "sent_attention_layers":2,
    "doc_attention_layers":2,
    "rnn_dim":300,

    "drop_out":False,
    'batch_normalization':False,

    'Model': 'D_HAN_MC'
}
conf.update({'save_path' : os.path.join(model_path, conf['Model'] + '/attr5/')})
conf.update({'emb_dim' : conf['domain_emb_dim'] + conf['tencent_emb_dim']})


if __name__ == '__main__':

    model = Attr_Net(conf)
    attr_test.test(conf, model)


