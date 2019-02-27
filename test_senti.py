import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from model.senti_net import Senti_Net
from model.attr_net import Attr_Net
import bin.senti_test as senti_test

# configure
model_data_path = '/hdd/lujunyu/dataset/meituan/'
model_path = '/hdd/lujunyu/model/meituan/'

conf = {
    'train_data_path' : os.path.join(model_data_path, 'train_han_fasttext.pkl'),
    'dev_data_path' : os.path.join(model_data_path, 'dev_han_fasttext.pkl'),
    'testa_data_path' : os.path.join(model_data_path, 'testa_han_fasttext.pkl'),

    "attr_init_model": '/hdd/lujunyu/model/meituan/D_HAN_MC/attr/',
    "senti_init_model": '/hdd/lujunyu/model/meituan/D_HAN_MC/senti/', #should be set for test

    "rand_seed": 1,
    "learning_rate":1e-3,
    "attribute_threshold":0.5,
    "vocab_size": 266078,    #111695
    "emb_dim": 300,
    "batch_size": 50, #200 for test

    "max_rev_len": 25,
    "max_sent_len": 251,
    'attribute_num': 20,
    'sentiment_num': 4,
    "attribute_prototype_num":4,

    'local_window_size':6,
    'cnn_channel':100,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "<PAD>": 0, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising

    "rnn_dim":300,
    'bert_dim':768*2,

    'multi_head':8,

    'Model': 'D_HAN_MC'
}
conf.update({'save_path' : os.path.join(model_path, conf['Model'] + '/senti/')})


if __name__ == '__main__':

    attr_model = Attr_Net(conf)
    senti_model = Senti_Net(conf)
    senti_test.test(conf, attr_model, senti_model)


