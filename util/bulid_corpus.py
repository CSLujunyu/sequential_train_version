import pandas as pd
import os

corpus_path = '/hdd/lujunyu/dataset/meituan-sa/all_cut_sent.txt'

if __name__ == '__main__':

    # dataframe
    train = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/train/train_cut.pkl')
    dev = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/val/val_cut.pkl')
    testa = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/test/testa_cut.pkl')
    testb = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/test/testb_cut.pkl')

    all_sent = []
    for data in (train, dev, testa, testb):
        all_sent = all_sent + [' '.join(x) for x in data['content']]
    print('The number of sentences: ' , len(all_sent))

    with open(corpus_path, 'w',encoding='utf-8') as f:
        for x in all_sent:
         f.writelines(x + '\n')

    print('finish...')