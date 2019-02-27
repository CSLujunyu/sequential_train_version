import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 160000
max_rev_len = 30
max_sent_len = 300

save_train_path = '/hdd/lujunyu/dataset/meituan/train.pkl'
save_dev_path = '/hdd/lujunyu/dataset/meituan/dev.pkl'
save_testa_path = '/hdd/lujunyu/dataset/meituan/testa.pkl'
save_domain_emb_path = '/hdd/lujunyu/dataset/meituan/emb4data.pkl'
save_tencent_emb_path = '/hdd/lujunyu/dataset/meituan/tencent_emb4data.pkl'
save_fasttext_emb_path = '/hdd/lujunyu/dataset/meituan/fasttext_emb4data.pkl'

def padding_rev(rev):
    if len(rev)< max_rev_len:
        rev = rev + [''] * (max_rev_len - len(rev))
    else:
        rev = rev[:max_rev_len]
    return rev

def generate_domain_embedding(vocab):

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build domain emb')

    with open('/hdd/lujunyu/dataset/meituan/glove/vectors.txt', 'r', encoding='utf-8') as f:
        domain_dic = {}
        for line in f.readlines():
            row = line.strip().split(' ')
            if len(row[1:]) != 100:
                continue
            domain_dic[row[0]] = row[1:]

    print(len(set(vocab) & set(domain_dic.keys())),
          ' word have pre-trained domain embedding , total word:%d ' % len(vocab))

    domain_both = set(vocab) & set(domain_dic.keys())


    domain_embed4data = []
    for word in vocab:
        ###
        if word in domain_both:
            domain_embed4data.append(np.array(domain_dic[word]))
            domain_both = domain_both - set(word)
        else:
            domain_embed4data.append(np.random.normal(0, 0.01, size=100))

    domain_embed4data[0] = np.array([0.0] * 100)
    domain_embed4data = np.array(domain_embed4data)
    print('domain emb shape: ', domain_embed4data.shape)
    return domain_embed4data

def generate_tencent_embedding(vocab, rebuild=False):

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build tencent emb')

    if rebuild:
        with open('/hdd/lujunyu/dataset/tencent_emb/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8') as f:
            tencent_vocab = []
            for line in f.readlines():
                row = line.strip().split(' ')
                if len(row[1:]) != 200:
                    continue
                tencent_vocab.append(row[0])
        with open('/home/lujunyu/repository/sentiment_ai_challenge/sequential_train_version/util/Tencent_Embedding_Word.txt', 'w', encoding='utf-8') as f:
            for w in tencent_vocab:
                f.write(w+'\n')
    else:
        with open('/home/lujunyu/repository/sentiment_ai_challenge/sequential_train_version/util/Tencent_Embedding_Word.txt', 'r', encoding='utf-8') as f:
            tencent_vocab = [x.strip().strip('\n') for x in f.readlines()]

    tencent_both = set(vocab) & set(tencent_vocab)

    with open('/hdd/lujunyu/dataset/tencent_emb/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8') as f:
        tencent_dic = {}
        for line in f.readlines():
            row = line.strip().split(' ')
            if len(row[1:]) != 200:
                continue
            if row[0] not in tencent_both:
                continue
            tencent_dic[row[0]] = np.array(row[1:])
            tencent_both = tencent_both - set(row[0])

    print(len(list(tencent_dic.keys())),
          ' word have pre-trained tencent embedding , total word:%d ' % len(vocab))

    tencent_embed4data = []
    tencent_both = set(tencent_dic.keys())
    for word in vocab:
        ###
        if word in tencent_both:
            tencent_embed4data.append(tencent_dic[word])
            tencent_both = tencent_both - set(word)
        else:
            tencent_embed4data.append(np.array([0.0] * 200))

    tencent_embed4data[0] = np.array([0.0] * 200)
    tencent_embed4data = np.array(tencent_embed4data)
    print('tencent emb shape: ', tencent_embed4data.shape)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish building tencent embedding')

    return tencent_embed4data

def generate_fasttext_embedding(vocab):

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build fasttext emb')

    with open('/hdd/lujunyu/dataset/fasttext/chinese/cc.pkl','rb') as f:
        fasttext_vocab, fasttext_dic, fasttext_wv = pkl.load(f)

    print(len(set(vocab) & set(fasttext_vocab)),
          ' word have pre-trained fasttext embedding , total word:%d ' % len(vocab))

    fasttext_both = set(vocab) & set(fasttext_vocab)

    fasttext_embed4data = []
    for word in vocab:
        ###
        if word in fasttext_both:
            fasttext_embed4data.append(fasttext_wv[fasttext_dic[word]])
            fasttext_both = fasttext_both - set(word)
        else:
            fasttext_embed4data.append(np.array([0.0] * 300))

    fasttext_embed4data[0] = np.array([0.0] * 300)
    fasttext_embed4data = np.array(fasttext_embed4data)
    print('fasttext emb shape: ', fasttext_embed4data.shape)
    return fasttext_embed4data


def get_label(train, dev, testa):
    ## attribute dic
    attribute_dic = {}
    for i in range(20):
        attribute_dic[train.iloc[:, 2 + i].name] = i

    train_label = np.array(train.iloc[:, 2:-1]) + 2
    dev_label = np.array(dev.iloc[:, 2:-1]) + 2
    testa_label = np.array(testa.iloc[:, 2:-1]) + 2

    return train_label, dev_label, testa_label, attribute_dic

if __name__ == '__main__':

    # dataframe
    train = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/train/train_cut.pkl')
    dev = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/val/val_cut.pkl')
    testa = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/test/testa_cut.pkl')
    # testb = pd.read_pickle('/hdd/lujunyu/dataset/meituan-sa/test/testb_cut.pkl')

    train_sent = [' '.join(x) for x in train['content']]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<UNK>',split=' ',lower=False,filters='')
    tokenizer.fit_on_texts(train_sent)

    vocab = ['<PAD>'] + list(tokenizer.word_index)
    print('Vocabulary size: ', len(vocab))

    train['content'] = train['content'].apply(padding_rev)
    dev['content'] = dev['content'].apply(padding_rev)
    testa['content'] = testa['content'].apply(padding_rev)
    # testb['content'] = testb['content'].apply(padding_rev)

    train['word_id'] = train['content'].apply(tokenizer.texts_to_sequences)
    dev['word_id'] = dev['content'].apply(tokenizer.texts_to_sequences)
    testa['word_id'] = testa['content'].apply(tokenizer.texts_to_sequences)
    # testb['word_id'] = testb['content'].apply(tokenizer.texts_to_sequences)

    train['word_id'] = train['word_id'].apply(pad_sequences, maxlen=max_sent_len, padding='post', truncating='pre')
    dev['word_id'] = dev['word_id'].apply(pad_sequences, maxlen=max_sent_len, padding='post', truncating='pre')
    testa['word_id'] = testa['word_id'].apply(pad_sequences,maxlen=max_sent_len,padding='post',truncating='pre')
    # testb['word_id'] = testb['word_id'].apply(pad_sequences, maxlen=max_sent_len, padding='post', truncating='pre')

    train_labels, dev_labels, testa_labels, attr_dic = get_label(train, dev, testa)

    with open(save_train_path, 'wb') as f:
        pkl.dump((train_labels, np.array(train['word_id'].to_list())), f, protocol=4)
    print("Save train data  in " + save_train_path + " successfully...")
    with open(save_dev_path, 'wb') as f:
        pkl.dump((dev_labels, np.array(dev['word_id'].to_list())), f, protocol=4)
    print("Save dev data  in " + save_dev_path + " successfully...")
    with open(save_testa_path, 'wb') as f:
        pkl.dump((testa_labels, np.array(testa['word_id'].to_list())), f, protocol=4)
    print("Save testa data  in " + save_testa_path + " successfully...")



    domain_emb = generate_domain_embedding(vocab)
    with open(save_domain_emb_path, 'wb') as f:
        pkl.dump((attr_dic, vocab, domain_emb), f, protocol=4)
    print("Save embedding data in " + save_domain_emb_path + " successfully...")

    # fasttext_emb = generate_fasttext_embedding(vocab)
    # with open(save_fasttext_emb_path, 'wb') as f:
    #     pkl.dump((attr_dic, vocab, fasttext_emb), f, protocol=4)
    # print("Save embedding data in " + save_fasttext_emb_path + " successfully...")
    #
    # tencent_emb = generate_tencent_embedding(vocab)
    # with open(save_tencent_emb_path, 'wb') as f:
    #     pkl.dump((attr_dic, vocab, tencent_emb), f, protocol=4)
    # print("Save embedding data in " + save_tencent_emb_path + " successfully...")

    print('finish...')
