import pandas as pd
import re
import io
import jieba
from string import whitespace
import numpy as np
import array
import tensorflow as tf
import gensim
import pickle as pkl
from langconv import *

pd.options.mode.chained_assignment = None
jieba.load_userdict('/home/lujunyu/repository/sentiment_ai_challenge/sequential_train_version/util/THUOCL_food_userdict.txt')

FLAGS = {
    'renew': True,
    'min_frequency': 0,
    'max_rev_len': 30,
    'max_sent_len': 300
}

punctuation = '“”-—"\r#￥%&*@（）【】~\…～()：、\u3000『  』'
outtab = ' ' * len(punctuation)
intab = punctuation
trantab = str.maketrans(intab, outtab)


def tradition2simple(line):
    # 将繁体转换成简体
    line = Converter('zh-hans').convert(line)
    return line


def save_all_cut_res(train, val, testa, testb):
    train_sent = [' '.join(x) for x in train]
    val_sent = [' '.join(x) for x in val]
    testa_sent = [' '.join(x) for x in testa]
    testb_sent = [' '.join(x) for x in testb]
    all_sent = train_sent + val_sent + testa_sent + testb_sent
    print(len(all_sent))
    with open('/hdd/lujunyu/dataset/meituan-sa/all_cut_sent.txt', 'w', encoding='utf-8') as f:
        for x in all_sent:
            f.write(x + '\n')
    with open('/hdd/lujunyu/dataset/meituan-sa/train_cut_sent.txt', 'w', encoding='utf-8') as f:
        for x in all_sent:
            f.write(x + '\n')
    return train_sent


def zng(paragraph):
    for sent in re.findall(u'[^!?。\.\!\?\？\；]+[!?。\.\!\?\？\；]?', paragraph, flags=re.U):
        for x in sent.split('\n'):
            yield x

def replace_error(text):
    return text.strip('').strip(' ').replace("\x06", "").replace("\x05", "").replace("\x07", "")

def replace_number(word):
    try:
        word = float(word)
        if word <= 5:
            return '<NUM1>'
        elif word >5 and word <10 :
            return '<NUM2>'
        elif word >10 and word <50 :
            return '<NUM3>'
        elif word >50 and word <100 :
            return '<NUM4>'
        elif word >100 and word <200 :
            return '<NUM5>'
        elif word >200 and word <300 :
            return '<NUM6>'
        elif word >300 and word <400 :
            return '<NUM7>'
        elif word >400 and word <500 :
            return '<NUM8>'
        elif word >500 and word <1000 :
            return '<NUM9>'
        elif word >1000 and word <2000 :
            return '<NUM10>'
        elif word >2000 and word <10000 :
            return '<NUM11>'
        else:
            return '<NUM12>'
    except:
        return word

def cut_text(raw_text):
    a = list(zng(raw_text))
    # return a is a list of string
    rev = []
    for x in a:
        w_count = 0
        sent = []
        for xx in list(jieba.cut(tradition2simple(x))):
            if w_count >= FLAGS['max_sent_len']:
                rev.append(' '.join(sent))
                w_count = 0
                sent = []
                continue
            if re.findall(u'[^!?。^☆‿\rO∩✿◡#＃ಥ$★+๑ ˙∀❥･ ᷄ὢ ᷅￥%&*@（）/≧≦〈〉《》【】∇=:~\…～()：、\u3000『』_＿\.\!\?，？~～,"“”：；#\'、()▽（）！…【】-]+[!?。\.\!\?\n]?', xx):
                sent.append(replace_number(replace_error(xx)))
                w_count += 1
        if len(sent) >= 1:
            if len(sent) <= 3:
                if len(rev) > 0:
                    rev[-1] = ' '.join([rev[-1], ' '.join(sent)])
                else:
                    rev.append(' '.join(sent))
            else:
                rev.append(' '.join(sent))
        if len(rev) > FLAGS['max_rev_len']:
            tmp = ' '.join(rev[:len(rev) - FLAGS['max_rev_len'] + 1])
            rev = [tmp] + rev[len(rev) - FLAGS['max_rev_len'] + 1:]
    return rev



if __name__ == '__main__':

    ## Load raw data
    raw_train = pd.read_csv('/hdd/lujunyu/dataset/meituan-sa/train/sentiment_analysis_trainingset.csv')
    raw_dev = pd.read_csv('/hdd/lujunyu/dataset/meituan-sa/val/sentiment_analysis_validationset.csv')
    raw_testa = pd.read_csv('/hdd/lujunyu/dataset/meituan-sa/test/ai_challenger_sentiment_analysis_testa_label.csv')
    raw_testb = pd.read_csv('/hdd/lujunyu/dataset/meituan-sa/test/sentiment_analysis_testb.csv')

    raw_train['content'] = raw_train['content'].apply(cut_text)
    raw_dev['content'] = raw_dev['content'].apply(cut_text)
    raw_testa['content'] = raw_testa['content'].apply(cut_text)
    raw_testb['content'] = raw_testb['content'].apply(cut_text)

    raw_train.to_pickle('/hdd/lujunyu/dataset/meituan-sa/train/train__cut.pkl')
    raw_dev.to_pickle('/hdd/lujunyu/dataset/meituan-sa/val/val_cut.pkl')
    raw_testa.to_pickle('/hdd/lujunyu/dataset/meituan-sa/test/testa_cut.pkl')
    raw_testb.to_pickle('/hdd/lujunyu/dataset/meituan-sa/test/testb_cut.pkl')

    print('Finish preparing cutting data...')