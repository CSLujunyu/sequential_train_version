import numpy as np
import pickle
import os
import time
import util.operation as op

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs
        self.train_review, self.train_labels = self.load_train_data()
        self.train_data_size = len(self.train_review)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Training Data')

        self.dev_review, self.dev_labels = self.load_dev_data()
        self.dev_data_size = len(self.dev_review)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Dev Data')

        self.testa_review, self.testa_labels = self.load_testa_data()
        self.testa_data_size = len(self.testa_review)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Test Data')

        self.attribute_dic, self.vocab, self.domain_emb, self.tencent_emb = self.load_table()

    def train_data_generator(self,batch_num):

        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size

        # shuffle data at the beginning of every epoch
        if batch_num == 0:
            self.train_review, self.train_labels, _ = self.unison_shuffled(self.train_review, self.train_labels)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Shuffling Data:')

        if start < end:
            batches_review = self.train_review[start:end]
            batches_labels = self.train_labels[start:end]
        else:
            batches_review = self.train_review[train_size - self.configs['batch_size']:train_size]
            batches_labels = self.train_labels[train_size - self.configs['batch_size']:train_size]

        return batches_review, batches_labels

    def dev_data_generator(self, batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch.
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """

        dev_size = self.dev_data_size
        start = batch_num * self.configs['batch_size'] % dev_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % dev_size
        if start < end:
            batches_review = self.dev_review[start:end]
            batches_labels = self.dev_labels[start:end]
        else:
            batches_review = self.dev_review[start:]
            batches_labels = self.dev_labels[start:]

        return batches_review, batches_labels

    def testa_data_generator(self, batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch.
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """

        testa_size = self.testa_data_size
        start = batch_num * self.configs['batch_size'] % testa_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % testa_size
        if start < end:
            batches_review = self.testa_review[start:end]
            batches_labels = self.testa_labels[start:end]
        else:
            batches_review = self.testa_review[start:]
            batches_labels = self.testa_labels[start:]

        return batches_review, batches_labels


    def table_generator(self):

        if self.configs['word_emb_init'] is not None:
            with open(self.configs['word_emb_init'], 'rb') as f:
                self._word_embedding_init = pickle.load(f, encoding='latin1')
        else:
            self._word_embedding_init = np.random.random(size=[self.configs['vocab_size'], self.configs['emb_size']])

        return self._word_embedding_init


    def unison_shuffled(self, a, b):
        np.random.seed(self.configs['rand_seed'])
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], p

    def shuffle_response(self, candidate_id, response, response_len, label):
        """
        responses contain ground truth id
        :param response: (batch_size, options_num, max_turn_len)
        :param response_len: (batch_size, options_num)
        :param label: (batch_size, options_num)
        :return:
        """
        candidate_id, response, response_len, label = list(candidate_id), list(response), list(response_len), list(label)
        for i in range(len(response)):
            candidate_id[i],response[i], response_len[i], label[i], _ = self.unison_shuffled_copies(
                candidate_id[i],response[i],response_len[i],label[i])

        return candidate_id, response, response_len, label


    def generate_bert_sent_emb(self, revs):

        def id2w(id):
            for i in id:
                if i != self.configs['<PAD>']:
                    yield self.vocab[i]

        def sent2emb(rev):
            return self.bc.encode(rev)

        # batch_rev_emb = []
        # for rev in revs:
        #     rev_str = []
        #     for sent in rev:
        #         tmp = ''.join(list(id2w(sent)))
        #         if tmp:
        #             rev_str.append(tmp)
        #         else:
        #             rev_str.append('。')
        #     batch_rev_emb.append(sent2emb(rev_str))

        rev_str = []
        for rev in revs:
            for sent in rev:
                tmp = ''.join(list(id2w(sent)))
                if tmp:
                    rev_str.append(tmp)
                else:
                    rev_str.append('。')

        batch_rev_emb = sent2emb(rev_str)
        batch_rev_emb = np.reshape(batch_rev_emb, newshape=[-1, self.configs['max_rev_len'], self.configs['bert_dim']])

        return batch_rev_emb

    def load_train_data(self):

        assert os.path.exists(self.configs['train_data_path']) and os.path.getsize(self.configs['train_data_path']) > 0

        with open(self.configs['train_data_path'], 'rb') as f:
            train_labels, train_review = pickle.load(f)

        return train_review, train_labels

    def load_dev_data(self):

        assert os.path.exists(self.configs['dev_data_path']) and os.path.getsize(self.configs['dev_data_path']) > 0

        with open(self.configs['dev_data_path'], 'rb') as f:
            dev_labels, dev_review = pickle.load(f)

        return dev_review, dev_labels

    def load_testa_data(self):

        assert os.path.exists(self.configs['testa_data_path']) and os.path.getsize(self.configs['testa_data_path']) > 0

        with open(self.configs['testa_data_path'], 'rb') as f:
            testa_labels, testa_review= pickle.load(f)

        return testa_review, testa_labels

    def load_table(self):

        with open(self.configs['domain_emb_path'], 'rb') as f:
            _, domain_vocab, domain_emb = pickle.load(f)

        # with open(self.configs['fasttext_emb_path'], 'rb') as f:
        #     attribute_dic, vocab, fasttext_emb = pickle.load(f)

        with open(self.configs['tencent_emb_path'], 'rb') as f:
            attribute_dic, tencent_vocab, tencent_emb = pickle.load(f)

        assert domain_vocab == tencent_vocab

        return attribute_dic, tencent_vocab, domain_emb ,tencent_emb


