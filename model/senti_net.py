import tensorflow as tf
import model.layer as layers
import util.operation as op
from model.LOCAL_CONV import LOCAL_CONV

class Senti_Net(object):

    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config
        self.LOCAL_CONV = LOCAL_CONV(self.config)

    def build_graph(self):

        if self.config['rand_seed'] is not None:
            rand_seed = self.config['rand_seed']
            tf.set_random_seed(rand_seed)
            print('set tf random seed: %s' % self.config['rand_seed'])

        with self.graph.as_default():

            self.review, self.labels, self.is_training, self.domain_emb, self.tencent_emb = layers.attr_net_input(
                self.config)

            # convert table to variable
            domain_emb = tf.Variable(self.domain_emb, name='domain_emb', trainable=True)
            tencent_emb = tf.Variable(self.tencent_emb, name='tencent_emb', trainable=False)

            # review_embed : (batch, rev_len, sent_len, emb_dim)
            rev_domain = tf.nn.embedding_lookup(domain_emb, self.review)
            rev_tencent = tf.nn.embedding_lookup(tencent_emb, self.review)
            review_embed = tf.concat([rev_domain, rev_tencent], axis=-1)

            # rev_mask:(batch,)
            # sent_mask:(batch, rev)
            rev_len, sent_len = op.generate_mask(self.config, self.review)

            # sent_emb:(batch, rev, sent, emb)
            sent_emb = layers.stack_sent_sru(self.config, review_embed, sent_len)
            # attr_sent_emb:(batch, attr, rev, emb)
            senti_sent_emb = layers.stack_sent_attention(self.config, sent_emb, sent_len, is_training=self.is_training)

            # score:(batch, attr, 4)
            # att:(batch, attr, rev_len)
            senti_doc_emb, self.doc_att = layers.stack_doc_attention(self.config, senti_sent_emb, rev_len,
                                                                    is_training=self.is_training)
            # not_mention_emb:(batch, attr, emb)
            not_mention_emb = layers.not_mention_rep(self.config, senti_doc_emb, senti_sent_emb)

            # score:(batch, attr, 4)
            self.score = layers.predict_senti(self.config, senti_doc_emb, not_mention_emb, is_training=self.is_training)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.labels))
            self.senti_pred = tf.cast(tf.argmax(self.score, axis=-1), dtype=tf.float32)

            # Calculate cross-entropy loss
            tv = tf.trainable_variables()
            for v in tv:
                print(v)

            Optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            self.optimizer = Optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
            self.all_variables = tf.global_variables()
            self.grads_and_vars = Optimizer.compute_gradients(self.loss)

            for grad, var in self.grads_and_vars:
                if grad is None:
                    print(var)

            self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
            self.g_updates = Optimizer.apply_gradients(self.capped_gvs)

        return self.graph

