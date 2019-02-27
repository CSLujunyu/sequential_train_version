import tensorflow as tf
import model.layer as layers
import util.operation as op


class Attr_Net(object):
    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config

    def build_graph(self):

        if self.config['rand_seed'] is not None:
            rand_seed = self.config['rand_seed']
            tf.set_random_seed(rand_seed)
            print('set tf random seed: %s' % self.config['rand_seed'])

        with self.graph.as_default():

            self.review, self.labels, self.is_training, self.table = layers.attr_net_input(self.config)

            # convert table to variable
            table_v = tf.Variable(self.table, name='table')

            # review_embed : (batch, rev_len, sent_len, emb_dim)
            review_embed = tf.nn.embedding_lookup(table_v, self.review)

            # rev_mask:(batch,)
            # sent_mask:(batch, rev)
            rev_mask, sent_mask = op.generate_mask(self.config, self.review)

            #
            sent_emb = layers.sent_sru(self.config, review_embed, sent_mask, name='attr')

            attr_label = tf.cast(
                tf.where(tf.equal(self.labels, 0), tf.zeros_like(self.labels), tf.ones_like(self.labels)), tf.int32)

            # score:(batch, attr_num, 2)
            # att:(batch, attr, rev_len)
            self.score, self.doc_att = self.attr_esim_match(sent_emb, rev_mask, 'attr_doc')
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=attr_label))
            self.attr_pred = tf.cast(tf.argmax(self.score, axis=-1), dtype=tf.float32)

            # Calculate cross-entropy loss
            tv = tf.trainable_variables()
            for v in tv:
                print(v)

            self.global_step = tf.Variable(0, trainable=False)
            initial_learning_rate = self.config['learning_rate']
            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate,
                global_step=self.global_step,
                decay_steps=300,
                decay_rate=0.9,
                staircase=True)

            Optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = Optimizer.minimize(
                self.loss,
                global_step=self.global_step)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
            self.all_variables = tf.global_variables()
            self.grads_and_vars = Optimizer.compute_gradients(self.loss)

            for grad, var in self.grads_and_vars:
                if grad is None:
                    print(var)

            self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
            self.g_updates = Optimizer.apply_gradients(
                self.capped_gvs,
                global_step=self.global_step)

        return self.graph

    def attr_esim_match(self, rev, rev_len, name):

        doc_emb = layers.doc_sru(self.config, rev, rev_len, 'attr')

        # attr_esim_emb:(batch, attr, emb)
        score = self.ESIM(doc_emb, rev_len)

        return score

    def ESIM(self, R_s, R_len, mask_value=-2 ** 32 + 1):
        """

        :param R_s: (batch, rev, emb)
        :param R_len: (batch)
        :param mask_value:
        :return:
        """

        C_s = tf.get_variable(
            name='attr_context',
            shape=[self.config['attribute_num'], self.config['rnn_dim']],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        )
        not_mention = tf.get_variable(
            name='not_mention',
            shape=[self.config['attribute_num'], self.config['rnn_dim']],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        )

        # attr_context:(batch,attr,emb)
        C_s = tf.tile(tf.expand_dims(C_s, axis=0), multiples=[self.config['batch_size'], 1, 1])
        # attr_context_len:(batch)
        C_len = tf.ones(shape=[self.config['batch_size']]) * self.config['attribute_num']
        # not mention:(batch, attr,emb)
        not_mention = tf.tile(tf.expand_dims(not_mention, axis=0), multiples=[self.config['batch_size'], 1, 1])

        att = tf.einsum('bak,bik->bai', C_s, R_s)
        mask = op.mask(C_len, R_len, C_s.shape[-2], R_s.shape[-2])
        # att:(batch, attr, rev_len)
        att = tf.nn.softmax(att * mask + (1.0 - mask) * mask_value, axis=-1)

        C_d = tf.einsum('bai,bik->bak', att, R_s)

        attr_feature = tf.concat([C_s, C_d, C_s - C_d, C_s * C_d], axis=-1)
        not_mention_feature = tf.concat([not_mention, C_d, not_mention - C_d, not_mention * C_d], axis=-1)

        attr_score = []
        not_mention_score = []
        for i in range(self.config['attribute_num']):
            attr_score.append(tf.layers.dense(
                inputs=attr_feature[:, i],
                units=1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                  uniform=True),
                bias_initializer=tf.zeros_initializer,
                name='pred_' + str(i),
                reuse=tf.AUTO_REUSE
            ))
            not_mention_score.append(tf.layers.dense(
                not_mention_feature[:, i],
                units=1,
                name='pred_',
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                  uniform=True),
                bias_initializer=tf.zeros_initializer,
                reuse=tf.AUTO_REUSE
            ))
        attr_score = tf.stack(attr_score, axis=1)
        not_mention_score = tf.stack(not_mention_score, axis=1)

        score = tf.concat([not_mention_score, attr_score], axis=-1)

        return score, att

