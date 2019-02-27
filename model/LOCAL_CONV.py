# -*- coding: utf-8 -*-

import tensorflow as tf
import model.layer as layers


class LOCAL_CONV(object):
    def __init__(self, config):
        self.conf = config
        self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)

    def sentiment_scoring(self, sent_emb, sent_len, sent_att, is_training):
        """

        :param review_embed: (batch*rev, sent, emb)
        :param sent_len: (batch*rev)
        :param sent_att: 
        :return: 
        """

        # sent_emb:(batch*rev,sent,emb)
        sent_emb = tf.reshape(sent_emb, shape=[-1, self.conf['max_sent_len'], self.conf['emb_dim']])
        sent_len = tf.reshape(sent_len, shape=[-1, ])

        # sent_lstm:(batch*rev, sent, emb)
        sent_lstm = self.Bi_LSTM(sent_emb, sent_len, 'sent_lstm')
        # sent_conv:(batch*rev, sent, filter)
        sent_conv = self.Sent_Conv(sent_emb, sent_len, 'sent_conv')

        #
        attr_c = self.init_attribute()
        # self.local_att_mask:(batch*rev, attr, sent)
        self.local_att_mask, self.local_pos_mask = self.generate_att_mask(sent_lstm, sent_len, attr_c)

        # sent_conv :(batch*rev, attr, sent, channel)
        sent_conv = tf.einsum('bik,bai->baik', sent_conv, self.local_pos_mask)
        # sent_conv :(batch*rev, attr, channel)
        sent_conv = tf.reduce_max(sent_conv,axis=-2)
        # sent_lstm :(batch*rev, attr, emb)
        sent_lstm = tf.einsum('bik,bai->bak', sent_lstm, self.local_att_mask)

        # sent_conv:(batch, rev, attr, emb)
        sent_conv = tf.reshape(sent_conv, shape=[-1, self.conf['max_rev_len'], sent_conv.shape[-2], sent_conv.shape[-1]])
        # sent_conv:(batch, attr, rev, emb)
        sent_conv = tf.transpose(sent_conv, perm=[0, 2, 1, 3])
        # sent_lstm:(batch, rev, attr, emb)
        sent_lstm = tf.reshape(sent_lstm,shape=[-1, self.conf['max_rev_len'], sent_lstm.shape[-2], sent_lstm.shape[-1]])
        # sent_lstm:(batch, attr, rev, emb)
        sent_lstm = tf.transpose(sent_lstm, perm=[0, 2, 1, 3])


        # (batch, attr, emb)
        # use attr net attention weight
        sent_conv = tf.einsum('bark,bar->bak', sent_conv, sent_att)
        sent_lstm = tf.einsum('bark,bar->bak', sent_lstm, sent_att)


        # caculate not mention score
        # not_mention_score:(batch, attr, 1)
        not_mention_score = self.not_mention_scoring(sent_lstm)

        senti_score = tf.layers.dense(
            inputs=sent_conv,
            units=3,
            kernel_initializer=self.initializer,
            bias_initializer=tf.zeros_initializer,
            name='pred_senti',
            reuse=tf.AUTO_REUSE
            )
        senti_score = tf.concat([not_mention_score, senti_score], axis=-1)

        return senti_score

    def Bi_LSTM(self, sent_emb, sent_len, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # define parameters
            fw_cell = tf.contrib.rnn.SRUCell(
                self.conf['rnn_dim'] / 2
            )
            bw_cell = tf.contrib.rnn.SRUCell(
                self.conf['rnn_dim'] / 2
            )

            sent_emb, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=sent_emb,
                sequence_length=sent_len,
                dtype=tf.float32)

            # sent_emb:(batch*rev,sent,emb)
            sent_emb = tf.concat(sent_emb, axis=-1)

        return sent_emb

    def Sent_Conv(self, sent_emb, sent_len, name):
        """
        
        :param sent_emb: (batch*rev,sent,emb)
        :param sent_len: 
        :param name: 
        :return: 
        """

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            sent_emb_out = []
            for i, kernel in enumerate(self.conf['cnn_kernel_size']):
                sent_emb_out.append(tf.layers.conv1d(
                    inputs=sent_emb,
                    filters=self.conf['cnn_channel'],
                    kernel_size=kernel,
                    kernel_initializer=self.initializer,
                    bias_initializer=tf.zeros_initializer,
                    activation=tf.nn.relu,
                    padding='same',
                    name='conv_1d_'+str(i),
                    reuse=tf.AUTO_REUSE
                ))

            # sent_emb_out:(batch*rev, sent, emb*kernel_num)
            sent_emb_out = tf.concat(sent_emb_out, axis=-1)

            sent_len = tf.expand_dims(tf.sequence_mask(sent_len, maxlen=self.conf['max_sent_len'], dtype=tf.float32),
                                      axis=-1)

            sent_emb_out = sent_emb_out * sent_len

        return sent_emb_out

    def Sent_Multi_Head_Transformer(self, sent_emb, sent_len, is_training, name=None):
        """
        
        :param sent_emb: 
        :param sent_len: 
        :param name: 
        :return: 
        """
        sent_trans = sent_emb
        for i in range(self.conf['transformer_stack']):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                sent_trans = layers.multi_head_block(sent_trans, sent_trans, sent_trans, sent_len, sent_len,is_training=is_training)

        return sent_trans

    def Local_Sent_Multi_Head_Transformer(self, sent_emb, sent_attr_mask, is_training=False, name=None):
        """

        :param sent_emb: (batch*rev, sent, emb)
        :param sent_attr_mask: (batch*rev, attr, sent)
        :param name: 
        :return: 
        """

        sent_emb = tf.einsum('bik,bai->baik',sent_emb,sent_attr_mask)

        sent_attr_trans = []

        for sent_trans, sent_mask in zip(tf.unstack(sent_emb,axis=1), tf.unstack(sent_attr_mask,axis=1)):
            for i in range(self.conf['transformer_stack']):
                with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                    sent_trans = layers.local_multi_head_block(sent_trans, sent_trans, sent_trans, sent_mask, sent_mask,
                                                         is_training=is_training)
            sent_attr_trans.append(sent_trans)

        # sent_attr_trans:(batch*rev, attr, sent, emb)
        sent_attr_trans = tf.stack(sent_attr_trans, axis=1)
        # sent_attr_trans:(batch*rev, attr, emb)
        sent_attr_trans = tf.reduce_max(sent_attr_trans, axis=-2)

        return sent_attr_trans

    def init_attribute(self):
        self.attr_c = tf.placeholder(
            name='context',
            shape=[self.conf['attribute_num'], self.conf['emb_dim']],
            dtype=tf.float32
        )

        attr_c_v = tf.Variable(self.attr_c, name='attr_c_v', trainable=self.conf['attr_net_param_train'])

        return attr_c_v

    def generate_att_mask(self, sent, sent_len, context, mask_value=-2 ** 32 + 1):
        """

        :param sent: (batch*rev, sent, emb)
        :param sent_len: (batch*rev)
        :param context: (attr, emb)
        :return: 
        """

        att = tf.einsum('bik,ak->bai', sent, context)
        sent_mask = tf.sequence_mask(
            tf.tile(tf.expand_dims(sent_len, axis=-1), multiples=[1, self.conf['attribute_num']]),
            maxlen=self.conf['max_sent_len'],
            dtype=tf.float32
        )

        # att_id:(batch, attr, sent)
        att_id = tf.argmax(att, axis=-1)
        # lower_bound, upper_bound:(batch, attr)
        lower_bound = att_id - self.conf['local_window_size']
        lower_bound = tf.where(tf.less(lower_bound, 0), tf.zeros_like(lower_bound), lower_bound)
        lower_mask = 1.0 - tf.sequence_mask(lower_bound, maxlen=self.conf['max_sent_len'], dtype=tf.float32)
        upper_bound = att_id + self.conf['local_window_size']
        upper_mask = tf.sequence_mask(upper_bound, maxlen=self.conf['max_sent_len'], dtype=tf.float32)
        mask = lower_mask * upper_mask * sent_mask

        return tf.nn.softmax(att * mask + mask_value * (1-mask),axis=-1), mask

    def local_attention(self, sent, sent_len, mask_value=-2 ** 32 + 1):
        """

        :param sent: (batch*rev, attr, sent, emb)
        :param sent_len: (batch*rev)
        :return: 
        """

        with tf.variable_scope('local_attention', reuse=tf.AUTO_REUSE):
            local_attr_context = tf.get_variable(
                name='local_attr_context',
                shape=[self.conf['attribute_num'], self.conf['sentiment_num'], self.conf['emb_dim']],
                dtype=tf.float32,
                initializer=self.initializer
            )

            self.local_att = tf.einsum('baik,ajk->baij', sent, local_attr_context)
            # att_mask:(batch*rev, attr, sent, 1)
            att_mask = tf.expand_dims(tf.sequence_mask(
                tf.tile(tf.expand_dims(sent_len, axis=1), multiples=[1, self.conf['attribute_num']]),
                maxlen=self.conf['max_sent_len'],
                dtype=tf.float32
            ), axis=-1)
            self.local_att = tf.nn.softmax(self.local_att * att_mask + mask_value * (1 - att_mask), axis=-2)

            sent_at_attr = tf.einsum('baij,baik->bajk', self.local_att, sent)

            local_attr_context_expand = tf.tile(tf.expand_dims(local_attr_context, axis=0),
                                                multiples=[self.conf['batch_size'] * self.conf['max_rev_len'], 1, 1, 1])

            sent_rep = tf.concat([local_attr_context_expand, sent_at_attr, local_attr_context_expand - sent_at_attr,
                                  local_attr_context_expand * sent_at_attr], axis=-1)

            return sent_rep

    def not_mention_scoring(self, sent):
        """
        
        :param sent: (batch, attr, emb)
        :return: 
        """

        not_mention_context = tf.get_variable(
            name='not_mention_context',
            shape=[self.conf['attribute_num'], self.conf['emb_dim']],
            dtype=tf.float32,
            initializer=self.initializer
        )

        not_mention_score = tf.einsum('bak,ak->ba', sent, not_mention_context)

        return tf.expand_dims(not_mention_score, axis=-1)






