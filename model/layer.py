import tensorflow as tf
import util.operation as op

def attr_net_input(_conf):
    # define placehloders
    review = tf.placeholder(
        tf.int32,
        shape=[_conf['batch_size'], _conf['max_rev_len'], _conf['max_sent_len']],
        name='review'
    )

    labels = tf.placeholder(
        tf.int32,
        shape=[_conf['batch_size'], _conf['attribute_num']],
        name='labels'
    )

    is_training = tf.placeholder(
        tf.bool,
        shape=[],
        name='is_training'
    )

    domain_emb = tf.placeholder(
        tf.float32,
        shape=[_conf['vocab_size'], _conf['domain_emb_dim']],
    )
    tencent_emb = tf.placeholder(
        tf.float32,
        shape=[_conf['vocab_size'], _conf['tencent_emb_dim']],
    )

    return review, labels, is_training, domain_emb, tencent_emb

def senti_net_input(_conf):
    # define placehloders
    review = tf.placeholder(
        tf.int32,
        shape=[_conf['batch_size'], _conf['max_rev_len'], _conf['max_sent_len']],
        name='review'
    )

    sent_att = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['max_rev_len']],
        name='sent_att'
    )

    attr_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num']],
        name='attr_label'
    )

    senti_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['sentiment_num']],
        name='senti_label'
    )

    is_training = tf.placeholder(
        tf.bool,
        shape=[])

    table = tf.placeholder(
        tf.float32,
        shape=[_conf['vocab_size'], _conf['emb_dim']],
    )

    return review, sent_att, attr_label, senti_label, is_training, table

def senti_net_bert_input(_conf):
    # define placehloders
    review = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['max_rev_len'], _conf['bert_dim']])

    sent_att = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['max_rev_len']])

    attr_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num']])

    senti_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['sentiment_num']])

    is_training = tf.placeholder(
        tf.bool,
        shape=[])

    return review, sent_att, attr_label, senti_label, is_training

def sent_lstm(config, review_embed, sent_mask, name):

    # sent_emb:(batch*rev,sent,emb)
    sent_emb = tf.reshape(review_embed, shape=[-1, config['max_sent_len'], config['emb_dim']])
    # sent_emb:(batch*rev,)
    sent_len = tf.reshape(sent_mask, shape=[-1, ])

    with tf.variable_scope(name+'_sent', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )
        bw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=sent_emb,
            sequence_length=sent_len,
            dtype=tf.float32)

        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.concat([tf.reduce_mean(outputs, axis=1), tf.reduce_max(outputs, axis=1)], axis=-1)

        outputs = tf.layers.dense(
            outputs,
            units=config['emb_dim'],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                              uniform=True),
            bias_initializer=tf.zeros_initializer,
            name='dense',
            reuse=tf.AUTO_REUSE
        )

        # outputs:(batch, rev_len, emb_dim)
        outputs = tf.reshape(outputs, shape=[-1, config['max_rev_len'], config['emb_dim']])

    return outputs

def sent_sru(config, review_embed, sent_mask, is_training=False):

    def get_rnn_cell():
        return tf.contrib.rnn.SRUCell(config['rnn_dim'] / 2)

    # sent_emb:(batch*rev,sent,emb)
    sent_emb = tf.reshape(review_embed, shape=[-1, review_embed.shape[-2], review_embed.shape[-1]])
    # sent_emb:(batch*rev,)
    sent_len = tf.reshape(sent_mask, shape=[-1,])

    # define parameters
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(config['rnn_layers'])])
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(config['rnn_layers'])])

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=sent_emb,
        sequence_length=sent_len,
        dtype=tf.float32)
    # outputs:(batch, rev_len, sent_len, emb_dim)
    outputs = tf.concat(outputs, axis=-1)
    outputs = tf.concat([tf.reduce_mean(outputs, axis=1), tf.reduce_max(outputs, axis=1)], axis=-1)

    outputs = tf.layers.dense(
        outputs,
        units=config['emb_dim'],
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        bias_initializer=tf.zeros_initializer,
        name='dense',
        reuse=tf.AUTO_REUSE
    )

    # outputs:(batch, rev_len, emb_dim)
    outputs = tf.reshape(outputs, shape=[-1, config['max_rev_len'], outputs.shape[-1]])

    return outputs

def stack_sent_sru(config, review_embed, sent_mask, is_training=False):

    def get_rnn_cell():
        return tf.contrib.rnn.SRUCell(config['rnn_dim'] / 2)

    # sent_emb:(batch*rev,sent,emb)
    sent_emb = tf.reshape(review_embed, shape=[-1, review_embed.shape[-2], review_embed.shape[-1]])
    # sent_emb:(batch*rev,)
    sent_len = tf.reshape(sent_mask, shape=[-1,])

    # define parameters
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(config['rnn_layers'])])
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(config['rnn_layers'])])

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=sent_emb,
        sequence_length=sent_len,
        dtype=tf.float32)
    # outputs:(batch, rev_len, sent_len, emb_dim)
    outputs = tf.concat(outputs, axis=-1)

    outputs = tf.reshape(outputs, shape=[-1, config['max_rev_len'],outputs.shape[-2], outputs.shape[-1]])

    if config['batch_normalization']:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)

    return outputs

def stack_sent_attention(config, sent_emb, sent_len, is_training=False, mask_value=-2 ** 32 + 1):

    ### init sent_attr_context
    with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
        attr_context = tf.get_variable(
            name='attr_context',
            shape=[config['attribute_num'],config['attribute_prototype'], config['emb_dim']],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer
        )

    # sent_emb:(batch*rev,sent,emb)
    sent_emb = tf.reshape(sent_emb, shape=[-1, sent_emb.shape[-2], sent_emb.shape[-1]])
    # sent_emb:(batch*rev,)
    sent_len = tf.reshape(sent_len, shape=[-1,])

    sent_att = []
    for i in range(config['sent_attention_layers']):
        with tf.variable_scope('sent_attention_'+str(i), reuse=tf.AUTO_REUSE):
            # define parameters
            fw_cell = tf.contrib.rnn.SRUCell(
                config['rnn_dim'] / 2
            )
            bw_cell = tf.contrib.rnn.SRUCell(
                config['rnn_dim'] / 2
            )

            sent_emb, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=sent_emb,
                sequence_length=sent_len,
                dtype=tf.float32)
            # sent_emb:(batch, rev_len, sent_len, emb_dim)
            sent_emb = tf.concat(sent_emb, axis=-1)

            # att: (batch*rev, attr, sent)
            att = tf.einsum('aik,bjk->baij',attr_context, sent_emb)
            # sent_mask.shape = (batch*rev, 1, sent, 1)
            sent_mask = tf.sequence_mask(sent_len, maxlen=sent_emb.shape[-2], dtype=tf.float32)
            attr_mask = tf.ones(shape=[config['attribute_num'],config['attribute_prototype']])
            # mask: (batch*rev, attr, sent)
            mask = tf.einsum('ai,bj->baij',attr_mask,sent_mask)
            att = tf.nn.softmax(att*mask + mask_value * (1-mask), axis=-1)
            # sent_att: list of (batch*rev, attr, emb)
            sent_att.append(tf.reduce_max(tf.einsum('baij,bjk->baik',att,sent_emb),axis=-2))

    # sent_att:(batch*rev, attr, emb)
    sent_att = tf.reduce_sum(tf.stack(sent_att, axis=-1), axis=-1)
    if config['batch_normalization']:
        sent_att = tf.layers.batch_normalization(sent_att, training=is_training)

    # sent_att:(batch, attr, rev, emb)
    sent_att = tf.transpose(tf.reshape(sent_att, [-1,config['max_rev_len'],sent_att.shape[-2],sent_att.shape[-1]]), perm=[0,2,1,3])

    return sent_att

def stack_doc_attention(config, rev, rev_len, is_training, mask_value=-2 ** 32 + 1):
    """
    
    :param config: 
    :param rev: (batch, attr, rev, emb)
    :param rev_len: (batch)
    :param mask_value: 
    :return: 
    """

    ### init sent_attr_context
    with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
        attr_context = tf.get_variable(
            name='attr_context',
            shape=[config['attribute_num'], config['attribute_prototype'], config['emb_dim']],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer
        )

    # rev:(batch*attr, rev, emb)
    rev = tf.reshape(rev, shape=[-1, rev.shape[-2], rev.shape[-1]])
    # rev_len_tile:(batch*attr, )
    rev_len_tile = tf.reshape(tf.tile(tf.expand_dims(rev_len, axis=-1),multiples=[1,config['attribute_num']]), shape=[-1, ])

    rev_att = []
    for i in range(config['doc_attention_layers']):
        with tf.variable_scope('doc_attention_' + str(i), reuse=tf.AUTO_REUSE):
            # define parameters
            fw_cell = tf.contrib.rnn.SRUCell(
                config['rnn_dim'] / 2
            )
            bw_cell = tf.contrib.rnn.SRUCell(
                config['rnn_dim'] / 2
            )

            rev, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=rev,
                sequence_length=rev_len_tile,
                dtype=tf.float32)
            # rev:(batch*attr, rev, emb_dim)
            rev = tf.concat(rev, axis=-1)

            # rev:(batch, attr, rev, emb_dim)
            rev_reshape = tf.reshape(rev, shape=[-1, config['attribute_num'], rev.shape[-2], rev.shape[-1]])

            # att.shape = (batch, attr, attr_type, rev)
            att = tf.einsum('aik,bajk->baij', attr_context, rev_reshape)
            # rev_mask.shape = (batch, 1, 1, rev)
            rev_mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(rev_len, maxlen=rev.shape[-2], dtype=tf.float32),axis=1), axis=1)
            att = tf.nn.softmax(att * rev_mask + mask_value * (1 - rev_mask), axis=-1)
            # sent_att: list of (batch, attr, emb)
            rev_att.append(tf.reduce_max(tf.einsum('baij,bajk->baik', att, rev_reshape), axis=-2))

    # rev_att:(batch, attr, emb*layers)
    rev_att = tf.concat(rev_att, axis=-1)
    if config['batch_normalization']:
        rev_att = tf.layers.batch_normalization(rev_att, training=is_training)
    # rev_att:(batch, attr, emb)
    rev_att = tf.layers.dense(
        rev_att,
        units=config['emb_dim'],
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        bias_initializer=tf.zeros_initializer,
        name='doc_att_dense',
        reuse=tf.AUTO_REUSE
    )


    return rev_att, att


def doc_attention(config, rev, rev_len, is_training, mask_value=-2 ** 32 + 1):
    """

    :param config: 
    :param rev: (batch, rev, emb)
    :param rev_len: (batch)
    :param mask_value: 
    :return: 
    """

    ### init sent_attr_context
    with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
        attr_context = tf.get_variable(
            name='attr_context',
            shape=[config['attribute_num'], config['emb_dim']],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer
        )


    # att.shape = (batch, attr, rev)
    att = tf.einsum('ak,bik->bai', attr_context, rev)
    # rev_mask.shape = (batch, 1, rev)
    rev_mask = tf.expand_dims(tf.sequence_mask(rev_len, maxlen=rev.shape[-2], dtype=tf.float32), axis=1)
    att = tf.nn.softmax(att * rev_mask + mask_value * (1 - rev_mask), axis=-1)
    # sent_att: list of (batch, attr, emb)
    rev_att = tf.einsum('bai,bik->bak', att, rev)

    if config['batch_normalization']:
        rev_att = tf.layers.batch_normalization(rev_att, training=is_training)

    return rev_att, att

def not_mention_rep(config, attr_doc_emb, attr_sent_emb, is_training=False):

    # attr_sent_emb = tf.reduce_mean(attr_sent_emb, axis=-2)
    rep = tf.layers.dense(
        tf.concat([attr_doc_emb - attr_sent_emb, attr_doc_emb * attr_sent_emb], axis=-1),
        units=config['emb_dim'],
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        bias_initializer=tf.zeros_initializer,
        name='not_mention_rep',
        reuse=tf.AUTO_REUSE
    )
    if config['batch_normalization']:
        rep = tf.layers.batch_normalization(rep, training=is_training)

    rep = tf.nn.relu(rep)

    if config['drop_out']:
        rep = tf.layers.dropout(rep, training=is_training)

    return rep

# def predict(config, attr_doc_emb, not_mention_emb):
#
#     attr_score = []
#     not_mention_score = []
#     for i in range(config['attribute_num']):
#         attr_score.append(tf.layers.dense(
#             inputs=attr_doc_emb[:, i],
#             units=1,
#             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
#                                                                               uniform=True),
#             bias_initializer=tf.zeros_initializer,
#             name='pred_attr_' + str(i),
#             reuse=tf.AUTO_REUSE
#         ))
#         not_mention_score.append(tf.layers.dense(
#             not_mention_emb[:, i],
#             units=1,
#             name='pred_not',
#             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
#                                                                               uniform=True),
#             bias_initializer=tf.zeros_initializer,
#             reuse=tf.AUTO_REUSE
#         ))
#     attr_score = tf.stack(attr_score, axis=1)
#     not_mention_score = tf.stack(not_mention_score, axis=1)
#     score = tf.concat([not_mention_score, attr_score], axis=-1)
#
#     return score

def predict_attr(config, attr_doc_emb, not_mention_emb, is_training=False):

    attr_score = []

    for i in range(config['attribute_num']):

        attr_pred_score = tf.layers.dense(
            inputs=attr_doc_emb[:,i],
            units=1,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                              uniform=True),
            bias_initializer=tf.zeros_initializer,
            name='pred_attr_'+str(i),
            reuse=tf.AUTO_REUSE
        )
        attr_score.append(attr_pred_score)

    attr_score = tf.stack(attr_score, axis=1)

    not_mention_emb = tf.layers.dense(
        not_mention_emb,
        units=not_mention_emb.shape[-1]//2,
        name='pred_not_1',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        bias_initializer=tf.zeros_initializer,
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE
    )

    not_mention_score = tf.layers.dense(
        not_mention_emb,
        units=1,
        name='pred_not_2',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        use_bias=False,
        reuse=tf.AUTO_REUSE
    )

    score = tf.concat([not_mention_score, attr_score], axis=-1)

    return score

def predict_senti(config, attr_doc_emb, not_mention_emb, is_training=False):

    attr_doc_emb = tf.layers.dense(
        inputs=attr_doc_emb,
        units=attr_doc_emb.shape[-1]//2,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        bias_initializer=tf.zeros_initializer,
        activation=tf.nn.relu,
        name='pred_senti_1',
        reuse=tf.AUTO_REUSE
    )

    attr_score = tf.layers.dense(
        inputs=attr_doc_emb,
        units=3,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        use_bias=False,
        name='pred_senti_2',
        reuse=tf.AUTO_REUSE
    )

    not_mention_emb = tf.layers.dense(
        not_mention_emb,
        units=not_mention_emb.shape[-1]//2,
        name='pred_not_1',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        bias_initializer=tf.zeros_initializer,
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE
    )

    not_mention_score = tf.layers.dense(
        not_mention_emb,
        units=1,
        name='pred_not_2',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True),
        use_bias=False,
        reuse=tf.AUTO_REUSE
    )

    score = tf.concat([not_mention_score, attr_score], axis=-1)

    return score


def doc_lstm(config, rev, rev_len, name):

    with tf.variable_scope(name+'_doc', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )
        bw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=rev,
            sequence_length=rev_len,
            dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)

    return outputs

def doc_sru(config, rev, rev_len, name):

    with tf.variable_scope(name+'_doc', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.contrib.rnn.SRUCell(
            config['rnn_dim'] / 2
        )
        bw_cell = tf.contrib.rnn.SRUCell(
            config['rnn_dim'] / 2
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=rev,
            sequence_length=rev_len,
            dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)

    return outputs


def attention(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add attention layer.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, Q_time, V_dimension]

    Raises:
        AssertionError: if
            Q_dimension not equal to K_dimension when attention type is dot.
    '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K) / tf.sqrt(1546.0)  # [batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K) / tf.sqrt(1546.0)

    if is_mask:
        mask = op.mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
        logits = mask * logits + (1 - mask) * mask_value

    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        print('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)

def local_attention(
        Q, K, V,
        Q_mask, K_mask,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add attention layer.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, Q_time, V_dimension]

    Raises:
        AssertionError: if
            Q_dimension not equal to K_dimension when attention type is dot.
    '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K) / tf.sqrt(1546.0)  # [batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K) / tf.sqrt(1546.0)

    if is_mask:
        mask = tf.einsum('bi,bj->bij',Q_mask,K_mask)
        logits = mask * logits + (1 - mask) * mask_value

    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        print('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)


def FFN(x, out_dimension_0=64, out_dimension_1=300):
    '''Add two dense connected layer, max(0, x*W0+b0)*W1+b1.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Returns:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    y = tf.layers.dense(
        x,
        units=x.shape[-1],
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True),
        use_bias=False,
        activation=tf.nn.relu,
        name='FFN_1',
        reuse=tf.AUTO_REUSE
    )
    # with tf.variable_scope('FFN_2', reuse=tf.AUTO_REUSE):
    #     z = op.dense(y, out_dimension_1)  # , add_bias=False)  #!!!!

    # z = tf.layers.dense(
    #     y,
    #     units=out_dimension_1,
    #     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True),
    #     use_bias=False,
    #     activation=tf.nn.relu,
    #     name='FFN_2',
    #     reuse=tf.AUTO_REUSE
    # )

    return y


def block(
        Q, K, V,
        Q_lengths, K_lengths,is_training=False,is_layer_norm=True,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''



    # att.shape = (batch_size, max_turn_len, emb_size)
    att = attention(Q, K, V,
                    Q_lengths, K_lengths,
                    attention_type='dot',
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob)
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm', reuse=tf.AUTO_REUSE):
            y = op.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = FFN(y)

    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm', reuse=tf.AUTO_REUSE):
            # w = tf.layers.batch_normalization(y + z, training=is_training)
            w = op.layer_norm_debug(y + z)
    else:
        w = y + z
    # w.shape = (batch_size, max_turn_len, emb_size)
    return w


def local_block(
        Q, K, V,
        Q_mask, K_mask,is_training=False,is_layer_norm=True,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''



    # att.shape = (batch_size, max_turn_len, emb_size)
    att = local_attention(Q, K, V,
                    Q_mask, K_mask,
                    attention_type='dot',
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob)
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm', reuse=tf.AUTO_REUSE):
            y = op.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = FFN(y)

    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm', reuse=tf.AUTO_REUSE):
            # w = tf.layers.batch_normalization(y + z, training=is_training)
            w = op.layer_norm_debug(y + z)
    else:
        w = y + z
    # w.shape = (batch_size, max_turn_len, emb_size)
    return w


def multi_head_block(
        Q, K, V,
        Q_lengths, K_lengths, is_training, is_layer_norm=True, multi_head=6,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''

    O = []

    for i in range(multi_head):
        with tf.variable_scope('head_'+str(i), reuse=tf.AUTO_REUSE):
            W_q = tf.get_variable(
                name='W_q',
                shape=[Q.shape[-1],50],
                dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            )
            W_k = tf.get_variable(
                name='W_k',
                shape=[K.shape[-1], 50],
                dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            )
            W_v = tf.get_variable(
                name='W_v',
                shape=[V.shape[-1], 50],
                dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            )

            Q_w = tf.einsum('bik,km->bim',Q, W_q)
            K_w = tf.einsum('bik,km->bim', K, W_k)
            V_w = tf.einsum('bik,km->bim', V, W_v)

        with tf.variable_scope('self-att', reuse=tf.AUTO_REUSE):
            O.append(block(Q_w,K_w,V_w,Q_lengths,K_lengths))

    O =  tf.concat(O, axis=-1)

    W_o = tf.get_variable(
        name='W_o',
        shape=[O.shape[-1], 300],
        dtype=tf.float32,
        initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
    )

    return tf.einsum('bik,km->bim',O,W_o)

def local_multi_head_block(
        Q, K, V,
        Q_mask, K_mask, is_training, is_layer_norm=True, multi_head=6,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''

    O = []

    for i in range(multi_head):
        with tf.variable_scope('head_'+str(i), reuse=tf.AUTO_REUSE):
            W_q = tf.get_variable(
                name='W_q',
                shape=[Q.shape[-1],75],
                dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            )
            W_k = tf.get_variable(
                name='W_k',
                shape=[K.shape[-1], 75],
                dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            )
            W_v = tf.get_variable(
                name='W_v',
                shape=[V.shape[-1], 75],
                dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            )

            Q_w = tf.einsum('bik,km->bim',Q, W_q)
            K_w = tf.einsum('bik,km->bim', K, W_k)
            V_w = tf.einsum('bik,km->bim', V, W_v)

        with tf.variable_scope('self-att', reuse=tf.AUTO_REUSE):
            O.append(local_block(Q_w,K_w,V_w,Q_mask,K_mask))

    O =  tf.concat(O, axis=-1)

    W_o = tf.get_variable(
        name='W_o',
        shape=[O.shape[-1], 300],
        dtype=tf.float32,
        initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
    )

    return tf.einsum('bik,km->bim',O,W_o)