import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
import json

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
        logits = op.dot_sim(Q, K)  # [batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K)

    if is_mask:
        mask = op.mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
        logits = mask * logits + (1 - mask) * mask_value

    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        print('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)


def FFN(x, out_dimension_0=None, out_dimension_1=None):
    '''Add two dense connected layer, max(0, x*W0+b0)*W1+b1.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Returns:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    with tf.variable_scope('FFN_1', reuse=tf.AUTO_REUSE):
        y = op.dense(x, out_dimension_0)
        y = tf.nn.relu(y)
    with tf.variable_scope('FFN_2', reuse=tf.AUTO_REUSE):
        z = op.dense(y, out_dimension_1)  # , add_bias=False)  #!!!!
    return z


def block(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_layer_norm=True,
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
            w = op.layer_norm_debug(y + z)
    else:
        w = y + z
    # w.shape = (batch_size, max_turn_len, emb_size)
    return w

def doc_att_visualization(config, doc_att, rev, label, pred, dg):
    ind = np.random.randint(0, config['batch_size'])
    rev, label, doc_att, pred = rev[ind], label[ind], doc_att[ind], pred[ind]
    label = np.argmax(np.reshape(label, newshape=[-1, 4]), axis=-1)
    for i, attr in enumerate(dg.attribute_dic):
        max_att_ind = np.argmax(doc_att[i], axis=-1)
        rev_word = id2w(rev[max_att_ind], config['<PAD>'], dg.vocab)
        print(attr + '\t' +
              'label:' + str(label[i]) + '\t' +
              'pred:' + str(pred[i]) + '\t' +
              'att:' + str(doc_att[i][max_att_ind]) + '\t' +
              ' '.join(list(rev_word)))

def doc_sent_att_visualization(config, doc_att, sent_att, rev, label, pred, dg):
    ind = np.random.randint(0, config['batch_size'])
    rev, label, doc_att, pred = rev[ind], label[ind], doc_att[ind], pred[ind]
    label = np.argmax(np.reshape(label, newshape=[-1, 4]), axis=-1)
    for i, attr in enumerate(dg.attribute_dic):
        max_doc_att_ind = np.argmax(doc_att[i], axis=-1)

        _sent_att = sent_att[ind*config['max_rev_len']+max_doc_att_ind]
        max_sent_att_ind = np.argmax(_sent_att[i])

        rev_word = list(id2w(rev[max_doc_att_ind], config['<PAD>'], dg.vocab))
        print(attr + '\t' +
              'label:' + str(label[i]) + '\t' +
              'pred:' + str(pred[i]) + '\t' +
              'doc_att:' + str(doc_att[i][max_doc_att_ind]) + '\t' +
              'sent_att:' + str(_sent_att[i][max_sent_att_ind]) + '\t' +
              'sent_att_word:' + rev_word[max_sent_att_ind] +'\t' +
              'review:' + ' '.join(rev_word))

def id2w(id, pad_id, vocab):
    for i in id:
        if i != pad_id:
            yield vocab[i]


def generate_mask(config, review):

    # sent_mask : (batch, rev_len)
    sent_mask = tf.reduce_sum(
        tf.where(
            condition=tf.equal(review, tf.ones_like(review) * config['<PAD>']),
            x=tf.zeros_like(review),
            y=tf.ones_like(review)
        ),
        axis=-1
    )
    # review_mask : (batch)
    review_mask = tf.reduce_sum(
        tf.where(
            condition=tf.equal(sent_mask, tf.zeros_like(sent_mask)),
            x=tf.zeros_like(sent_mask),
            y=tf.ones_like(sent_mask)
        ),
        axis=-1
    )

    return review_mask, sent_mask

def generate_mask_for_bert(review):

    # rev_mask : (batch)
    review = review[:,:,0]
    rev_mask = tf.cast(tf.reduce_sum(
        tf.where(
            condition=tf.equal(review, tf.expand_dims(review[:,-1],axis=-1)),
            x=tf.zeros_like(review),
            y=tf.ones_like(review)
        ),
        axis=-1
    ),dtype=tf.int32)

    return rev_mask

def generate_bert_mask(config, review):

    # review_mask : (batch)
    review_mask = tf.reduce_sum(
        tf.where(
            condition=tf.equal(review, tf.ones_like(review) * config['<PAD>']),
            x=tf.zeros_like(review),
            y=tf.ones_like(review)
        ),
        axis=-1
    )

    return review_mask

def mask(row_lengths, col_lengths, max_row_length, max_col_length):
    '''Return a mask tensor representing the first N positions of each row and each column.

    Args:
        row_lengths: a tensor with shape [batch]
        col_lengths: a tensor with shape [batch]

    Returns:
        a mask tensor with shape [batch, max_row_length, max_col_length]

    Raises:
    '''
    row_mask = tf.sequence_mask(row_lengths, max_row_length) #bool, [batch, max_row_len]
    col_mask = tf.sequence_mask(col_lengths, max_col_length) #bool, [batch, max_col_len]

    row_mask = tf.cast(tf.expand_dims(row_mask, -1), tf.float32)
    col_mask = tf.cast(tf.expand_dims(col_mask, -1), tf.float32)

    return tf.einsum('bik,bjk->bij', row_mask, col_mask)

def weighted_sum(weight, values):
    '''Calcualte the weighted sum.

    Args:
        weight: a tensor with shape [batch, time, dimension]
        values: a tensor with shape [batch, dimension, values_dimension]

    Return:
        a tensor with shape [batch, time, values_dimension]

    Raises:
    '''
    return tf.einsum('bij,bjk->bik', weight, values)


def bilinear_sim(x, y, is_nor=True):
    '''calculate bilinear similarity with two tensor.
    Args:
        x: a tensor with shape [batch, time_x, dimension_x]
        y: a tensor with shape [batch, time_y, dimension_y]

    Returns:
        a tensor with shape [batch, time_x, time_y]
    Raises:
        ValueError: if
            the shapes of x and y are not match;
            bilinear matrix reuse error.
    '''
    M = tf.get_variable(
        name="bilinear_matrix",
        shape=[x.shape[-1], y.shape[-1]],
        dtype=tf.float32,
        initializer=tf.orthogonal_initializer())
    sim = tf.einsum('bik,kl,bjl->bij', x, M, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[-1] * y.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return sim


def dot_sim(x, y, is_nor=True):
    """

    :param x: shape = (batch)
    :param y:
    :param is_nor:
    :return:
    """
    assert x.shape[-1] == y.shape[-1]

    sim = tf.einsum('bik,bjk->bij', x, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return sim


def layer_norm(x, axis=None, epsilon=1e-6):
    '''Add layer normalization.

    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    print('wrong version of layer_norm')
    scale = tf.get_variable(
        name='scale',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    if axis is None:
        axis = [-1]

    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axis, keepdims=True)
    norm = (x - mean) * tf.rsqrt(variance + epsilon)
    return scale * norm + bias


def layer_norm_debug(x, axis=None, epsilon=1e-6):
    '''Add layer normalization.

    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    if axis is None:
        axis = [-1]
    shape = [x.shape[i] for i in axis]

    scale = tf.get_variable(
        name='scale',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axis, keepdims=True)
    norm = (x - mean) * tf.rsqrt(variance + epsilon)
    return scale * norm + bias


def dense(x, out_dimension=None, add_bias=True):
    '''Add dense connected layer, Wx + b.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Return:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    if out_dimension is None:
        out_dimension = x.shape[-1]

    W = tf.get_variable(
        name='weights',
        shape=[x.shape[-1], out_dimension],
        dtype=tf.float32,
        initializer=tf.orthogonal_initializer())
    if add_bias:
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        return tf.einsum('bik,kj->bij', x, W) + bias
    else:
        return tf.einsum('bik,kj->bij', x, W)

def doc_att_visualization(config, doc_att, rev, label, pred, dg, batch_id):
    """
    
    :param config: 
    :param doc_att: (batch, attr, rev_len)
    :param rev: (batch, rev_len, sent_len)
    :param label: (batch, attr)
    :param pred: (batch, attr)
    :param dg: 
    :return: 
    """

    label_index = ['Not mention', 'Mention']
    ind = np.random.randint(0, config['batch_size'])
    rev, label, doc_att, pred = rev[ind], label[ind], doc_att[ind], pred[ind]

    attr_label = np.where(pred==0, np.zeros_like(label), np.ones_like(label))

    sample = {
        'example-id': str(batch_id * config['batch_size'] +ind),
    }

    display_id = label + pred
    for i, attr in enumerate(dg.attribute_dic):
        if display_id[i] == 0:
            continue
        max_att_ind = np.argmax(doc_att[i], axis=-1)
        # if pred[i] == 0:
        #     rev_word = ''
        # else:
        #     rev_word = id2w(rev[max_att_ind], config['<PAD>'], dg.vocab)
        rev_word = id2w(rev[max_att_ind], config['<PAD>'], dg.vocab)

        # print(attr + '\t' +
        #       'label:' + str(label[i]) + '\t' +
        #       'pred:' + str(pred[i]) + '\t' +
        #       'att:' + str(doc_att[i][max_att_ind]) + '\t' +
        #       ' '.join(list(rev_word)))

        attr_content = {
            'label':label_index[int(attr_label[i])],
            'pred':label_index[int(pred[i])],
            'max att':str(doc_att[i][max_att_ind]),
            'key sentence':' '.join(list(rev_word))
        }
        sample[attr] = attr_content
    return sample

def doc_sent_att_visualization(config, doc_att, sent_att, rev, label, pred, dg):
    """
    
    :param config: 
    :param doc_att: (batch, attr, rev_len)
    :param sent_att: 
    :param rev: 
    :param label: 
    :param pred: 
    :param dg: 
    :return: 
    """
    ind = np.random.randint(0, config['batch_size'])
    rev, label, doc_att, pred = rev[ind], label[ind], doc_att[ind], pred[ind]
    label = np.argmax(np.reshape(label, newshape=[-1, 4]), axis=-1)
    for i, attr in enumerate(dg.attribute_dic):
        max_doc_att_ind = np.argmax(doc_att[i], axis=-1)

        _sent_att = sent_att[ind*config['max_rev_len']+max_doc_att_ind]
        max_sent_att_ind = np.argmax(_sent_att[i])

        rev_word = list(id2w(rev[max_doc_att_ind], config['<PAD>'], dg.vocab))
        print(attr + '\t' +
              'label:' + str(label[i]) + '\t' +
              'pred:' + str(pred[i]) + '\t' +
              'doc_att:' + str(doc_att[i][max_doc_att_ind]) + '\t' +
              'sent_att:' + str(_sent_att[i][max_sent_att_ind]) + '\t' +
              'sent_att_word:' + rev_word[max_sent_att_ind] +'\t' +
              'review:' + ' '.join(rev_word))


def attr_confusion_matrix(pred, label, dg):

    cm = np.sum(np.einsum('bi,bj->bij',pred,label), axis=0)
    Label = list(dg.attribute_dic.keys())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(Label)
    ax.set_yticklabels(Label)
    im = ax.imshow(cm, cmap=plt.cm.hot_r)
    plt.show()
    plt.savefig('./img.jpg')

    return cm

def generate_final_pred_label(attr_pred, attr_label, senti_pred, senti_label):

    # pred,label:(dev_data_size,attr_num,4)
    pred = convert_to_one_hot_label(attr_pred * (senti_pred + 1), 4)
    label = np.argmax(senti_label, axis=-1)
    label = convert_to_one_hot_label(attr_label * (label + 1), 4)
    return pred, label

def senti_confusion_matrix(pred, label, dg, save_path):
    """
    
    :param attr_pred: 
    :param senti_pred: 
    :param dg: 
    :return: 
    """

    Label = ['Not_mention', 'Negative', 'Neutral', 'Positive']

    fig, axn = plt.subplots(4, 5, figsize=(40, 25))

    for row_id, ax_list in enumerate(axn):
        for col_id, ax in enumerate(ax_list):
            attr_id = row_id*5+col_id
            _pred = pred[:,attr_id,:]
            _label = label[:,attr_id,:]

            cm = np.sum(np.einsum('bi,bj->bij',_pred,_label), axis=0)
            ax.set_title(list(dg.attribute_dic.keys())[attr_id])
            ax.set_ylabel('label')
            ax.set_xlabel('prediction')
            g = sns.heatmap(cm,
                            ax=ax,
                            cmap="Blues",
                            square=True,
                            annot=True,
                            cbar=False,
                            xticklabels=Label,
                            yticklabels=Label,
                            linewidths=1.7)
    plt.show()
    plt.savefig(save_path + 'senti_confusion_matrix.png')

    return

def attr_context_cos_similarity(attr_c, dg, save_path):

    dot_p = np.einsum('ak,bk->ab',attr_c,attr_c)

    norm = np.sqrt(np.sum(attr_c * attr_c, axis=-1))
    norm = np.einsum('a,b->ab',norm,norm)

    cos_sim = dot_p / norm

    f, ax = plt.subplots(figsize=(20, 20))
    g = sns.heatmap(cos_sim,
                    ax=ax,
                    cmap="Blues",
                    square=True,
                    annot=True,
                    cbar=False,
                    xticklabels=np.array(list(dg.attribute_dic.keys())),
                    yticklabels=np.array(list(dg.attribute_dic.keys())),
                    linewidths=1.7)
    plt.savefig(save_path)
    plt.close('all')

    return

def convert_to_one_hot_label(y, C):
    y = np.int32(y)
    return np.eye(C)[y.reshape(-1)].reshape([-1,20,4])

def visualize_attr_bad_case(config, rev, pred, label, sent_att, attr_id, dg):
    """
    
    :param config: 
    :param rev: 
    :param pred: 
    :param label: 
    :param sent_att: (dev_data_size, attr, max_rev_len)
    :param attr_id: 
    :param dg: 
    :return: 
    """


    label, pred, sent_att = label[:,attr_id,:], pred[:,attr_id,:], sent_att[:,attr_id,:]

    sample = []

    sentiment_dic = ['Not_mention', 'Negative', 'Neutral', 'Positive']

    for example_id in range(label.shape[0]):

        example_lable, example_pred, example_sent_att, example_rev = sentiment_dic[np.argmax(label[example_id],axis=-1)], \
                                                sentiment_dic[np.argmax(pred[example_id],axis=-1)], \
                                                sent_att[example_id],\
                                                rev[example_id]

        if example_lable != example_pred:
            rev_word = {}
            for i, att in enumerate(example_sent_att):
                if att != 0:
                    rev_word[str(att)] = ' '.join(list(id2w(example_rev[i], config['<PAD>'], dg.vocab)))

            sample.append({
                'example-id': str(example_id),
                'attribute':list(dg.attribute_dic.keys())[attr_id],
                'pred': example_pred,
                'label': example_lable,
                'rev': rev_word
            })

    with open(config['save_path'] + list(dg.attribute_dic.keys())[attr_id] +'_bad_case.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(sample, indent=4, ensure_ascii=False))

    return

def visualize_local_att(config, rev, pred, label, sent_att, local_att, dg):
    """
    
    :param config: 
    :param rev: 
    :param pred: 
    :param label: 
    :param sent_att: 
    :param local_att: (batch*rev, attr, sent)
    :param dg: 
    :param num_display: 
    :return: 
    """

    sentiment_dic = ['Not_mention', 'Negative', 'Neutral', 'Positive']
    local_att = np.reshape(local_att, newshape=[-1,config['max_rev_len'],config['attribute_num'],config['max_sent_len']])
    local_att = np.float32(local_att!=0)

    display_sample = []
    for example_id in range(rev.shape[0]):

        sample = {
            'example-id': str(example_id),
        }

        for i, attr in enumerate(dg.attribute_dic):

            senti_label = sentiment_dic[np.argmax(label[example_id, i], axis=-1)]
            pred_label = sentiment_dic[np.argmax(pred[example_id, i], axis=-1)]

            if senti_label == pred_label:
                continue

            max_att_ind = np.argmax(sent_att[example_id,i], axis=-1)

            key_sent = id2w(rev[example_id,max_att_ind], config['<PAD>'], dg.vocab)
            local_word_id = rev[example_id,max_att_ind] * local_att[example_id,max_att_ind,i]
            local_word_id = local_word_id.astype(int)
            local_key_word = id2w(local_word_id,0, dg.vocab)

            attr_content = {
                'label': senti_label,
                'pred': pred_label,
                'max att': str(sent_att[example_id,i,max_att_ind]),
                'key sentence': ' '.join(list(key_sent)),
                'local key word': ' '.join(list(local_key_word))
            }
            sample[attr] = attr_content

        display_sample.append(sample)

    with open(config['save_path'] + 'local_attention_visualization.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(display_sample, indent=4, ensure_ascii=False))

    return
