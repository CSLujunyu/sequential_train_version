import os
import time

import tensorflow as tf
import numpy as np
import json

import util.evaluation as eva
from util.data_generator import DataGenerator
import util.operation as op


def test(conf, attr_model, senti_model):
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # config display
    print('configurations: %s' % conf)

    # Data Generate
    dg = DataGenerator(conf)
    print('Train data size: ', dg.train_data_size)
    print('Dev data size: ', dg.dev_data_size)
    print('Test data size: ', dg.testa_data_size)

    # refine conf
    train_batch_num = int(dg.train_data_size / conf["batch_size"])
    val_batch_num = int(dg.dev_data_size / conf["batch_size"])

    conf["train_steps"] = conf["num_scan_data"] * train_batch_num
    conf["save_step"] = int(max(1., train_batch_num / 10))
    conf["print_step"] = int(max(1., train_batch_num / 50))

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build attribute graph')
    attr_graph = attr_model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build attribute graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build senti graph')
    senti_graph = senti_model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build senti graph sucess')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    attr_sess = tf.Session(graph=attr_graph, config=config)
    senti_sess = tf.Session(graph=senti_graph, config=config)
    ## init pre-training attr net
    if conf["attr_init_model"]:
        model_path = tf.train.latest_checkpoint(conf["attr_init_model"])
        attr_model.saver.restore(attr_sess, model_path)
        print("sucess init attr model %s" % model_path)

    if conf['senti_init_model']:
        model_path = tf.train.latest_checkpoint(conf["senti_init_model"])
        senti_model.saver.restore(senti_sess, model_path)
        print("sucess init senti model %s" % model_path)

    all_attr_pred = []
    all_senti_pred = []
    all_sent_att = []
    all_local_att_mask = []
    dev_senti_loss = 0
    # caculate dev score
    for batch_index in range(val_batch_num):
        dev_review, dev_attr_label, dev_senti_label = dg.dev_data_generator(batch_index)

        attr_feed = {
            attr_model.review: dev_review,
            attr_model.attr_label: dev_attr_label,
            attr_model.is_training: True
        }

        dev_att, dev_attr_pred = attr_sess.run([attr_model.doc_att, attr_model.attr_pred], feed_dict=attr_feed)

        senti_feed = {
            senti_model.review: dev_review,
            senti_model.sent_att: dev_att,
            senti_model.attr_label:dev_attr_label,
            senti_model.senti_label: dev_senti_label,
            senti_model.is_training:False
        }

        senti_loss, senti_scores, senti_pred, local_att_mask  = senti_sess.run(
            [senti_model.loss, senti_model.senti_score, senti_model.senti_pred, senti_model.LOCAL_CONV.local_att_mask],
            feed_dict=senti_feed
        )
        dev_senti_loss += senti_loss
        all_attr_pred.append(dev_attr_pred)
        all_senti_pred.append(senti_pred)
        all_sent_att.append(dev_att)
        all_local_att_mask.append(np.reshape(local_att_mask, newshape=[-1,conf['max_rev_len'],local_att_mask.shape[-2],local_att_mask.shape[-1]]))

    all_attr_pred = np.concatenate(all_attr_pred, axis=0)
    all_senti_pred = np.concatenate(all_senti_pred,axis=0)
    all_sent_att = np.concatenate(all_sent_att,axis=0)
    all_local_att_mask = np.concatenate(all_local_att_mask,axis=0)


    # visualize confusion matrix
    op.senti_confusion_matrix(all_senti_pred, dg.dev_senti_label, dg, conf['save_path'])

    # visualize bad case
    # for attr_id in range(conf['attribute_num']):
    #     op.visualize_attr_bad_case(conf,dg.dev_review,all_senti_pred,dg.dev_senti_label,all_sent_att,attr_id,dg)

    # visualize local attention
    op.visualize_local_att(conf,dg.dev_review,all_senti_pred,dg.dev_senti_label,all_sent_att,all_local_att_mask,dg)

    # write evaluation result
    senti_f1, senti_class_f1 = eva.senti_evaluate(all_senti_pred, dg.dev_senti_label)
    dev_result_file_path = conf["save_path"] + "test_result"
    result = {'senti F1 score': senti_f1}
    for i, att in enumerate(dg.attribute_dic):
        assert dg.attribute_dic[att] == i
        result[att] = senti_class_f1[i]
    with open(dev_result_file_path, 'w') as f:
        f.write(json.dumps(result, indent=4))

    print('finish dev evaluation')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('val loss: ', dev_senti_loss / val_batch_num)

    print('best senti result: ', senti_f1)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

