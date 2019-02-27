import os
import time

import tensorflow as tf
import numpy as np
import json

import util.evaluation as eva
from util.data_generator import DataGenerator
import util.operation as op


def train(conf, attr_model, senti_model):
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

    with tf.device('/cpu:0'):
        ## get attr net table
        attr_table, aspect_context = attr_sess.run([attr_graph.get_tensor_by_name('table:0'), attr_graph.get_tensor_by_name('attr_context:0')])
    ## init senti net
    senti_sess.run(senti_model.init, feed_dict={senti_model.table:attr_table, senti_model.LOCAL_CONV.attr_c:aspect_context})

    average_loss = 0.0
    step = 0
    best_result = 0

    for step_i in range(conf["num_scan_data"]):
        for batch_index in range(train_batch_num):
            train_review, train_attr_label, train_senti_label = dg.train_data_generator(batch_index)
            attr_feed = {
                attr_model.review: train_review,
                attr_model.attr_label: train_attr_label,
                attr_model.is_training: True
            }

            curr_att, curr_attr_pred = attr_sess.run([attr_model.doc_att, attr_model.attr_pred], feed_dict=attr_feed)

            senti_feed = {
                senti_model.review: train_review,
                senti_model.sent_att: curr_att,
                senti_model.attr_label:train_attr_label,
                senti_model.senti_label: train_senti_label,
                senti_model.is_training: True
            }
            _, curr_loss, curr_lr = senti_sess.run([senti_model.g_updates, senti_model.loss, senti_model.learning_rate], feed_dict=senti_feed)

            average_loss += curr_loss

            step += 1

            if step % conf["print_step"] == 0 and step > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                      " processed: [" + str(step * 1.0 / train_batch_num) +
                      "] loss: [" + str(average_loss / conf["print_step"]) + "]" +
                      "] lr: [" + str(curr_lr) + "]")
                average_loss = 0
                # op.senti_att_visualization(conf, curr_senti_att, train_review, train_label, curr_pred, dg)

            if step % conf["save_step"] == 0 and step > 0:
                index = step / conf['save_step']
                print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())), '  Save step: %s' % index)

                all_attr_pred = []
                all_senti_pred = []
                dev_senti_loss = 0
                # caculate dev score
                for batch_index in range(val_batch_num):
                    dev_review, dev_attr_label, dev_senti_label = dg.dev_data_generator(batch_index)

                    attr_feed = {
                        attr_model.review: dev_review,
                        attr_model.attr_label: dev_attr_label,
                        attr_model.is_training: False
                    }

                    dev_att, dev_attr_pred = attr_sess.run([attr_model.doc_att, attr_model.attr_pred], feed_dict=attr_feed)

                    senti_feed = {
                        senti_model.review: dev_review,
                        senti_model.sent_att: dev_att,
                        senti_model.attr_label:dev_attr_label,
                        senti_model.senti_label: dev_senti_label,
                        senti_model.is_training:False
                    }

                    senti_loss, senti_scores, senti_pred  = senti_sess.run(
                        [senti_model.loss, senti_model.senti_score, senti_model.senti_pred],
                        feed_dict=senti_feed
                    )
                    dev_senti_loss += senti_loss
                    all_attr_pred.append(dev_attr_pred)
                    all_senti_pred.append(senti_pred)

                all_attr_pred = np.concatenate(all_attr_pred, axis=0)
                all_senti_pred = np.concatenate(all_senti_pred,axis=0)
                # write evaluation result
                senti_f1, senti_class_f1 = eva.senti_evaluate(all_senti_pred, dg.dev_senti_label)
                dev_result_file_path = conf["save_path"] + "result." + str(index)
                result = {'senti F1 score': senti_f1}
                for i, att in enumerate(dg.attribute_dic):
                    assert dg.attribute_dic[att] == i
                    result[att] = senti_class_f1[i]
                with open(dev_result_file_path, 'w') as f:
                    f.write(json.dumps(result, indent=4))

                print('finish dev evaluation')
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print('val loss: ', dev_senti_loss / val_batch_num)

                if senti_f1 > best_result:
                    best_result = senti_f1
                    print('best senti result: ', senti_f1)
                    _save_path = senti_model.saver.save(senti_sess, conf["save_path"] + "model.ckpt." + str(step / conf["save_step"]))
                    print("succ saving model in " + _save_path)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

