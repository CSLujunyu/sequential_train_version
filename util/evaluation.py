import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import util.operation as op

def attr_evaluate(attr_pred, attr_label, attribute_dic):
    """
    
    :param attr_pred: ( ,attr)
    :param senti_pred: ( ,attr,3)
    :param attr_label: 
    :param senti_label: 
    :return: 
    """

    attr_label = np.where(attr_label == 0, np.zeros_like(attr_label), np.ones_like(attr_label))

    f1_score_dict = dict()
    precision_score_dict = dict()
    recall_score_dict = dict()
    for i, attr in enumerate(list(attribute_dic.keys())):
        assert attribute_dic[attr] == i
        f1_score_dict[attr] = f1_score(attr_label[:,i], attr_pred[:,i],average="binary")
        precision_score_dict[attr] = precision_score(attr_label[:,i], attr_pred[:,i],average="binary")
        recall_score_dict[attr] = recall_score(attr_label[:,i], attr_pred[:,i], average="binary")

    f1_score_mean = np.mean(list(f1_score_dict.values()))
    precision_score_mean = np.mean(list(precision_score_dict.values()))
    recall_score_mean = np.mean(list(recall_score_dict.values()))
    print("macro_f1_mean:%s" % f1_score_mean)
    print("macro_precision_mean:%s" % precision_score_mean)
    print("macro_recall_mean:%s" % recall_score_mean)

    return f1_score_mean, f1_score_dict


def senti_evaluate(senti_pred, senti_label, attribute_dic):
    """
    :param senti_pred: ( , attr)
    :param senti_label: ( , attr)
    :return: 
    """

    f1_score_dict = dict()
    precision_score_dict = dict()
    recall_score_dict = dict()
    for i, attr in enumerate(list(attribute_dic.keys())):
        assert attribute_dic[attr] == i
        f1_score_dict[attr] = f1_score(y_true=senti_label[:, i], y_pred=senti_pred[:, i], average="macro")
        precision_score_dict[attr] = precision_score(y_true=senti_label[:, i], y_pred=senti_pred[:, i], average="macro")
        recall_score_dict[attr] = recall_score(y_true=senti_label[:, i], y_pred=senti_pred[:, i], average="macro")

    f1_score_mean = np.mean(list(f1_score_dict.values()))
    precision_score_mean = np.mean(list(precision_score_dict.values()))
    recall_score_mean = np.mean(list(recall_score_dict.values()))
    print("macro_f1_mean:%s" % f1_score_mean)
    print("macro_precision_mean:%s" % precision_score_mean)
    print("macro_recall_mean:%s" % recall_score_mean)

    return f1_score_mean, f1_score_dict

def attr_f1(pred, label, epsilon = 1e-10):

    TP = np.sum(pred * label, axis=0)
    FP = np.sum(pred * (1-label), axis=0)
    FN = np.sum((1-pred) * label, axis=0)

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = np.mean(2 * precision * recall / (precision + recall + epsilon))

    return f1

def senti_f1(pred, label, epsilon = 1e-10):

    # TP:(4,)
    TP = np.sum(pred * label, axis=0)
    FP = np.sum(pred * (1-label), axis=0)
    FN = np.sum((1-pred) * label, axis=0)

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = np.mean(2 * precision * recall / (precision + recall + epsilon))

    return f1


def softmax(x, axis):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x, axis=axis)
    exp_x = np.exp(x, axis=axis)
    softmax_x = exp_x / np.sum(exp_x, axis=axis)
    return softmax_x

