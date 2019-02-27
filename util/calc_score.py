#!/user/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import argparse
import pandas as pd
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

test_type_a = "testA"
test_type_b = "testB"


def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def save_sentiment_analysis_score(filename, context, score):
    f = open(filename, "a+")
    f.write("%s\t%s\n" % (context, str(score)))
    f.close()


def check_test_data(test_label_data, test_predict_data, iterm):

    test_content_predict_list = test_predict_data[iterm].tolist()
    test_content_validate_list = test_label_data[iterm].tolist()
    is_column_match = True
    is_content_count_equal = False
    count = 0
    try:
        for index, content in enumerate(test_content_validate_list):
            if content == test_content_predict_list[index]:
                count += 1
        columns_predict = test_predict_data.columns.values.tolist()
        columns_validate = test_label_data.columns.values.tolist()

        for index, column in enumerate(columns_validate):
            if column != columns_predict[index]:
                is_column_match = False

        if count == len(test_content_validate_list):
            is_content_count_equal = True
        is_legal = is_column_match and is_content_count_equal

    except:
        is_legal = False

    return is_legal


def calc_f1_score(test_label_data, test_predict_data):

    columns = test_label_data.columns.values.tolist()

    f1_score_dict = dict()
    precision_score_dict = dict()
    recall_score_dict = dict()
    for column in columns[2:]:
        f1_score_dict[column] = f1_score(test_label_data[column], test_predict_data[column], average="macro")
        logger.info("%s f1 score: %s" % (column, f1_score_dict[column]))
        precision_score_dict[column] = precision_score(test_label_data[column], test_predict_data[column], average="macro")
        recall_score_dict[column] = recall_score(test_label_data[column], test_predict_data[column], average="macro")
    f1_score_mean = np.mean(list(f1_score_dict.values()))
    precision_score_mean = np.mean(list(precision_score_dict.values()))
    recall_score_mean = np.mean(list(recall_score_dict.values()))
    columns_aspect = columns[2:]

    score_df = pd.DataFrame(columns=['aspect', 'macro_f1', 'macro_precision', 'macro_recall'])
    score_df["aspect"] = columns_aspect
    score_df["macro_f1"] = list(f1_score_dict.values())
    score_df["macro_precision"] = list(precision_score_dict.values())
    score_df["macro_recall"] = list(recall_score_dict.values())
    score_df.to_csv(os.path.abspath('..') + "/data/f1_precision_recall_2.csv", header=True, index=False, encoding="utf_8_sig")
    print("macro_f1_mean:%s" % f1_score_mean)
    print("macro_precision_mean:%s" % precision_score_mean)
    print("macro_recall_mean:%s" % recall_score_mean)
    return f1_score_mean


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pf', '--test_predict_data_file_path', type=str, nargs='?',
                        help='the test data path')
    parser.add_argument('-vf', '--test_validate_data_file_path', type=str, nargs='?',
                        help='the validate data path')

    parser.add_argument('-t', '--test_type', type=str, nargs='?',
                        choices=[test_type_a, test_type_b],
                        help='the test data set type')

    args = parser.parse_args()
    test_predict_data_path = args.test_predict_data_file_path

    logger.info(test_predict_data_path)
    test_type = args.test_type
    if not test_type:
        test_type = test_type_a

    test_label_data_path = args.test_validate_data_file_path
    if not test_label_data_path:
        if test_type == test_type_a:
            test_label_data_path = os.path.abspath('..') + "/data/ai_challenger_sentiment_analysis_testa_label.csv"

    score_result_file_path = os.path.abspath('..') + "/data/ai_challenger_1_sentiment_analysis_score_result.txt"

    test_label_data = load_data_from_csv(test_label_data_path)
    test_predict_data = load_data_from_csv(test_predict_data_path)

    if check_test_data(test_label_data, test_predict_data, "id"):

        f1_score_value = calc_f1_score(test_label_data, test_predict_data)
        logger.info("f1_score:%s" % f1_score_value)

        save_sentiment_analysis_score(score_result_file_path, test_predict_data_path, f1_score_value)
    else:
        print("提交数据不合法，无法计算分数")
