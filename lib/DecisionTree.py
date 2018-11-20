# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     DecisionTree
   Description :    C4.5
   Author :        zhuguiliang
   date：          18/11/19
-------------------------------------------------
"""
import numpy as np
from collections import Counter


def entropy(labels: list) -> float:
    """
    ent = sum(-pi * log2(pi))
    :param labels:
    :return:
    """
    labelcount = Counter()
    labelcount.update(labels)
    num = len(labels)
    ent = 0.
    for k, v in labelcount.most_common():
        prob = float(v/num)
        ent -= prob * np.log2(prob)
    return ent


def chossebestfeature(features: list, labels: list) -> int:
    """

    :param features:
    :param labels:
    :return:
    """
    num_feature = len(features)
    base_entropy = entropy(labels)
    best_info_gain_ratio = 0.
    best_feature = -1
    for i in range(num_feature):
        feature_values = [example[i] for example in features]
        unique_values = set(feature_values)
        new_entropy = 0.
        split_info = 0.
        for value in unique_values:
            sub_data =

    return best_feature


class DecisionTree(object):
    def __init__(self):
        pass

    def split(self):
        pass

    def fit(self, x, y):
        self.labelcount = Counter()
        self.labelcount.update(y)

    def pred(self):
        pass