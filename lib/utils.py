# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :        zhuguiliang
   date：          18/11/19
-------------------------------------------------
"""
import numpy as np


def precision(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true*y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives/ (predicted_positives + 0.00001)
    return precision


def recall(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true*y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 0.00001)
    return recall


def f1(y_true, y_pred, beta=1):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    score = (1 + bb) * (p*r)/(bb*p + r + 0.00001)
    return score