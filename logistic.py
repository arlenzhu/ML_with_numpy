# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logistic
   Description :
   Author :        zhuguiliang
   date：          18/11/19
-------------------------------------------------
"""
from sklearn.datasets import load_breast_cancer
from lib.logistic_prue_numpy import LogisticClassifiter
from sklearn.model_selection import train_test_split
from lib.utils import precision, recall, f1

lr = LogisticClassifiter(epochs=10000, lr=0.0001)
x, y = load_breast_cancer(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

lr.fit(train_x.T, train_y, versob=1)
pred = lr.predict(test_x.T)
print('precision:', precision(test_y, pred))
print('recall:', recall(test_y, pred))
print('f1 score:', f1(test_y, pred))