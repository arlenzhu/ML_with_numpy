# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logistic_prue_numpy
   Description :
   Author :        zhuguiliang
   date：          18/11/18
-------------------------------------------------
"""
import numpy as np
np.random.seed(110)


def sigmoid(z: np.float) -> np.float:
    return 1.0/(1+np.exp(-1.0*z))


def loss(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -1.0*np.mean(np.multiply(y, np.log(pred+0.00001)) + np.multiply(1.0-y, np.log(1.0-pred+0.00001)))


class LogisticClassifiter(object):
    def __init__(self, epochs=10, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.historay = []

    def optimize(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        # forward
        z = np.matmul(self.w.T, x) + self.b
        z = sigmoid(z)

        cost = loss(z, y)

        # backward
        self.dw = np.matmul(x, np.transpose(z-y))*(1.0/x.shape[1])
        self.db = np.mean((z-y))

        cost = np.squeeze(cost)

        self.historay.append(cost)

        # update params
        self.w = self.w - self.dw * self.lr
        self.b = self.b - self.db * self.lr

    def fit(self, x: np.ndarray, y: np.ndarray, versob=1):
        """

        :param x: shape(number of features, number of examples)
        :param y:
        :param versob:
        :return:
        """
        self.w = np.random.uniform(-1, 1, x.shape[0])
        self.b = 0

        for i in range(self.epochs):
            self.optimize(x, y)
            if versob <= 0:
                print('epoch {0} loss:{1}'.format(i, self.historay[-1]))
            elif versob > 0 and i % 100 == 0:
                print('epoch {0} loss:{1}'.format(i, self.historay[-1]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.matmul(self.w.T, x) + self.b
        z = sigmoid(z)
        return np.round(np.clip(z, 0, 1))