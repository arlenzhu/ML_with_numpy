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
import copy


class DecisionTree(object):
    def __init__(self, max_depth=5, min_examples_split=2):
        self.max_depth = max_depth
        self.min_examples_split = min_examples_split
        self.depth = 0
        self.my_tree = dict()

    @staticmethod
    def entropy(labels: list) -> float:
        """
        ent = sum(-pi * log2(pi))
        :param labels:
        :return:
        """
        val, counts = np.unique(labels, return_counts=True)
        counts = counts / len(labels)
        ent = 0.
        for prob in counts:
            if prob != 0.0:
                ent -= prob * np.log2(prob)
        return ent

    @staticmethod
    def chossebestfeature(features: list, labels: list):
        """

        :param features:
        :param labels:
        :return:
        """
        num_feature = len(features[0])
        example = len(features)
        base_entropy = DecisionTree.entropy(labels)
        best_info_gain_ratio = -np.Inf
        best_feature = -1
        best_split = None

        for i in range(num_feature):  # 遍历计算所有特征，计算信息增益率，选择最优的属性及划分点
            info_gain_ratio = []
            feature_values = [example[i] for example in features]
            unique_values = np.unique(feature_values)  # 选择可能划分的属性值

            for idx, value in enumerate(unique_values):  # 遍历所有属性值，计算各属性划分的信息增益，划分为左右两个节点
                new_entropy = 0.
                split_info = 0.
                if isinstance(value, str):  # 如果特征是字符，用是否相等来划分
                    sub_data_left = [l for v, l in zip(features, labels) if v[i] == value]
                    sub_data_right = [l for v, l in zip(features, labels) if v[i] != value]
                else:
                    if idx+1 == unique_values.shape[0]:
                        break
                    value = np.median([value, unique_values[idx + 1]])  # 数字型选择前后值的中位数进行划分
                    sub_data_left = [l for v, l in zip(features, labels) if v[i] <= value]
                    sub_data_right = [l for v, l in zip(features, labels) if v[i] > value]
                for sub_data in [sub_data_left, sub_data_right]:
                    if len(sub_data) > 0:
                        prob = float(len(sub_data) / example)
                        new_entropy += prob * DecisionTree.entropy(sub_data)
                        split_info -= prob * np.log2(prob)

                if split_info == 0.:
                    continue
                info_gain = base_entropy - new_entropy
                info_gain_ratio.append(info_gain / split_info)

            if info_gain_ratio:  # 保存最大信息增益率的特征索引及划分值
                max_info_gaim_ratio = max(info_gain_ratio)
                if max_info_gaim_ratio > best_info_gain_ratio:
                    best_info_gain_ratio = max_info_gaim_ratio
                    best_feature = i
                    best_index = np.argmax(info_gain_ratio)
                    best_split = np.median([unique_values[best_index], unique_values[best_index+1]])

        return best_feature, best_split

    def create_tree(self, x, y, featues_name):
        val, counts = np.unique(y, return_counts=True)
        # 判断是否满足停止划分的条件
        if len(y) <= self.min_examples_split or self.depth >= self.max_depth or len(val) == 1:
            return val[np.argmax(counts)]

        best_feature, best_split = DecisionTree.chossebestfeature(x, y)
        best_feature_label = featues_name[best_feature]
        my_tree = {best_feature_label: {}}

        self.depth += 1
        if isinstance(best_split, str):
            sub_labels = featues_name[:]
            sub_data_left = [(v, l) for v, l in zip(x, y) if v[best_feature] == best_split]
            sub_data_right = [(v, l) for v, l in zip(x, y) if v[best_feature] != best_split]
            sub_x_left = [i[0] for i in sub_data_left]
            sub_y_left = [i[1] for i in sub_data_left]

            sub_x_right = [i[0] for i in sub_data_right]
            sub_y_right = [i[1] for i in sub_data_right]

            if len(sub_data_left) > 0:
                my_tree[best_feature_label]['==' + str(best_split)] = self.create_tree(sub_x_left, sub_y_left,
                                                                                       sub_labels)
            if len(sub_data_right) == 0:
                my_tree[best_feature_label]['!=' + str(best_split)] = self.create_tree(sub_x_right, sub_y_right,
                                                                                      sub_labels)

        else:
            sub_labels = featues_name[:]
            sub_data_left = [(v, l) for v, l in zip(x, y) if v[best_feature] <= best_split]
            sub_data_right = [(v, l) for v, l in zip(x, y) if v[best_feature] > best_split]
            sub_x_left = [i[0] for i in sub_data_left]
            sub_y_left = [i[1] for i in sub_data_left]

            sub_x_right = [i[0] for i in sub_data_right]
            sub_y_right = [i[1] for i in sub_data_right]

            if len(sub_data_left) > 0:
                my_tree[best_feature_label]['<=' + str(best_split)] = self.create_tree(sub_x_left, sub_y_left,
                                                                                       sub_labels)
            if len(sub_data_right) > 0:
                my_tree[best_feature_label]['>' + str(best_split)] = self.create_tree(sub_x_right, sub_y_right,
                                                                                      sub_labels)

        return my_tree

    def fit(self, x, y, featues_name):
        self.my_tree = self.create_tree(x, y, featues_name)
        # 创建feature对应索引位置字典
        self.featues_name = {v: k for k, v in enumerate(feature_name)}

    def pred(self, x):
        y = []
        for row in x:
            y_ = self._tree(row, self.my_tree)
            y.append(y_)
        return y

    def _tree(self, row, tree):
        """

        :param row:
        :param tree:
        :return:
        """
        root_tree = copy.deepcopy(tree)
        for k, v in root_tree.items():
            real_value = row[self.featues_name.get(k)]
            split_left, split_right = list(v.keys())
            if eval('{0}{1}'.format(real_value, split_left)):  # 判断是否符合节点
                new_tree = copy.deepcopy(root_tree[k][split_left])
                if isinstance(new_tree, dict):  # 未到叶节点继续遍历
                    return self._tree(row, new_tree)
                else:
                    return new_tree  # 返回叶节点标签
            else:
                new_tree = copy.deepcopy(root_tree[k][split_right])
                if isinstance(new_tree, dict):
                    return self._tree(row, new_tree)
                else:
                    return new_tree


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTree(max_depth=100, min_examples_split=2)
    x, y = load_iris(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5)
    feature_name = load_iris().feature_names

    dt.fit(train_x, train_y, feature_name)
    print(dt.my_tree)

    pred = dt.pred(test_x)

    print('precision:', precision_score(test_y, pred, average='macro'))
    print('recall:', recall_score(test_y, pred, average='macro'))
    print('f1 score:', f1_score(test_y, pred, average='macro'))

    skdt = DecisionTreeClassifier(max_depth=100, min_samples_split=2)
    skdt.fit(train_x, train_y)
    pred = skdt.predict(test_x)

    print('sk precision:', precision_score(test_y, pred, average='macro'))
    print('sk recall:', recall_score(test_y, pred, average='macro'))
    print('sk f1 score:', f1_score(test_y, pred, average='macro'))
