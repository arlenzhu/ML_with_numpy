# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tfclassifiter
   Description :
   Author :        zhuguiliang
   date：          18/11/20
-------------------------------------------------
"""
from collections import defaultdict
import math


class TfClassifiter(object):
    def __init__(self, min_df=2, max_df=10000, max_features=20000, ngram_range=3, stop_words=None, **kwargs):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.tf = dict()
        self.idf = dict()

    def fit(self, texts_iter, labels):
        """

        :param texts_iter: ['word1 word2 word3', 'word1 word2 word3']
        :param labels: ['class1', class2']
        :return:
        """
        ccc = defaultdict()
        for text, label in zip(texts_iter, labels):
            words = text.split()
            features = self._add_ngram(words)
            for word in features:
                ccc.setdefault(word, []).append(label)
            self.tf.setdefault(label, []).extend(features)

        self.idf = {word: math.log((len(labels)+1.)/(1.+len(set(clist)))) for word, clist in ccc.items() if
                    self.min_df <= len(clist) <= self.max_df}

    def get_features_names(self):
        return self.idf.keys()

    def get_k_features(self, k=20) -> list:
        res = []
        for label, clist in self.tf.items():
            top_words = dict()
            print('class #:', label)
            for word, idf in self.idf.items():
                top_words[word] = idf * (clist.count(word)/(len(clist)))
            top = sorted(top_words.items(), key=lambda kv: kv[1])[-k:]
            top = [i[0] for i in top if i[1] > 0]
            print(' '.join(top))
            res.append((label, top))
        return res

    def _add_ngram(self, words):
        words = [i for i in words if i not in self.stop_words]
        new_sequences = words[:]
        if self.ngram_range > 1:
            for ngram in range(2, self.ngram_range+1):
                ngrams = list(zip(*[words[i:] for i in range(ngram)]))
                ngrams = self._duplicate_list(ngrams)
                ngrams = [''.join(i) for i in ngrams]
                new_sequences.extend(ngrams)
        return new_sequences

    def _duplicate_list(self, clist: list) -> list:
        new_list = []
        for i in clist:
            if i not in new_list and i not in self.stop_words:
                new_list.append(i)
        return new_list

    def transformer(self):
        pass


if __name__ == '__main__':
    import jieba
    tfidf = TfClassifiter(min_df=2, stop_words=['的', '是', '啥', '一'])
    texts = ["""打电话的还有多少""",
             """我的电话打完了吗""",
             """查询当前余额""",
             """请再重复一遍话费余额"""
             ]
    labels = ['class1', 'class1', 'class2', 'class2']
    texts = [' '.join(jieba.lcut(sent)) for sent in texts]
    tfidf.fit(texts, labels)
    tfidf.get_k_features(10)
    print(tfidf.tf)