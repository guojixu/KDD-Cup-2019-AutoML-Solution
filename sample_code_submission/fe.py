
import numpy as np
from collections import Counter
from time import time
import multiprocessing
import heapq
from itertools import combinations, permutations
import pandas as pd
import os

from util import log, timeit
from parmap import parmap

class FE:
    def __init__(self,
                 CAT_num,
                 max_keep,
                 top_binary=50,
                 top_unary=100,  # ensure top_unary>top_binary
                 bin_num=64,
                 use_binary=False,
                 ):

        self.CAT_num = CAT_num
        self.top_binary = top_binary
        self.top_unary = top_unary
        self.cpu_num = multiprocessing.cpu_count()
        self.bin_num = bin_num
        self.use_binary = use_binary
        self.max_keep = max_keep

    @timeit
    def fit_transform(self, X, y):
        instance_num, feature_num = np.shape(X)[0], np.shape(X)[1]
        y = y.ravel()
        CAT_X, self.CAT_encoder = self._CAT_fit_transform(X[:, -self.CAT_num:], y)
        X = np.concatenate([X[:, :-self.CAT_num], CAT_X], axis=1)
        return X.astype(np.float32)

    @timeit
    def _CAT_fit_transform(self, X, y):
        X[X == 0] = 1
        feature_num = np.shape(X)[1]
        encoder, scores = [], [self._feature_importance(X[:, i], y) for i in range(feature_num)]

        index_unary = [i[0] for i in heapq.nlargest(self.top_unary, enumerate(scores), key=lambda x: x[1])]
        print('index_unary', index_unary)
        index_binary = index_unary[:self.top_binary]

        thr_index = 30
        if len(index_binary) < thr_index:
            thr_index = len(index_binary)
        print('scores', scores)
        threshold = np.sort(np.array(scores))[::-1][thr_index - 2]
        print('threshold ', threshold)

        #####################################  二阶 count start   ##############################################
        def do_job(job):
            if isinstance(job, int):
                feature = X[:, job]
            else:
                feature = self._get_hash_feature(X[:, job[0]], X[:, job[1]])
            feature = self._get_count_feature(feature)
            score = self._feature_importance(feature, y)
            if score > threshold:
                return np.reshape(feature, [-1, 1]), job, score
            else:
                return None, None, None

        print('-'*10, 'FE starts two order count')
        res = parmap(do_job, list(combinations(index_binary, 2)) + index_unary)
        X_count = [r[0] for r in res if not r[0] is None]
        encoder_count = [r[1] for r in res if not r[1] is None]
        scores_count = [r[2] for r in res if not r[2] is None]

        ############################## 相对 count percent start   #############################################

        def do_job(job):

            feature = self._get_hash_feature(X[:, job[0]], X[:, job[1]])
            feature_f1 = X[:, job[0]]

            feature = self._get_count_feature(feature)
            feature_f1 = self._get_count_feature(feature_f1)
            feature = feature / (feature_f1 + 0.01)
            score = self._feature_importance(feature, y)
            if score > threshold:
                return np.reshape(feature, [-1, 1]), job, score
            else:
                return None, None, None

        print('-'*10, 'FE starts relatively count percent')
        res = parmap(do_job, list(permutations(index_binary, 2)))
        X_count_percent = [r[0] for r in res if not r[0] is None]
        encoder_count_percent = [r[1] for r in res if not r[1] is None]
        scores_count_percent = [r[2] for r in res if not r[2] is None]

        ############################## 相对 count count / nunique start   #############################################
        def do_job(job):
            feature_f1 = X[:, job[0]]
            feature_f2 = X[:, job[1]]

            feature = self._get_count_feature(feature_f1)
            feature_f1 = self._get_nunique_feature(feature_f1, feature_f2)

            feature = feature / (feature_f1 + 0.01)

            score = self._feature_importance(feature, y)
            if score > threshold:
                return np.reshape(feature, [-1, 1]), job, score
            else:
                return None, None, None

        print('-'*10, 'FE starts relatively count / nunnique')
        res = parmap(do_job, list(permutations(index_binary, 2)))
        X_count_nunique_divide = [r[0] for r in res if not r[0] is None]
        encoder_count_nunique_divide = [r[1] for r in res if not r[1] is None]
        scores_count_nunique_divide = [r[2] for r in res if not r[2] is None]

        #####################################  二阶 nunique start   ##############################################
        def do_job(job):
            feature = self._get_nunique_feature(X[:, job[0]], X[:, job[1]])
            score = self._feature_importance(feature, y)
            if score > threshold:
                return np.reshape(feature, [-1, 1]), job, score
            else:
                return None, None, None

        print('-'*10, 'FE starts two order nunique')
        res = parmap(do_job, permutations(index_binary, 2))
        X_nunique = [r[0] for r in res if not r[0] is None]
        encoder_nunique = [r[1] for r in res if not r[1] is None]
        scores_nunique = [r[2] for r in res if not r[2] is None]

        print('-'*10, 'FE ends feature generation')

        index = [i[0] for i in heapq.nlargest(self.max_keep, enumerate(
            scores_count + scores_count_percent + scores_count_nunique_divide + scores_nunique), key=lambda x: x[1])]
        index_count = [i for i in index if i < len(scores_count)]
        index_count_percent = [i - len(scores_count) for i in index if
                               i < len(scores_count) + len(scores_count_percent) and i >= len(scores_count)]
        index_count_nunique_divide = [i - len(scores_count) - len(scores_count_percent) for i in index if
                                      i < len(scores_count) + len(scores_count_percent) + len(
                                          scores_count_nunique_divide) and i >= len(scores_count) + len(
                                          scores_count_percent)]
        index_nunique = [i - len(scores_count) - len(scores_count_percent) - len(scores_count_nunique_divide) for i in
                         index if i >= len(scores_count) + len(scores_count_percent) + len(scores_count_nunique_divide)]

        encoder_count, encoder_count_percent, encoder_count_nunique_divide, encoder_nunique = [encoder_count[i] for i in
                                                                                               index_count], [
                                                                                                  encoder_count_percent[
                                                                                                      i] for i in
                                                                                                  index_count_percent], [
                                                                                                  encoder_count_nunique_divide[
                                                                                                      i] for i in
                                                                                                  index_count_nunique_divide], [
                                                                                                  encoder_nunique[i] for
                                                                                                  i in index_nunique]
        print('index_count ', len(index_count), 'index_count_percent', len(encoder_count_percent),
              'index_count_nunique_divide', len(index_count_nunique_divide), ' index_nunique ', len(index_nunique))

        if len(index_count) > 0:
            X = np.concatenate([X, np.concatenate([X_count[i] for i in index_count], axis=1)], axis=1)
        if len(index_count_percent) > 0:
            X = np.concatenate([X, np.concatenate([X_count_percent[i] for i in index_count_percent], axis=1)], axis=1)
        if len(index_count_nunique_divide) > 0:
            X = np.concatenate(
                [X, np.concatenate([X_count_nunique_divide[i] for i in index_count_nunique_divide], axis=1)], axis=1)
        if len(index_nunique) > 0:
            X = np.concatenate([X, np.concatenate([X_nunique[i] for i in index_nunique], axis=1)], axis=1)

        return X.astype(np.float32), [encoder_count, encoder_count_percent, encoder_count_nunique_divide,
                                      encoder_nunique]

    @timeit
    def transform(self, X):
        #         start = time()
        X = np.concatenate([X[:, :-self.CAT_num], self._CAT_transform(X[:, -self.CAT_num:])], axis=1)
        #         print('fe transform finish![%.4f], shape after transform:%s' % (time()-start,str(np.shape(X))))
        return X.astype(np.float32)

    @timeit
    def _CAT_transform(self, X):
        if not len(self.CAT_encoder[0]) == 0:
            def do_job(job):
                if isinstance(job, int):
                    feature = X[:, job]
                else:
                    feature = self._get_hash_feature(X[:, job[0]], X[:, job[1]])
                return np.reshape(self._get_count_feature(feature), [-1, 1])

            X = np.concatenate([X, np.concatenate(parmap(do_job, self.CAT_encoder[0]), axis=1)], axis=1)

        if not len(self.CAT_encoder[1]) == 0:
            def do_job(job):
                feature = self._get_hash_feature(X[:, job[0]], X[:, job[1]])
                feature = self._get_count_feature(feature)
                feature_f = self._get_count_feature(X[:, job[0]])
                feature = feature / (feature_f + 0.01)
                return np.reshape(feature, [-1, 1])

            X = np.concatenate([X, np.concatenate(parmap(do_job, self.CAT_encoder[1]), axis=1)], axis=1)

        if not len(self.CAT_encoder[2]) == 0:
            def do_job(job):
                feature = self._get_count_feature(X[:, job[0]])
                feature_n = self._get_nunique_feature(X[:, job[0]], X[:, job[1]])
                feature = feature / (feature_n + 0.01)
                return np.reshape(feature, [-1, 1])

            X = np.concatenate([X, np.concatenate(parmap(do_job, self.CAT_encoder[2]), axis=1)], axis=1)

        if not len(self.CAT_encoder[3]) == 0:
            def do_job(job):
                feature = self._get_nunique_feature(X[:, job[0]], X[:, job[1]])
                return np.reshape(feature, [-1, 1])

            X = np.concatenate([X, np.concatenate(parmap(do_job, self.CAT_encoder[3]), axis=1)], axis=1)
        return X.astype(np.float32)

    def _get_hash_feature(self, feature1, feature2):
        return np.remainder(feature1, feature2) + feature1 * feature2 + feature1 / feature2

    def _get_count_feature(self, feature):

        return (pd.DataFrame(feature).fillna(0)).groupby([0])[0].transform('count').values

    def _get_cumcount_feature(self, feature):
        return (pd.DataFrame(feature).fillna(0)).groupby([0]).cumcount().values + 1.0

    def _get_nunique_feature(self, feature1, feature2):
        feature = np.concatenate([np.reshape(feature1, [-1, 1]), np.reshape(feature2, [-1, 1])], axis=1)
        return (pd.DataFrame(feature).fillna(0)).groupby(0)[1].transform('nunique').values

    def _get_single_nunique_feature(self, feature):
        feature = np.reshape(feature, [-1, 1])
        return (pd.DataFrame(feature).fillna(0)).groupby(0)[0].transform('nunique').values

    def _feature_importance(self, x, y):
        max_x, min_x = np.max(x), np.min(x)
        x_positive = x[np.where(y == 1)]
        x_negative = x[np.where(y == 0)]
        try:
            distribution_positive = np.histogram(x_positive, bins=self.bin_num, range=(min_x, max_x))[0] / \
                                  np.shape(x_positive)[0]
            distribution_negative = np.histogram(x_negative, bins=self.bin_num, range=(min_x, max_x))[0] / \
                                  np.shape(x_negative)[0]
            return self._hellinger_distance(distribution_positive, distribution_negative)
        except IndexError:
            return -1.0

    def _hellinger_distance(self, p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

