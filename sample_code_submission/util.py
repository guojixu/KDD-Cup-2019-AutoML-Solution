import os
import pickle
import time
from typing import Any

import CONSTANT

nesting_level = 0
is_start = None

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")


from scipy import stats


class Config:
    def __init__(self, info):
        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        import numpy as np

        def my_nunique(x):
            return x.nunique()

        def func_mv_union(x):

            x = [m for m in x if isinstance(m, str) and len(m) != 1]

            if len(x) != 0:
                return ','.join(x)

            else:
                return '0'

        def func_mv_last(x):

            # print(len(x))

            x = [m for m in x if isinstance(m, str) and len(m) != 1][-1:]

            # print(x)

            if len(x) != 0:
                # return ','.join(x[0].split(','))
                return x[0]
            else:
                return '0'

            # x = [m for m in x if isinstance(m, str) and len(m) != 1]
            # if len(x) != 0:
            #     return ','.join(sorted(set(','.join(x).split(','))))
            #     # return ','.join(sorted(','.join(x).split(',')))
            # else:
            #     return '0'

        def func_cat_last(x):

            # print(x)
            x = [c for c in list(x) if c is not np.nan]
            # print(x)
            # if len(x) != 0:
            #     print(stats.mode(x)[0][0])
            # print("=" * 10)

            if len(x) != 0:
                return x[-1:][0]
            else:
                return '0'

        def func_cat_union(x):

            # x = [c for c in list(x) if isinstance(c, float)]
            #
            # if len(x) != 0:
            #     return stats.mode(x)[0][0]
            # else:
            #     return '0'

            x = [c for c in list(x) if c is not np.nan]
            if len(x) != 0:
                return ','.join(map(str, sorted(set(list(x)))))
                # return ','.join(map(str, sorted(list(x))))
            else:
                return '0'

        my_nunique.__name__ = 'nunique'
        ops = {
            CONSTANT.NUMERICAL_TYPE: ["mean", "sum"],
            CONSTANT.CATEGORY_TYPE: [func_cat_last, func_cat_union],
            #  TIME_TYPE: ["max"],
            #  MULTI_CAT_TYPE: [my_unique]
            CONSTANT.MULTI_CAT_TYPE: [func_mv_last, func_mv_union]

        }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.CATEGORY_TYPE]
        if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            # assert False, f"MultiCategory type feature's aggregate op are not supported."
            return ops[CONSTANT.MULTI_CAT_TYPE]
        if col.startswith(CONSTANT.TIME_PREFIX):
            assert False, f"Time type feature's aggregate op are not implemented."
        assert False, f"Unknown col type {col}"

    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)


from util import log, timeit
# from FeatureSelector import FeatureSelector
from preprocess import gen_mutual_info
from preprocess import sample_indices

from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import gc
from sklearn.preprocessing import MinMaxScaler


@timeit
def feature_selector(X, y, threshold=0.001, max_numb_cols_to_select=0):


    # fs = FeatureSelector(data=X, labels=y)
    # fs.identify_single_unique()
    # print(fs.ops)
    #
    # X = fs.remove(methods=['single_unique'], keep_one_hot=False)


    print("===========================================\n"
          "====================WAIT=======================\n"
          "===========================================\n")
    # time.sleep(6)


    # if max_numb_cols_to_select - 30 > len(X.columns.tolist()):
    #
    #     return X, X.columns.tolist()


    print(max_numb_cols_to_select)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4
    }

    # X_sample, y_sample = data_sample(X, y, 30000)
    sample_num = int(len(y) / 6)
    print(sample_num)
    X_sample, y_sample = data_sample_new_hyperopt(X, y, sample_num)

    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, None)

    # hyperparams = hyperopt_lightgbm(X, y, params, config)

    # X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    X_train, X_val, y_train, y_val = data_split(X, y, 0.1)



    #
    #
    # train_data = lgb.Dataset(X_train, label=y_train)
    # valid_data = lgb.Dataset(X_val, label=y_val)
    # # train_data = lgb.Dataset(X, label=y)
    # model = lgb.train({**params, **hyperparams},
    #                             train_data,
    #                             150,
    #                             valid_data,
    #                             early_stopping_rounds=30,
    #                             verbose_eval=100
    #                             )
    #
    # importance_sorted = pd.DataFrame({'column': X_train.columns.tolist(),
    #                                   'importance': model.feature_importance(),
    #                                  }).sort_values(by='importance')
    # print(importance_sorted)
    #
    # if importance_sorted['importance'].max() <= 100:
    #     print('#'*20)
    #     print("remove 0 importance feature")
    #     print(importance_sorted['importance'].max(), 'max')
    #     print(importance_sorted['importance'].mean(), 'mean')
    #     print(importance_sorted['importance'].mad(), 'mad')
    #     print(importance_sorted['importance'].median(), 'median')
    #     print(importance_sorted['importance'].std(), 'std')
    #     print(importance_sorted['importance'].var(), 'var')
    #
    #     mean_bar = importance_sorted['importance'].mean()
    #
    #
    #     low_importance_cols = list(importance_sorted[importance_sorted['importance'] <= mean_bar]['column'].values)
    #     print(importance_sorted.shape)
    #     print(len(low_importance_cols))
    #     print('#' * 20)
    #     print(importance_sorted.shape[0] - len(low_importance_cols))
    #
    #
    # else:
    #     print('#'*20)
    #     print("remove 1 importance feature")
    #
    #     print(importance_sorted['importance'].max(), 'max')
    #     print(importance_sorted['importance'].mean(), 'mean')
    #     print(importance_sorted['importance'].mad(), 'mad')
    #     print(importance_sorted['importance'].median(), 'median')
    #     print(importance_sorted['importance'].std(), 'std')
    #     print(importance_sorted['importance'].var(), 'var')
    #
    #     mean_bar = importance_sorted['importance'].mean()
    #
    #
    #     low_importance_cols = list(importance_sorted[importance_sorted['importance'] <= mean_bar]['column'].values)
    #     print(importance_sorted.shape)
    #     print(len(low_importance_cols))
    #     print(importance_sorted.shape[0] - len(low_importance_cols))
    #
    #     print('#' * 20)
    # print(low_importance_cols)
    #
    # X_cols = X.columns.tolist()
    # print(X_cols)
    #
    # after_remove_low_importance_cols = [col for col in X_cols if col not in low_importance_cols]
    #
    # print(after_remove_low_importance_cols)
    # print(len(after_remove_low_importance_cols))
    #
    #
    #
    #
    #
    # # select feature from low importance feature
    #
    # train_data = lgb.Dataset(X_train[low_importance_cols], label=y_train)
    # valid_data = lgb.Dataset(X_val[low_importance_cols], label=y_val)
    # # train_data = lgb.Dataset(X, label=y)
    # model = lgb.train({**params, **hyperparams},
    #                             train_data,
    #                             150,
    #                             valid_data,
    #                             early_stopping_rounds=30,
    #                             verbose_eval=100
    #                             )
    #
    # importance_sorted_2 = pd.DataFrame({'column': X_train[low_importance_cols].columns.tolist(),
    #                                   'importance': model.feature_importance(),
    #                                   }).sort_values(by='importance')
    #
    # print(importance_sorted_2)
    #
    # if importance_sorted_2['importance'].max() <= 200:
    #     print('#'*20)
    #     print("remove 0 importance feature")
    #     print(importance_sorted_2['importance'].max(), 'max')
    #     print(importance_sorted_2['importance'].mean(), 'mean')
    #     print(importance_sorted_2['importance'].mad(), 'mad')
    #     print(importance_sorted_2['importance'].median(), 'median')
    #     print(importance_sorted_2['importance'].std(), 'std')
    #     print(importance_sorted_2['importance'].var(), 'var')
    #
    #     mean_bar = importance_sorted_2['importance'].mean()
    #
    #     low_importance_cols_2 = list(importance_sorted_2[importance_sorted_2['importance'] <= mean_bar]['column'].values)
    #     print(importance_sorted_2.shape)
    #     print(len(low_importance_cols_2))
    #     print(importance_sorted_2.shape[0] - len(low_importance_cols_2))
    #     print('#' * 20)
    #
    # else:
    #     print('#'*20)
    #     print("remove 1 importance feature")
    #
    #     print(importance_sorted_2['importance'].max(), 'max')
    #     print(importance_sorted_2['importance'].mean(), 'mean')
    #     print(importance_sorted_2['importance'].mad(), 'mad')
    #     print(importance_sorted_2['importance'].median(), 'median')
    #     print(importance_sorted_2['importance'].std(), 'std')
    #     print(importance_sorted_2['importance'].var(), 'var')
    #
    #     mean_bar = importance_sorted_2['importance'].mean()
    #
    #     low_importance_cols_2 = list(importance_sorted_2[importance_sorted_2['importance'] <= mean_bar]['column'].values)
    #     print(importance_sorted_2.shape)
    #     print(len(low_importance_cols_2))
    #     print(importance_sorted_2.shape[0] - len(low_importance_cols_2))
    #
    #     print('#' * 20)
    #
    #
    #
    # after_remove_low_importance_cols_2 = [col for col in low_importance_cols if col not in low_importance_cols_2]
    #
    # after_remove_low_importance_cols = after_remove_low_importance_cols + after_remove_low_importance_cols_2
    #
    # print(after_remove_low_importance_cols)
    # print(len(after_remove_low_importance_cols))




    ################################################################

    # num_select = 1
    #
    # labeled_cols = [col for col in X_train.columns.tolist() if 'comb' in col]
    #              # or ('m_' in col and 'cnt_' not in col and 'length' not in col and 'sum' not in col)]
    #
    # print(labeled_cols)
    #
    # low_importance_labeled_cols = labeled_cols
    #
    # hyperparams_1 = hyperopt_lightgbm(X_sample[low_importance_labeled_cols], y_sample, params, None)
    #
    # after_remove_low_importance_labeled_cols = []
    #
    # for i in range(num_select):
    #
    #     cols_to_select = low_importance_labeled_cols
    #
    #     train_data = lgb.Dataset(X_train[cols_to_select], label=y_train)
    #     valid_data = lgb.Dataset(X_val[cols_to_select], label=y_val)
    #     model = lgb.train({**params, **hyperparams_1},
    #                       train_data,
    #                       150,
    #                       valid_data,
    #                       early_stopping_rounds=30,
    #                       verbose_eval=100
    #                       )
    #
    #     importance_sorted = pd.DataFrame({'column': X_train[cols_to_select].columns.tolist(),
    #                                         'importance': model.feature_importance(),
    #                                         }).sort_values(by='importance', ascending=False)
    #
    #     print(importance_sorted)
    #
    #     print('#'*20)
    #     print("remove 1 importance feature")
    #
    #     print(importance_sorted['importance'].max(), 'max')
    #     print(importance_sorted['importance'].mean(), 'mean')
    #     print(importance_sorted['importance'].mad(), 'mad')
    #     print(importance_sorted['importance'].median(), 'median')
    #     print(importance_sorted['importance'].std(), 'std')
    #     print(importance_sorted['importance'].var(), 'var')
    #
    #     thres_imp = importance_sorted['importance'].mean()
    #
    #     low_importance_labeled_cols = list(importance_sorted[importance_sorted['importance'] <= thres_imp]['column'].values)
    #     high_importance_labeled_cols = list(importance_sorted[importance_sorted['importance'] > thres_imp]['column'].values)
    #
    #     print(low_importance_labeled_cols)
    #
    #     # after_remove_low_importance_cols = after_remove_low_importance_cols + [col for col in cols_to_select if col not in low_importance_cols]
    #     after_remove_low_importance_labeled_cols = after_remove_low_importance_labeled_cols + high_importance_labeled_cols
    #     print(importance_sorted.shape)
    #     print(len(low_importance_labeled_cols))
    #
    #     print(importance_sorted.shape[0] - len(low_importance_labeled_cols))
    #     print('#'*20)
    #
    #
    #
    #
    #
    # low_importance_cols = [col for col in X_train.columns.tolist() if col not in labeled_cols]
    #
    # hyperparams_2 = hyperopt_lightgbm(X_sample[low_importance_cols], y_sample, params, None)


    ################################################################




    # reserve_cols = [col for col in low_importance_cols if 't_0' in col
    #                 and 'c_' in col and 'count' in col]

    # print('#' * 20)
    # print(reserve_cols)
    # print('#' * 20)

    low_importance_cols = X_train.columns.tolist()

    hold_cols = []

    # hold_cols = [col for col in low_importance_cols if 't_0' in col and 'c_0' in col and 'count' in col]



    low_importance_cols = [col for col in low_importance_cols if col not in hold_cols]

    after_remove_low_importance_cols = []
    num_select = 2

    for i in range(num_select):

        cols_to_select = low_importance_cols

        train_data = lgb.Dataset(X_train[cols_to_select], label=y_train)
        valid_data = lgb.Dataset(X_val[cols_to_select], label=y_val)
        model = lgb.train({**params, **hyperparams},
                          train_data,
                          150,
                          valid_data,
                          early_stopping_rounds=30,
                          verbose_eval=100
                          )

        importance_sorted = pd.DataFrame({'column': X_train[cols_to_select].columns.tolist(),
                                            'importance': model.feature_importance(),
                                            }).sort_values(by='importance', ascending=False)

        print(importance_sorted)

        print('#'*20)
        print("remove 1 importance feature")

        print(importance_sorted['importance'].max(), 'max')
        print(importance_sorted['importance'].mean(), 'mean')
        print(importance_sorted['importance'].mad(), 'mad')
        print(importance_sorted['importance'].median(), 'median')
        print(importance_sorted['importance'].std(), 'std')
        print(importance_sorted['importance'].var(), 'var')

        thres_imp = importance_sorted['importance'].mean()

        low_importance_cols = list(importance_sorted[importance_sorted['importance'] <= thres_imp]['column'].values)
        high_importance_cols = list(importance_sorted[importance_sorted['importance'] > thres_imp]['column'].values)

        print(low_importance_cols)

        # after_remove_low_importance_cols = after_remove_low_importance_cols + [col for col in cols_to_select if col not in low_importance_cols]
        after_remove_low_importance_cols = after_remove_low_importance_cols + high_importance_cols
        print(importance_sorted.shape)
        print(len(low_importance_cols))
        print(importance_sorted.shape[0] - len(low_importance_cols))
        print('#'*20)


    # reserve_cols = [col for col in reserve_cols if col not in after_remove_low_importance_cols]

    # after_remove_low_importance_cols = after_remove_low_importance_cols + reserve_cols

    # after_remove_low_importance_cols = after_remove_low_importance_cols + after_remove_low_importance_labeled_cols




    # len_labeled_cols = len(after_remove_low_importance_labeled_cols)

    is_drop = len(after_remove_low_importance_cols) - max_numb_cols_to_select
    if is_drop > 0:
        print('#' * 50)
        print('remove unnecessary')
        print('#' * 50)
        after_remove_low_importance_cols = after_remove_low_importance_cols[:max_numb_cols_to_select - len(hold_cols) - 30]
        print(after_remove_low_importance_cols)
        print(len(after_remove_low_importance_cols))

    # after_remove_low_importance_cols = after_remove_low_importance_cols + after_remove_low_importance_labeled_cols


    after_remove_low_importance_cols = after_remove_low_importance_cols + hold_cols

    print(after_remove_low_importance_cols)
    print(len(after_remove_low_importance_cols))

    print(max_numb_cols_to_select, 'max_numb_cols_to_select')

    # tmp = pd.concat(sorted_dfs_list, axis=0, sort=False)
    #
    # print(tmp.sort_values(by='importance'))
    # print('#' * 20)
    # print(tmp['importance'].mean(), 'mean')
    # thres_imp = tmp['importance'].mean()



    # after_remove_low_importance_cols = list(tmp[tmp['importance'] >= thres_imp]['column'].values)
    # print(len(after_remove_low_importance_cols))
    # print('#' * 20)
    # print(after_remove_low_importance_cols)



    #
    # cols_selected = X.columns.tolist()
    #
    # sampled_indices = sample_indices(y)
    #
    # metric = gen_mutual_info(X[X.columns.tolist()].loc[sampled_indices, :], y[sampled_indices])
    #
    # print('=========================================WAIT================================================'
    #       '=========================================================================================')
    #
    # time.sleep(10)
    #
    #
    # # metric_dict = dict(zip(X.columns.tolist(), metric))
    #
    # metric_dict = dict(zip(X.columns.tolist(), metric))
    #
    #
    # bool_selected_cols = [value > threshold for value in metric]
    #
    # X = X.loc[:, bool_selected_cols]
    #
    # print(X.head())
    #
    # print(metric_dict)
    #
    # print(X.columns.tolist())
    #
    #
    # cols_selected = X.columns.tolist()


    return X[after_remove_low_importance_cols], after_remove_low_importance_cols


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        # "max_depth": hp.choice("max_depth", np.linspace(10, 30, 20, dtype=int)),
        # "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "max_depth": hp.choice("max_depth", [-1, 4, 5, 6, 7]),

        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.6, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 150,
                          valid_data, early_stopping_rounds=30, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=10, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample

import random
def sample_indices(y, nrows):
    pos_index = np.where(y.ravel() == 1)[0].tolist()
    neg_index = np.where(y.ravel() == 0)[0].tolist()

    print("#" * 50)
    print(len(pos_index), len(neg_index))
    print("#" * 50)


    sample_num = min(len(pos_index), len(neg_index))
    sample_num = min(sample_num, nrows)


    if len(pos_index) / len(neg_index) < 0.001:
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, int(len(neg_index) / 4))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(len(p_indics))
        print(len(n_indics))
        return p_indics + n_indics


    if sample_num < (nrows / 5):
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, nrows)
    else:

        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, sample_num)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(len(p_indics))
    print(len(n_indics))

    return p_indics + n_indics

def data_sample_new(X: pd.DataFrame, y: pd.Series, nrows: int=10000):

    sampled_indices = sample_indices(y, nrows)
    return X.iloc[sampled_indices, :], y[sampled_indices]




def sample_indices_hyperopt(y, nrows):
    pos_index = np.where(y.ravel() == 1)[0].tolist()
    neg_index = np.where(y.ravel() == 0)[0].tolist()

    print(len(pos_index), len(neg_index))

    sample_num = min(len(pos_index), len(neg_index))
    sample_num = min(sample_num, nrows)

    if len(pos_index) / len(neg_index) < 0.01:
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, int(len(neg_index) / 6))
        print("#"*20)
        print(len(p_indics), len(n_indics))
        print("#" * 20)
        return p_indics + n_indics

    if sample_num < (len(neg_index) / 5):
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, len(neg_index))

    else:
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, sample_num)

    print("#" * 20)
    print(len(p_indics), len(n_indics))
    print("#" * 20)
    return p_indics + n_indics
    # print(len(pos_index), len(neg_index), "in select hyperOpt sample")
    # return pos_index + neg_index

def data_sample_new_hyperopt(X: pd.DataFrame, y: pd.Series, nrows: int=10000):

    sampled_indices = sample_indices_hyperopt(y, nrows)
    return X.iloc[sampled_indices, :], y[sampled_indices]


import signal
from contextlib import contextmanager
import math
import datetime

def my_mprint(msg):
    """info"""
    cur_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    print(f"INFO  [{cur_time}] {msg}")

class MyTimeoutException(Exception):

    pass


class MyTimer:
    def __init__(self):
        self.duration = 0
        self.total = 0
        self.remain = None
        self.exec = 0

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname='my timer'):
        def signal_handler(signum, frame):
            raise MyTimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.remain)
        start_time = time.time()
        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            remain_time = math.ceil(self.total - self.exec)
            self.remain = remain_time

            my_mprint(f'{pname} success, time spent so far {self.exec} sec')


