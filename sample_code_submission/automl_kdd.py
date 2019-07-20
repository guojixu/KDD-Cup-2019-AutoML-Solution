from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from util import Config, log, timeit
import datetime


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4
    }

    # X_sample, y_sample = data_sample(X, y, 30000)
    # sample_num = int(len(y) / 2)
    sample_num = 20000

    X_sample, y_sample = data_sample_new(X, y, sample_num)
    # print(X_sample)
    # print(y_sample)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    # hyperparams = hyperopt_lightgbm(X, y, params, config)

    # X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    # train_data = lgb.Dataset(X_train, label=y_train)
    # valid_data = lgb.Dataset(X_val, label=y_val)
    train_data = lgb.Dataset(X, label=y)

    config["model"] = lgb.train({**params, **hyperparams},
                                train_data,
                                150
                                # valid_data,
                                # early_stopping_rounds=30,
                                # verbose_eval=100
                                )

    try:
        # print("###############################"*100)
        config['models'].append(config["model"])
    except:
        # print("###############################" * 100)
        config["models"] = []
        config["models"].append(config["model"])


    importance_sorted = pd.DataFrame({'column': X.columns.tolist(),
                                      'importance': config["model"].feature_importance(),
                                     }).sort_values(by='importance')
    print(importance_sorted)

    # low_importance_cols = list(importance_sorted[importance_sorted['importance'] <= 5]['column'].values)
    #
    # print(low_importance_cols)
    #
    # X_cols = X.columns.tolist()
    # print(X_cols)
    #
    # remove_low_importance_cols = [col for col in X_cols if col not in low_importance_cols]
    #
    # print(remove_low_importance_cols)
    #
    # X = X[remove_low_importance_cols]
    #
    # X_sample, y_sample = data_sample_new(X, y, 30000)
    # hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)
    #
    # # hyperparams = hyperopt_lightgbm(X, y, params, config)
    #
    # X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    # train_data = lgb.Dataset(X_train, label=y_train)
    # valid_data = lgb.Dataset(X_val, label=y_val)
    # # train_data = lgb.Dataset(X, label=y)
    # config["model"] = lgb.train({**params, **hyperparams},
    #                             train_data,
    #                             150,
    #                             valid_data,
    #                             early_stopping_rounds=50,
    #                             verbose_eval=100
    #                             )
    #
    # print(pd.DataFrame({
    #     'column': X_train.columns.tolist(),
    #     'importance': config["model"].feature_importance(),
    # }).sort_values(by='importance'))
    # config["remove_low_importance_cols"] = remove_low_importance_cols




    # params = {
    #     "objective": "binary",
    #     "metric": "auc",
    #     "verbosity": -1,
    #     "seed": 1,
    #     "num_threads": 4,
    #
    #     "learning_rate": 0.01,
    #     "max_depth": -1,#-1
    #     "num_leaves": 31,##31
    #     "feature_fraction": 0.95,# 0.95
    #     "bagging_fraction": 0.95, #0.95
    #     "bagging_freq": 25,
    #     "reg_alpha": 1,
    #     "reg_lambda": 1,
    #     "min_child_weight": 0.001, #0.001
    # }
    #
    # # X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    #
    # # train_data = lgb.Dataset(X_train, label=y_train)
    # # valid_data = lgb.Dataset(X_val, label=y_val)
    #
    # train_data = lgb.Dataset(X, label=y)
    # config["model"] = lgb.train({**params},
    #                             train_set=train_data,
    #                             # valid_sets=valid_data,#
    #                             num_boost_round=150,
    #                             # early_stopping_rounds=30,#
    #                             # verbose_eval=50#
    #                             )





@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    # return config["model"].predict(X)

    tmp = []
    model_num = len(config["models"])
    for model in config["models"]:
        tmp.append(model.predict(X))
    print(tmp)
    tmp = list(np.sum(tmp, axis=0))
    tmp = list(map(lambda a: a / model_num, tmp))
    # print(tmp)
    return tmp

@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        # "max_depth": hp.choice("max_depth", np.linspace(2, 30, 14, dtype=int)),
        # "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "max_depth": hp.choice("max_depth", [-1, 5, 6, 7]),

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
                         rstate=np.random.RandomState((datetime.datetime.now()-datetime.datetime(1970, 1, 1)).microseconds))

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


    # if len(pos_index) / len(neg_index) < 0.001:
    #     p_indics = random.sample(pos_index, sample_num)
    #     n_indics = random.sample(neg_index, int(len(neg_index) / 10))
    #
    #     print("#" * 50)
    #     print(len(p_indics), len(n_indics))
    #     print("#" * 50)
    #
    #     return p_indics + n_indics


    if sample_num < (nrows / 5):
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, sample_num)

    else:
        p_indics = random.sample(pos_index, sample_num)
        n_indics = random.sample(neg_index, sample_num)

    print("#" * 50)
    print(len(p_indics), len(n_indics))
    print("#" * 50)

    return p_indics + n_indics

    # print(len(pos_index), len(neg_index), "in train hyperOpt sample")
    # return pos_index + neg_index

def data_sample_new(X: pd.DataFrame, y: pd.Series, nrows: int=10000):

    # print(X)
    # print(y)
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    # print(X)
    # print(y)
    sampled_indices = sample_indices(y, nrows)
    return X.iloc[sampled_indices, :], y[sampled_indices]
