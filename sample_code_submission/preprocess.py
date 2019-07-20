import datetime

import CONSTANT
from util import log, timeit
from category_encoders import OrdinalEncoder
import multiprocessing as mp
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import gc
from sklearn.preprocessing import MinMaxScaler
import copy
import os
import time
import numpy as np
import random
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations, product, permutations
import pandas as pd

from fe import FE

pool_maxtasksperchild = 100


@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


@timeit
def feature_engineer(df, config, y=None,
                     two_order_cols=None,
                     two_group_cols=None,
                     mv_encs=None,
                     c_one_order_cols=None,
                     c_two_order_cols=None,
                     c_two_order_group_cnt_cols=None,
                     c_two_order_n_groupby_cat_cols=None,
                     n_minus_mean_cols=None,
                     cols_selected=None,
                     big_df_memory=0,
                     big_df_len=0,
                     fe_model=None,
                     ):
    size_info_df = (0, 0)

    idx = df.index
    # 1.preprocess raw data

    time_col = df[config['time_col']]

    t_cols = [t for t in df if t.startswith(CONSTANT.TIME_PREFIX)]
    n_cols = [n for n in df if n.startswith(CONSTANT.NUMERICAL_PREFIX)]
    c_cols = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    mv_cols = [m for m in df if m.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    print(mv_cols)

    flag_t = True
    flag_n = True
    flag_c = True
    flag_mv = True

    if len(t_cols) <= 0:
        flag_t = False
    if len(n_cols) <= 0:
        flag_n = False
    if len(c_cols) <= 0:
        flag_c = False
    if len(mv_cols) <= 0:
        flag_mv = False

    t_df = None
    n_df = None
    c_df = None
    mv_df = None
    n_tmmporal_df = None

    if flag_t:
        t_df = df[t_cols].reset_index(drop=True)
    if flag_n:
        n_df = df[n_cols].reset_index(drop=True)
        time_col.reset_index(drop=True, inplace=True)

        n_tmporal_df = temporal_feature_engineer(pd.concat([copy.deepcopy(n_df), time_col], axis=1), config['time_col'], cols_selected)
        # n_tmporal_df.fillna(0, inplace=True)
        if n_tmporal_df is not None:
            print(n_tmporal_df.columns.tolist())
            # print(n_tmporal_df.head())
    if flag_c:
        c_df = df[c_cols].reset_index(drop=True)
    if flag_mv:
        mv_df = df[mv_cols].reset_index(drop=True)

    small_df_len = df.shape[0]

    del df
    del time_col
    gc.collect()

    # print(t_df.head())
    t_minus_t_df = None
    t_df_to_int_processed = None
    if flag_t:
        t_minus_t_df = t_minus_t(t_df, config)

    if t_minus_t_df is not None:
        print(t_minus_t_df.columns.tolist())
        t_df.drop(t_minus_t_df.columns.tolist(), axis=1, inplace=True)

    if flag_t:
        t_df_to_int_processed = time_to_int(t_df)
    if t_df_to_int_processed is not None:
        print(t_df_to_int_processed.columns.tolist())

    # size_info_t_df = (t_df_to_int_processed.memory_usage().sum(), len(t_df_to_int_processed.columns.tolist()))

    del t_df
    gc.collect()

    # n_df_processed = n_df

    n_df_processed = None
    size_info_n_df = None
    if flag_n:
        n_df_processed = func_to_n(n_df, config)
        size_info_n_df = (n_df_processed.memory_usage().sum(), len(n_df_processed.columns.tolist()))

    if n_df_processed is not None:
        # print(n_df_processed.head())
        # print(n_df_processed.tail())
        print(n_df_processed.columns.tolist())

    # c_df_to_int_processed = get_dummy_encode(c_df)
    c_df_to_int_processed = None
    size_info_c_df = None
    if flag_c:
        c_df_to_int_processed = get_dummy_encode_new(c_df)
        size_info_c_df = (c_df_to_int_processed.memory_usage().sum(), len(c_df_to_int_processed.columns.tolist()))

    if c_df_to_int_processed is not None:
        print(c_df_to_int_processed.columns.tolist())

        cat_num = len(c_df_to_int_processed.columns.tolist())

        if fe_model is None:
            fe = FE(cat_num, 3 * cat_num)
            # print(c_df_to_int_processed.values)
            # print(y.values)
            fe_c_arr = fe.fit_transform(c_df_to_int_processed.values, y.values)
            print(fe_c_arr.shape)
            # print(fe_c_arr)

            fe_c_df = pd.DataFrame(fe_c_arr)
            fe_c_df.columns = fe_c_df.columns.map(lambda a: 'fe_' + str(a))

            del fe_c_arr

            gc.collect()

            print(fe_c_df.shape)
            # print(fe_c_df)

            fe_model = fe

        else:
            fe_c_arr = fe_model.transform(c_df_to_int_processed.values)

            print(fe_c_arr.shape)
            # print(fe_c_arr)

            fe_c_df_tmp = pd.DataFrame(fe_c_arr)
            fe_c_df_tmp.columns = fe_c_df_tmp.columns.map(lambda a: 'fe_' + str(a))

            fe_cols_selected = [col for col in fe_c_df_tmp if col in cols_selected]

            fe_c_df = fe_c_df_tmp[fe_cols_selected]

            del fe_c_df_tmp
            del fe_c_arr

            gc.collect()

            print(fe_c_df.shape)
            # print(fe_c_df)




    mv_df_to_label_processed = None
    mv_df_to_label_processed_2 = None
    mv_df_to_label_processed_1 = None
    if flag_mv:
        mv_cols = [col for col in mv_df if 'c_' not in col]
        print(mv_cols)
        print("^" * 20)
        mv_df_to_label_processed = label_encode_mv_as_cat_mulprocess(mv_df[mv_cols])
        mv_df_to_label_processed_2 = label_encode_mv_as_cat_mulprocess_2(mv_df[mv_cols])
        mv_df_to_label_processed_1 = label_encode_mv_as_cat_mulprocess_1(mv_df[mv_cols])
        # print(mv_df_to_label_processed.columns.tolist())
        # print(mv_df_to_label_processed_2.columns.tolist())
        # print(mv_df_to_label_processed_1.columns.tolist())

    # c_diff = ((c_df_to_int_processed.diff().sum(axis=1)) != 0).astype(int)
    # n_diff = ((n_df_processed.diff().sum(axis=1)) != 0).astype(int)
    # mv_diff = ((mv_df_to_label_processed.diff().sum(axis=1)) != 0).astype(int)
    #
    # diff = pd.DataFrame({'diff': c_diff * n_diff * mv_diff})
    #
    # print(diff)

    # size_info_mv_df = (mv_df_to_label_processed.memory_usage().sum(), len(mv_df_to_label_processed.columns.tolist()))

    # size_info_df = (size_info_df[0] + size_info_t_df[0] + size_info_n_df[0] + size_info_c_df[0] + size_info_mv_df[0],
    #                 size_info_df[1] + size_info_t_df[1] + size_info_n_df[1] + size_info_c_df[1] + size_info_mv_df[1])

    # memory_threshold = 12 * 1024 * 1024 * 1024 - 2 * big_df_memory
    memory_threshold = 16 * 1024 * 1024 * 1024 - 2 * big_df_memory

    max_numb_cols_to_select = 0
    if big_df_len != 0:
        if size_info_c_df is not None:
            max_numb_cols_to_select = memory_threshold / (size_info_c_df[0] * (big_df_len / small_df_len)) * \
                                      size_info_c_df[1]
            max_numb_cols_to_select = int(max_numb_cols_to_select / 10)
        elif size_info_n_df is not None:
            max_numb_cols_to_select = memory_threshold / (size_info_n_df[0] * (big_df_len / small_df_len)) * \
                                      size_info_n_df[1]
            max_numb_cols_to_select = int(max_numb_cols_to_select / 10)
        else:
            max_numb_cols_to_select = 150

    print("#" * 20)
    # print(size_info_t_df, size_info_n_df, size_info_c_df, size_info_mv_df)
    print(size_info_df)
    print(memory_threshold, "memory_threshold")
    print(big_df_memory, 'big_df_memory')
    print(big_df_len, 'big_df_len')
    print(small_df_len, 'small_df_len')
    print(max_numb_cols_to_select, 'max_numb_cols_to_select')
    print("#" * 20)

    # del t_df
    del n_df
    del c_df
    # del mv_df
    gc.collect()

    # df_processed = pd.concat([t_df_processed, n_df_processed, c_df_processed, mv_df_processed], axis=1)

    # 2.feature_engineer

    # 2.1 Categorical feature engineering
    # one order count
    # two roder count
    # group count

    # 2.2 mv feature_engineer
    # countvectorizer
    # tfidfvectorizer

    t_c_mv_processed = pd.concat(
        [t_df_to_int_processed, c_df_to_int_processed, mv_df_to_label_processed, mv_df_to_label_processed_2,
         mv_df_to_label_processed_1], axis=1, sort=False)

    del mv_df_to_label_processed
    del mv_df_to_label_processed_2
    del mv_df_to_label_processed_1

    gc.collect()

    # c_one_order_cnt, c_one_order_cols = generate_one_order_cnt_cat_feature(t_c_mv_processed, c_one_order_cols, y, 0)
    # c_one_order_cnt, c_one_order_cols = generate_one_order_cnt_cat_feature_add_select(t_c_mv_processed, c_one_order_cols, y, 0, cols_selected)
    c_one_order_cnt = generate_one_order_cnt_cat_feature_mulp_new_add_select(t_c_mv_processed, cols_selected)

    del t_c_mv_processed
    gc.collect()
    if c_one_order_cnt is not None:
        print(c_one_order_cnt.columns.tolist())

    c_two_order_cnt = None
    c_two_order_group_cnt = None
    c_two_order_group_mode = None
    c_two_order_group_n = None
    c_three_order_key_cnt = None
    c_two_order_labeled_df = None
    c_tree_order_labeled_df = None
    c_two_order_all_cnt = None
    c_two_order_same_table_cnt = None
    c_two_order_diff_table_label_cnt_df = None
    c_two_order_main_table_label_cnt_df = None
    c_two_order_cnt_2 = None
    c_two_order_cnt_3 = None

    len_table_c_cols = []
    for table_name, value in config['tables'].items():
        c_cols_list_tmp = [col for col in value['type'].keys()
                           if 'c_' in col]
        len_table_c_cols.append(len(c_cols_list_tmp))
    print(len_table_c_cols)

    if False:
    # if max(len_table_c_cols) < 0 and len(len_table_c_cols) > 0:

        # c_two_order_labeled_df = label_two_order_cat_key_feature_mulp_add_select(c_df_to_int_processed, config, None)
        #
        # if c_two_order_labeled_df is not None:
        #     print(c_two_order_labeled_df.columns.tolist())
        #     print(c_two_order_labeled_df.head())
        #
        # c_two_order_cnt_2, c_one_order_cols_2 = generate_one_order_cnt_cat_feature_mulp_new_add_select(
        #     c_two_order_labeled_df,
        #     c_one_order_cols,
        #     cols_selected)

        c_cols_tmp = [col for col in c_df_to_int_processed if 'c_0' not in col]
        c_two_order_diff_table_label_cnt_df = cnt_labeled_two_order_cat_diff_table_key_feature_mulp_add_select(
            c_df_to_int_processed[c_cols_tmp], config, cols_selected)

        if c_two_order_diff_table_label_cnt_df is not None:
            print(c_two_order_diff_table_label_cnt_df.columns.tolist())

        c_two_order_main_table_label_cnt_df = cnt_labeled_two_order_cat_main_table_key_feature_mulp_add_select(
            c_df_to_int_processed[c_cols_tmp], config, cols_selected)

        if c_two_order_main_table_label_cnt_df is not None:
            print(c_two_order_main_table_label_cnt_df.columns.tolist())

        # c_two_order_cnt_3, c_one_order_cols_3 = generate_one_order_cnt_cat_feature_mulp_new_add_select(
        #     c_two_order_diff_table_labeled_df,
        #     c_one_order_cols,
        #     cols_selected)
        #
        # if c_two_order_cnt_3 is not None:
        #     print(c_two_order_cnt_3.columns.tolist())

        # del c_two_order_labeled_df
        # del c_two_order_diff_table_labeled_df
        # gc.collect()

    # c_two_order_cnt = generate_two_order_cnt_cat_key_feature(c_df_to_int_processed, c_two_order_cols)
    # c_two_order_cnt = generate_two_order_cnt_cat_key_feature_mulp(c_df_to_int_processed)
    # c_two_order_cnt = generate_two_order_cnt_cat_key_feature_mulp_new(c_df_to_int_processed, config)

    if c_df_to_int_processed is not None:
        c_two_order_cnt = generate_two_order_cnt_cat_key_feature_mulp_new_add_select(c_df_to_int_processed, config,
                                                                                     cols_selected)
        c_two_order_group_cnt = generate_two_order_grpcnt_cat_key_feature_mulp_new_add_select(c_df_to_int_processed,
                                                                                              config, cols_selected)

    if c_two_order_cnt is not None:
        print(c_two_order_cnt.columns.tolist())

    # c_two_order_all_cnt = generate_two_order_cnt_cat_all_comb_feature_mulp_new_add_select(c_df_to_int_processed, config, cols_selected)
    # if c_two_order_all_cnt is not None:
    #     print(c_two_order_all_cnt.columns.tolist())
    #     print(c_two_order_all_cnt.head())

    # c_two_order_same_table_cnt = generate_two_order_cnt_cat_same_table_comb_feature_mulp_new_add_select(c_df_to_int_processed, config, cols_selected)
    #
    # if c_two_order_same_table_cnt is not None:
    #     print(c_two_order_same_table_cnt.columns.tolist())
    #     print(c_two_order_same_table_cnt.head())

    # c_two_order_group_cnt = generate_two_order_grpcnt_cat_key_feature(c_df_to_int_processed)
    # c_two_order_group_cnt = generate_two_order_grpcnt_cat_key_feature_mulp(c_df_to_int_processed)
    # c_two_order_group_cnt = generate_two_order_grpcnt_cat_key_feature_mulp_new(c_df_to_int_processed, config)
    if c_two_order_group_cnt is not None:
        print(c_two_order_group_cnt.columns.tolist())

    # c_two_order_group_mode = generate_two_order_grpmode_cat_key_feature_mulp_new_add_select(c_df_to_int_processed, config, cols_selected)
    #
    # print(c_two_order_group_mode.columns.tolist())

    # c_two_order_group_n = generate_two_order_grp_cat_key_n_feature(c_df_to_int_processed, n_df_processed)
    # c_two_order_group_n = generate_two_order_grp_cat_key_n_feature_mulp(c_df_to_int_processed, n_df_processed)
    # c_two_order_group_n = generate_two_order_grp_cat_key_n_feature_mulp_new(c_df_to_int_processed, n_df_processed, config)
    # print(c_two_order_group_n.columns.tolist())
    if c_df_to_int_processed is not None and n_df_processed is not None:
        c_two_order_group_n = generate_two_order_grp_cat_key_n_feature_mulp_new_add_select(c_df_to_int_processed,
                                                                                           n_df_processed, config,
                                                                                           cols_selected)

    if c_two_order_group_n is not None:
        print(c_two_order_group_n.columns.tolist())
        clean_df(c_two_order_group_n)
    if cols_selected is not None:
        print(cols_selected)
        print([col for col in cols_selected if 'MEAN' in col
               or 'SUM' in col
               or 'MAX' in col
               or 'MIN' in col
               or 'VAR' in col])

    # c_tree_order_labeled_df = label_three_order_cat_key_feature(c_df_to_int_processed, t_df_to_int_processed, config, cols_selected)

    # if c_tree_order_labeled_df is not None:
    #     print(c_tree_order_labeled_df.columns.tolist())
    #
    #     print(c_tree_order_labeled_df.head())

    # if c_df_to_int_processed is not None and t_df_to_int_processed is not None:
    #     c_three_order_key_cnt = generate_three_order_cnt_cat_key_feature(c_df_to_int_processed, t_df_to_int_processed,
    #                                                                      config, cols_selected)
    # if c_three_order_key_cnt is not None:
    #     print(c_three_order_key_cnt.columns.tolist())

    # generate_two_order_grp_cat_key_n_feature_mulp_new(c_df_to_int_processed, n_df_processed, config)

    # c_two_order_cnt, c_two_order_cols = generate_two_order_cnt_cat_feature(c_df_to_int_processed, c_two_order_cols, y, 0.01)
    # c_two_order_group_cnt, c_two_order_group_cnt_cols = generate_two_order_group_cnt_cat_feature(c_df_to_int_processed,
    #                                                                                              c_two_order_group_cnt_cols,
    #                                                                                              y, 0.01)

    # not used yet
    # c_two_order_n_groupby_cat, c_two_order_n_groupby_cat_cols = generate_two_order_n_groupby_cat_feature(c_df_to_int_processed,
    #                                                                                                      n_df_processed,
    #                                                                                                      c_two_order_n_groupby_cat_cols,
    #                                                                                                      y)

    # print(c_one_order_cnt.head(), "c_one_order_cnt+++++++++++++++++++++")
    # print(c_one_order_cols, "c_one_order_cols+++++++++++++++++++++")
    # print(c_two_order_cnt.head(), "c_two_order_cnt.head()+++++++++++++++++")
    # print(c_two_order_cols, "c_two_order_cols+++++++++++++++++++++")
    # print(c_two_order_group_cnt.head(), "c_two_order_group_cnt.head()+++++++++++++++++")
    # print(c_two_order_group_cnt_cols, "c_two_order_group_cnt_cols+++++++++++++++++++++")

    # cat_processed = pd.concat(
    #     [c_df_to_int_processed, c_two_order_cnt_3, c_one_order_cnt, c_two_order_cnt_2, c_two_order_cnt,
    #      c_two_order_diff_table_label_cnt_df, c_two_order_main_table_label_cnt_df, c_two_order_all_cnt,
    #      c_two_order_same_table_cnt, c_two_order_group_cnt, c_two_order_group_mode, c_two_order_group_n,
    #      c_tree_order_labeled_df, c_three_order_key_cnt], axis=1,
    #     sort=False)

    cat_processed = pd.concat(
        [c_df_to_int_processed, c_one_order_cnt,
         fe_c_df, c_two_order_cnt,
         c_two_order_group_cnt, c_two_order_group_n,
         c_three_order_key_cnt], axis=1,
        sort=False)

    del c_df_to_int_processed
    del c_one_order_cnt
    del c_two_order_all_cnt
    del c_two_order_same_table_cnt
    del c_two_order_cnt
    del c_two_order_group_cnt
    del c_two_order_group_n
    del c_tree_order_labeled_df
    del c_three_order_key_cnt
    del c_two_order_cnt_2
    del fe_c_df
    del c_two_order_cnt_3
    del c_two_order_diff_table_label_cnt_df
    del c_two_order_main_table_label_cnt_df

    gc.collect()

    mv_length = None
    if flag_mv:
        mv_length = generate_length_of_mv_feature(mv_df)
        # print(mv_length.head())

    del mv_df
    gc.collect()

    mv_two_order_cnt = None
    mv2_two_order_cnt = None
    mv1_two_order_cnt = None

    # mv_df_to_label_processed = pd.concat([mv_df_to_label_processed, mv_df_to_label_processed_2, mv_df_to_label_processed_1], axis=1, sort=False)

    # mv_two_order_cnt = generate_two_order_cnt_mv_feature_mulp_new_add_select(mv_df_to_label_processed, config, cols_selected)
    # if mv_two_order_cnt is not None:
    #     print(mv_two_order_cnt)
    #     print(mv_two_order_cnt.columns.tolist())

    # mv2_two_order_cnt = generate_two_order_cnt_mv_feature_mulp_new_add_select(mv_df_to_label_processed_2, config, cols_selected)
    # if mv2_two_order_cnt is not None:
    #     print(mv2_two_order_cnt)
    #     print(mv2_two_order_cnt.columns.tolist())
    #
    # mv1_two_order_cnt = generate_two_order_cnt_mv_feature_mulp_new_add_select(mv_df_to_label_processed_1, config, cols_selected)
    # if mv1_two_order_cnt is not None:
    #     print(mv1_two_order_cnt)
    #     print(mv1_two_order_cnt.columns.tolist())

    mv_processed = None
    if flag_mv:
        mv_processed = pd.concat([mv_length, mv_two_order_cnt, mv2_two_order_cnt, mv1_two_order_cnt], axis=1,
                                 sort=False)
    # print(len(t_df_processed.index), len(mv_df_processed.index), len(cat_processed.index), len(n_df_processed.index),
    #       "len of dfs ++++++++++++++++++++++++++++++++++++++++++")
    #
    # print(t_df_processed.head(), mv_df_processed.head(), cat_processed.head(), n_df_processed.head(),
    #       "len of dfs ++++++++++++++++++++++++++++++++++++++++++")
    #
    # print(t_df_processed.shape, mv_df_processed.shape, cat_processed.shape, n_df_processed.shape,
    #       "len of dfs ++++++++++++++++++++++++++++++++++++++++++")

    del mv_length
    del mv_two_order_cnt
    gc.collect()

    X_processed = pd.concat(
        [n_tmporal_df, t_df_to_int_processed, t_minus_t_df, n_df_processed, cat_processed, mv_processed], axis=1,
        sort=False)
    print(X_processed.columns.tolist())
    print(X_processed.shape)

    X_processed.index = idx

    # del c_one_order_cnt
    # del c_two_order_cnt
    # del c_two_order_group_cnt
    # del c_two_order_group_n



    del n_tmporal_df
    del t_df_to_int_processed
    del t_minus_t_df
    del n_df_processed
    del cat_processed
    del mv_processed
    # del mv_df_to_label_processed
    # del mv_df_to_label_processed_1
    # del mv_df_to_label_processed_2

    gc.collect()

    # two_order_cols, two_group_cols= transform_categorical_hash(df, y, two_order_cols, two_group_cols)
    # transform_datetime(df, config)

    # df = transform_numerical(df, config)
    # print(list(df.columns), "&&&&&&&&&&&")
    return X_processed, \
           two_order_cols, \
           two_group_cols, \
           mv_encs, \
           c_one_order_cols, \
           c_two_order_cols, \
           c_two_order_group_cnt_cols, \
           c_two_order_n_groupby_cat_cols, \
           n_minus_mean_cols, \
           max_numb_cols_to_select, \
           fe_model


def get_suitable_time_window_size(time_col):
    return 10


@timeit
def temporal_feature_engineer(n_df, time_col_name, cols_selected=None):

    if cols_selected is None:

        print(n_df.shape)
        n_df.sort_values(time_col_name, inplace=True)
        window_size = get_suitable_time_window_size(n_df[time_col_name])
        n_df.drop(time_col_name, axis=1, inplace=True)
        n_df_diff_1 = generate_difference_of_num_feature(n_df, 1)
        n_df_diff_2 = generate_difference_of_num_feature(n_df, 2)
        # n_df_rolling = gernerate_rolling_of_num_feature(n_df, window_size)
        # print(n_df_diff_1.shape)
        # print(n_df_diff_2.shape)
        # print(n_df_rolling.shape)
        n_df_temporal = pd.concat([n_df_diff_1, n_df_diff_2], axis=1)
        print(n_df_temporal.shape)

        del n_df
        del n_df_diff_1
        del n_df_diff_2
        gc.collect()

        n_df_temporal.fillna(0, inplace=True)
        return n_df_temporal

    else:

        n_df_cols = [col for col in n_df if f'{col}_temporal_diff_1' in cols_selected
                     or f'{col}_temporal_diff_2' in cols_selected]

        print(n_df.shape)
        n_df.sort_values(time_col_name, inplace=True)
        window_size = get_suitable_time_window_size(n_df[time_col_name])
        n_df.drop(time_col_name, axis=1, inplace=True)
        n_df_diff_1 = generate_difference_of_num_feature(n_df[n_df_cols], 1)
        n_df_diff_2 = generate_difference_of_num_feature(n_df[n_df_cols], 2)
        # n_df_rolling = gernerate_rolling_of_num_feature(n_df, window_size)
        # print(n_df_diff_1.shape)
        # print(n_df_diff_2.shape)
        # print(n_df_rolling.shape)
        n_df_temporal = pd.concat([n_df_diff_1, n_df_diff_2], axis=1)
        print(n_df_temporal.shape)

        del n_df
        del n_df_diff_1
        del n_df_diff_2
        gc.collect()

        n_df_temporal.fillna(0, inplace=True)
        return n_df_temporal



def gernerate_rolling_of_num_feature(n_df, window_size):
    n_df_rolling = n_df.rolling(window_size, min_periods=1)
    n_cols = n_df.columns.tolist()
    n_df_rolling_mean = n_df_rolling.mean()
    n_df_rolling_mean.columns = [f'{n}_temporal_mmeeaann' for n in n_cols]
    n_df_rolling_max = n_df_rolling.max()
    n_df_rolling_max.columns = [f'{n}_temporal_mmaaxx' for n in n_cols]
    n_df_rolling_min = n_df_rolling.min()
    n_df_rolling_min.columns = [f'{n}_temporal_mmiinn' for n in n_cols]

    del n_df
    gc.collect()
    return pd.concat([n_df_rolling_mean, n_df_rolling_max, n_df_rolling_min], axis=1)


def generate_difference_of_num_feature(n_df, n_diff_order):
    n_cols = n_df.columns.tolist()
    n_difference_cols = [f'{n}_temporal_diff_{n_diff_order}' for n in n_cols]
    n_diff = n_df[n_cols].diff(n_diff_order)
    n_diff.columns = n_difference_cols

    del n_df
    gc.collect()
    return n_diff


# from datetime import timedelta
#
# def get_timespan(time_col, start_time, time_delta):
#   # 利用time_delta，选取一个window
#   return time_col[(time_col <= start_time) & (time_col >= start_time - time_delta)].index
#
#
# def get_suitable_time_delta(time_col):
#   return timedelta(minutes=2)
#
#
# def generate_aggregation_feature(n_df, time_delta):
#   time_col = n_df['t_01']
#   n_df.drop(['t_01'], axis=1, inplace=True)
#   def my_sum(x):
#     row_index = x.size - 1
#     window_index = get_timespan(time_col, time_col.iloc[row_index], time_delta)
#     return x[window_index].sum()
#   return n_df.expanding(min_periods=1, axis=0).agg({'sum': my_sum})
#
# @timeit
# def temporal_feature_engineer(n_df):
#     n_df.sort_values('t_01', inplace=True)
#     time_delta = get_suitable_time_delta(n_df['t_01'])
#     n_df_diff_1 = generate_difference_of_num_feature(n_df, 1)
#     n_df_agg = generate_aggregation_feature(n_df, time_delta)
#     return pd.concat([n_df_diff_1, n_df_agg], axis=1, sort=False)
#
#
# def generate_difference_of_num_feature(n_df, n_diff_order):
#     n_cols = n_df.columns.tolist()
#     n_difference_cols = [f'{n}_diff_{n_diff_order}' for n in n_cols]
#     n_diff = n_df[n_cols].diff(n_diff_order)
#     n_diff.columns = n_difference_cols
#     return n_diff


@timeit
def cnt_labeled_two_order_cat_main_table_key_feature_mulp_add_select(c_df, config, cols_selected):
    if cols_selected is None:

        table_pair = []

        for relation in config['relations']:
            table_pair.append((relation['table_A'], relation['table_B']))

        table_pair = sorted(set(table_pair))

        print(table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        # for pair in table_pair:
        #     if 'main' in pair[0]:
        #         c_cols_xtable_1 = [col for col in c_cols if 'table' not in col]
        #         c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
        #         matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
        #     # elif pair[0] in c_cols:
        #     #
        #     #     c_cols_xtable = [col for col in c_cols if pair[1] in col]
        #     #     matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #
        #     else:
        #         c_cols_xtable_1 = [col for col in c_cols if pair[0] in col and 'last' not in col]
        #         c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
        #         matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
        #         # key = [col for col in c_cols if pair[0] in col]
        #         # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
        #         # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        matched_cols = list(combinations([col for col in c_cols if 'table' not in col], 2))

        print(matched_cols)
        sample_num = min(len(matched_cols), 20)
        matched_cols = random.sample(matched_cols, sample_num)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []
        two_order_c_encs = {}

        for res in result:
            r = res.get()
            # two_order_c_encs[r[1]] = r[3]
            tups.append((r[0], r[1]))

        tups.sort(key=lambda a: a[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:

            del c_df
            del tups
            gc.collect()

            return pd.concat(stck, axis=1, sort=False)
        else:
            return None

    else:

        table_pair = []

        for relation in config['relations']:
            table_pair.append((relation['table_A'], relation['table_B']))

        table_pair = sorted(set(table_pair))

        print(table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        # for pair in table_pair:
        #     if 'main' in pair[0]:
        #         c_cols_xtable_1 = [col for col in c_cols if 'table' not in col]
        #         c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
        #         matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
        #     # elif pair[0] in c_cols:
        #     #
        #     #     c_cols_xtable = [col for col in c_cols if pair[1] in col]
        #     #     matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #
        #     else:
        #         c_cols_xtable_1 = [col for col in c_cols if pair[0] in col and 'last' not in col]
        #         c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
        #         matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
        #         # key = [col for col in c_cols if pair[0] in col]
        #         # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
        #         # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        matched_cols = list(combinations([col for col in c_cols if 'table' not in col], 2))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_count' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]
                # enc = two_order_c_encs['-'.join(col)]

                abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []
        two_order_c_encs = {}

        for res in result:
            r = res.get()
            # two_order_c_encs[r[1]] = r[3]
            tups.append((r[0], r[1]))

        tups.sort(key=lambda a: a[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:

            del c_df
            del tups
            gc.collect()

            return pd.concat(stck, axis=1, sort=False)
        else:
            return None


@timeit
def cnt_labeled_two_order_cat_diff_table_key_feature_mulp_add_select(c_df, config, cols_selected):
    if cols_selected is None:

        table_pair = []

        for relation in config['relations']:
            table_pair.append((relation['table_A'], relation['table_B']))

        table_pair = sorted(set(table_pair))

        print(table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in table_pair:
            if 'main' in pair[0]:
                c_cols_xtable_1 = [col for col in c_cols if 'table' not in col]
                c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
                matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
            # elif pair[0] in c_cols:
            #
            #     c_cols_xtable = [col for col in c_cols if pair[1] in col]
            #     matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                c_cols_xtable_1 = [col for col in c_cols if pair[0] in col and 'last' not in col]
                c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
                matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)
        sample_num = min(len(matched_cols), 20)
        matched_cols = random.sample(matched_cols, sample_num)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []
        two_order_c_encs = {}

        for res in result:
            r = res.get()
            # two_order_c_encs[r[1]] = r[3]
            tups.append((r[0], r[1]))

        tups.sort(key=lambda a: a[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:

            del c_df
            del tups
            gc.collect()

            return pd.concat(stck, axis=1, sort=False)
        else:
            return None

    else:

        table_pair = []

        for relation in config['relations']:
            table_pair.append((relation['table_A'], relation['table_B']))

        table_pair = sorted(set(table_pair))

        print(table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in table_pair:
            if 'main' in pair[0]:
                c_cols_xtable_1 = [col for col in c_cols if 'table' not in col]
                c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
                matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
            # elif pair[0] in c_cols:
            #
            #     c_cols_xtable = [col for col in c_cols if pair[1] in col]
            #     matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                c_cols_xtable_1 = [col for col in c_cols if pair[0] in col and 'last' not in col]
                c_cols_xtable_2 = [col for col in c_cols if pair[1] in col and 'last' not in col]
                matched_cols = matched_cols + list(product(c_cols_xtable_1, c_cols_xtable_2))
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_count' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]
                # enc = two_order_c_encs['-'.join(col)]

                abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []
        two_order_c_encs = {}

        for res in result:
            r = res.get()
            # two_order_c_encs[r[1]] = r[3]
            tups.append((r[0], r[1]))

        tups.sort(key=lambda a: a[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:

            del c_df
            del tups
            gc.collect()

            return pd.concat(stck, axis=1, sort=False)
        else:
            return None


@timeit
def generate_two_order_cnt_cat_same_table_comb_feature_mulp_new_add_select(c_df, config, cols_selected):
    # cols_name_c = [col for col in c_df.columns.tolist() if 'c_0' not in col]

    if cols_selected is None:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                # matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
                matched_cols = matched_cols + list(combinations(c_cols_xtable, 2))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(combinations(c_cols_xtable, 2))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # matched_cols = list(combinations(cols_name_c, 2))
        #
        # indices = list(range(len(matched_cols)))
        #
        # id_pair = list(zip(indices, matched_cols))
        #
        # print(id_pair)

        # key_table_pair = []
        #
        # for relation in config['relations']:
        #     key_table_pair.append((relation['key'][0], relation['table_A']))
        #     key_table_pair.append((relation['key'][0], relation['table_B']))
        #
        # key_table_pair = sorted(set(key_table_pair))
        #
        # print(key_table_pair)
        #
        # c_cols = c_df.columns.tolist()
        #
        # matched_cols = []
        #
        # for pair in key_table_pair:
        #     if 'main' in pair[1]:
        #         c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #     elif pair[0] in c_cols:
        #
        #         c_cols_xtable = [col for col in c_cols if pair[1] in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #
        #     else:
        #         pass
        #         # key = [col for col in c_cols if pair[0] in col]
        #         # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
        #         # matched_cols = matched_cols + list(product(key, c_cols_xtable))
        #
        # print(matched_cols)
        #
        # indices = list(range(len(matched_cols)))
        #
        # id_pair = list(zip(indices, matched_cols))
        #
        # print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                # matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
                matched_cols = matched_cols + list(combinations(c_cols_xtable, 2))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(combinations(c_cols_xtable, 2))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # key_table_pair = []
        #
        # for relation in config['relations']:
        #     key_table_pair.append((relation['key'][0], relation['table_A']))
        #     key_table_pair.append((relation['key'][0], relation['table_B']))
        #
        # key_table_pair = sorted(set(key_table_pair))
        #
        # print(key_table_pair)
        #
        # c_cols = c_df.columns.tolist()
        #
        # matched_cols = []
        #
        # for pair in key_table_pair:
        #     if 'main' in pair[1]:
        #         c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #     elif pair[0] in c_cols:
        #
        #         c_cols_xtable = [col for col in c_cols if pair[1] in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #
        #     else:
        #         pass
        #         # key = [col for col in c_cols if pair[0] in col]
        #         # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
        #         # matched_cols = matched_cols + list(product(key, c_cols_xtable))
        #
        # print(matched_cols)
        #
        # indices = list(range(len(matched_cols)))
        #
        # id_pair = list(zip(indices, matched_cols))
        #
        # print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_count' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        return pd.concat(result_dfs, axis=1, sort=False)


def mv_two_order_cnt_func(idx, cols, s1, s2):
    tmp = s1.astype(str) + ',' + s2.astype(str)
    s_1_2 = tmp.map(tmp.value_counts())
    s_1_2_df = s_1_2.to_frame('-'.join(cols) + '_count')

    del tmp, s1, s2
    gc.collect()

    return (idx, s_1_2_df)


@timeit
def generate_two_order_cnt_mv_feature_mulp_new_add_select(mv_df_labeled, config, cols_selected):
    cols_name_mv = [col for col in mv_df_labeled.columns.tolist() if 'c_' not in col]
    # if len(cols_name_mv) == 0:
    #     cols_name_mv = [col for col in mv_df_labeled.columns.tolist()]

    if cols_selected is None:

        matched_cols = list(combinations(cols_name_mv, 2))

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = mv_df_labeled[col[0]]
            s2 = mv_df_labeled[col[1]]

            abortable_func = partial(abortable_worker, mv_two_order_cnt_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        if len(result_dfs) == 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        matched_cols = list(combinations(cols_name_mv, 2))

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_count' in cols_selected:
                s1 = mv_df_labeled[col[0]]
                s2 = mv_df_labeled[col[1]]

                abortable_func = partial(abortable_worker, mv_two_order_cnt_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        if len(result_dfs) == 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)


def label_two_order_cat_key_feature_func(idx, cols, s1, s2):
    tmp = s1.astype(str) + ',' + s2.astype(str)

    col_name = '-'.join(cols) + '_comb'

    df = pd.DataFrame({col_name: tmp})

    encoder = OrdinalEncoder(cols=[col_name])

    encoded_two_order_df = encoder.fit_transform(df)

    del tmp
    del df
    gc.collect()

    return (idx, col_name, encoded_two_order_df, encoder)


def label_two_order_cat_key_feature_func_predict(idx, cols, s1, s2, enc):
    tmp = s1.astype(str) + ',' + s2.astype(str)

    col_name = '-'.join(cols)

    df = pd.DataFrame({col_name: tmp})

    encoded_two_order_df = enc.fit_transform(df)

    del tmp
    del df
    gc.collect()

    return (idx, col_name, encoded_two_order_df, enc)


@timeit
def label_two_order_cat_key_feature_mulp_add_select(c_df, config, cols_selected):
    if cols_selected is None:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, label_two_order_cat_key_feature_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []
        two_order_c_encs = {}

        for res in result:
            r = res.get()
            two_order_c_encs[r[1]] = r[3]
            tups.append((r[0], r[2]))

        tups.sort(key=lambda a: a[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:

            del c_df
            del tups
            gc.collect()

            return pd.concat(stck, axis=1, sort=False)
        else:
            return None

    else:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_comb' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]
                # enc = two_order_c_encs['-'.join(col)]

                abortable_func = partial(abortable_worker, label_two_order_cat_key_feature_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []
        two_order_c_encs = {}

        for res in result:
            r = res.get()
            two_order_c_encs[r[1]] = r[3]
            tups.append((r[0], r[2]))

        tups.sort(key=lambda a: a[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:

            del c_df
            del tups
            gc.collect()

            return pd.concat(stck, axis=1, sort=False)
        else:
            return None


@timeit
def label_three_order_cat_key_feature(c_df, t_df, config, cols_selected):
    # _day_of_week_and_hour

    ## 存在可能的bug

    if cols_selected is None:

        # t_key = ['t_01_day_of_week_and_hour']
        t_keys = ['t_01_hour', 't_01_day_and_hour', 't_01_day_of_week_and_hour']
        c_keys = []
        for relation in config['relations']:
            if 'main' in relation['table_A']:
                c_keys.append(relation['key'][0])

        c_keys = sorted(set(c_keys))

        c_s_list = []

        for c_key in c_keys:
            c_s_list.append(c_df[c_key])

        t_c_s_list = []

        for t_key in t_keys:

            if t_key in t_df.columns.tolist():
                t_s = t_df[t_key].astype(str)

            for c_s in c_s_list:
                t_s = t_s + ',' + c_s.astype(str)

            # print(t_s)

            # t_c_s = t_s.map(t_s.value_counts())
            col_name = '-'.join([t_key] + c_keys) + '_comb'

            tmp_df = pd.DataFrame({col_name: t_s})

            encoder = OrdinalEncoder(cols=[col_name])

            encoded_two_order_df = encoder.fit_transform(tmp_df)

            t_c_s_list.append(encoded_two_order_df)

        t_c_s = pd.concat(t_c_s_list, axis=1, sort=False)

        # print(t_c_s)
        # print(t_c_s.nunique())

        del c_df
        del t_df
        del c_s_list
        del tmp_df
        del encoded_two_order_df
        del t_c_s_list
        gc.collect()

        return t_c_s

    else:

        # t_key = ['t_01_day_of_week_and_hour']
        t_keys = ['t_01_hour', 't_01_day_and_hour', 't_01_day_of_week_and_hour']
        c_keys = []
        for relation in config['relations']:
            if 'main' in relation['table_A']:
                c_keys.append(relation['key'][0])

        c_keys = sorted(set(c_keys))

        c_s_list = []

        for c_key in c_keys:
            c_s_list.append(c_df[c_key])

        t_c_s_list = []

        for t_key in t_keys:

            col_name = '-'.join([t_key] + c_keys) + '_comb'

            if col_name in cols_selected:

                if t_key in t_df.columns.tolist():
                    t_s = t_df[t_key].astype(str)

                for c_s in c_s_list:
                    t_s = t_s + ',' + c_s.astype(str)

                # print(t_s)

                # t_c_s = t_s.map(t_s.value_counts())

                tmp_df = pd.DataFrame({col_name: t_s})

                encoder = OrdinalEncoder(cols=[col_name])

                encoded_two_order_df = encoder.fit_transform(tmp_df)

                t_c_s_list.append(encoded_two_order_df)

        if len(t_c_s_list) == 0:
            return None

        t_c_s = pd.concat(t_c_s_list, axis=1, sort=False)

        # print(t_c_s)
        # print(t_c_s.nunique())

        del c_df
        del t_df
        del c_s_list
        del tmp_df
        del encoded_two_order_df
        del t_c_s_list
        gc.collect()

        return t_c_s


@timeit
def generate_three_order_cnt_cat_key_feature(c_df, t_df, config, cols_selected):
    # _day_of_week_and_hour

    ## 存在可能的bug

    # t_key = ['t_01_day_of_week_and_hour']
    t_keys = ['t_01_hour', 't_01_day_and_hour', 't_01_day_of_week_and_hour']

    t_keys = [col for col in t_df if 't_0' in col and 'hour' in col]

    c_keys = []
    for relation in config['relations']:
        if 'main' in relation['table_A']:
            c_keys.append(relation['key'][0])

    c_keys = sorted(set(c_keys))

    c_s_list = []

    for c_key in c_keys:
        c_s_list.append(c_df[c_key])

    t_c_s_list = []

    for t_key in t_keys:

        if t_key in t_df.columns.tolist():
            t_s = t_df[t_key].astype(str)

        for c_s in c_s_list:
            t_s = t_s + ',' + c_s.astype(str)

        # print(t_s)

        t_c_s = t_s.map(t_s.value_counts())

        t_c_s_list.append(t_c_s.to_frame('-'.join([t_key] + c_keys) + '_count'))

    if len(t_c_s_list) <= 0:
        del c_df
        del t_df
        del c_s_list
        del t_c_s_list
        gc.collect()

        return None

    t_c_s = pd.concat(t_c_s_list, axis=1, sort=False)

    # print(t_c_s)
    # print(t_c_s.nunique())

    del c_df
    del t_df
    del c_s_list
    del t_c_s_list
    gc.collect()

    return t_c_s


def c_two_order_grpcnt_key_n_func(idx, cols, s1, s2):
    df = pd.DataFrame({cols[0]: s1, cols[1]: s2})
    # tmp = df.groupby(key).agg({n_col: ['mean', 'sum', 'max', 'min']})
    tmp = df.groupby(cols[0]).agg({cols[1]: ['sum', 'mean', 'var', 'max', 'min']})

    tmp.columns = tmp.columns.map(lambda a: f"n_{cols[0]}_grp_{a[1].upper()}.{a[0]}")

    tmp.reset_index(inplace=True)
    # print(tmp.head())

    df = df.merge(tmp, how='left')
    # print(df.head())
    df.drop([cols[0], cols[1]], axis=1, inplace=True)

    del tmp, s1, s2
    gc.collect()

    return (idx, df)


def c_two_order_grpcnt_key_n_func_add_select(idx, cols, s1, s2, dict_col_ops):
    df = pd.DataFrame({cols[0]: s1, cols[1]: s2})
    # tmp = df.groupby(key).agg({n_col: ['mean', 'sum', 'max', 'min']})
    tmp = df.groupby(cols[0]).agg(dict_col_ops)

    tmp.columns = tmp.columns.map(lambda a: f"n_{cols[0]}_grp_{a[1].upper()}.{a[0]}")

    tmp.reset_index(inplace=True)
    # print(tmp.head())

    df = df.merge(tmp, how='left')
    # print(df.head())
    df.drop([cols[0], cols[1]], axis=1, inplace=True)

    del tmp, s1, s2
    gc.collect()

    return (idx, df)


@timeit
def generate_two_order_grp_cat_key_n_feature_mulp_new_add_select(c_df, n_df, config, cols_selected):
    if cols_selected is None:
        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        n_cols = n_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:

            if 'main' in pair[1]:

                n_cols_xtable = [col for col in n_cols if 'table' not in col]
                matched_cols = matched_cols + list(product([pair[0]], n_cols_xtable))

            elif pair[0] in c_cols:

                n_cols_xtable = [col for col in n_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], n_cols_xtable))
            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # n_cols_xtable = [col for col in n_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, n_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        #
        # n_cols = n_df.columns.tolist()
        #
        # key_pair = list(product(key_cols, n_cols))
        #
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = n_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_n_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        del c_df
        del n_df

        gc.collect()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del result
        del tups
        gc.collect()

        if len(result_dfs) <= 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        n_cols = n_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:

            if 'main' in pair[1]:

                n_cols_xtable = [col for col in n_cols if 'table' not in col]
                matched_cols = matched_cols + list(product([pair[0]], n_cols_xtable))

            elif pair[0] in c_cols:

                n_cols_xtable = [col for col in n_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], n_cols_xtable))
            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # n_cols_xtable = [col for col in n_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, n_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        #
        # n_cols = n_df.columns.tolist()
        #
        # key_pair = list(product(key_cols, n_cols))
        #
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        print(cols_selected)
        for idx, col in id_pair:

            dict_col_ops = {}
            ops = []
            flag = False

            # f"n_{cols[0]}_grp_{a[1].upper()}.{a[0]}"
            #
            # col_ops = [f"n_{col[0]}_grp_{op.upper()}.{col[1]}" for op in ['sum', 'mean', 'max', 'min', 'var']]
            #
            # col_ops = [op for ]

            if 'table' not in col[1]:
                for e in cols_selected:
                    if col[0] in e \
                            and col[1] in e \
                            and 'table' not in e \
                            and col[1] + '0' not in e \
                            and col[1] + '1' not in e \
                            and col[1] + '2' not in e \
                            and col[1] + '3' not in e \
                            and col[1] + '4' not in e \
                            and col[1] + '5' not in e \
                            and col[1] + '6' not in e \
                            and col[1] + '7' not in e \
                            and col[1] + '8' not in e \
                            and col[1] + '9' not in e \
                            :
                        print(col[0], col[1], e)
                        if 'SUM' in e and 'sum' not in ops:
                            ops.append('sum')
                        elif 'MEAN' in e and 'mean' not in ops:
                            ops.append('mean')
                        elif 'VAR' in e and 'var' not in ops:
                            ops.append('var')
                        elif 'MAX' in e and 'max' not in ops:
                            ops.append('max')
                        elif 'MIN' in e and 'min' not in ops:
                            ops.append('min')
            else:
                for e in cols_selected:
                    if col[0] in e \
                            and col[1] in e \
                            and col[1] + '0' not in e \
                            and col[1] + '1' not in e \
                            and col[1] + '2' not in e \
                            and col[1] + '3' not in e \
                            and col[1] + '4' not in e \
                            and col[1] + '5' not in e \
                            and col[1] + '6' not in e \
                            and col[1] + '7' not in e \
                            and col[1] + '8' not in e \
                            and col[1] + '9' not in e \
                            :
                        print(col[0], col[1], e)
                        if 'SUM' in e and 'sum' not in ops:
                            ops.append('sum')
                        elif 'MEAN' in e and 'mean' not in ops:
                            ops.append('mean')
                        elif 'VAR' in e and 'var' not in ops:
                            ops.append('var')
                        elif 'MAX' in e and 'max' not in ops:
                            ops.append('max')
                        elif 'MIN' in e and 'min' not in ops:
                            ops.append('min')

            print(ops)

            if len(ops) > 0:
                flag = True
            if flag:
                dict_col_ops[col[1]] = ops
                print(dict_col_ops)
                s1 = c_df[col[0]]
                s2 = n_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_n_func_add_select)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2, dict_col_ops))

                result.append(res)

        pool.close()
        pool.join()

        del c_df
        del n_df
        gc.collect()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del result
        del tups
        gc.collect()

        if len(result_dfs) <= 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_grp_cat_key_n_feature_mulp_new(c_df, n_df, config):
    key_table_pair = []

    for relation in config['relations']:
        key_table_pair.append((relation['key'][0], relation['table_A']))
        key_table_pair.append((relation['key'][0], relation['table_B']))

    key_table_pair = sorted(set(key_table_pair))

    print(key_table_pair)

    c_cols = c_df.columns.tolist()

    n_cols = n_df.columns.tolist()

    matched_cols = []

    for pair in key_table_pair:

        if 'main' in pair[1]:

            n_cols_xtable = [col for col in n_cols if 'table' not in col]
            matched_cols = matched_cols + list(product([pair[0]], n_cols_xtable))

        elif pair[0] in c_cols:

            n_cols_xtable = [col for col in n_cols if pair[1] in col]
            matched_cols = matched_cols + list(product([pair[0]], n_cols_xtable))
        else:
            pass
            # key = [col for col in c_cols if pair[0] in col]
            # n_cols_xtable = [col for col in n_cols if pair[1] in col and key[0] not in col]
            # matched_cols = matched_cols + list(product(key, n_cols_xtable))

    print(matched_cols)

    indices = list(range(len(matched_cols)))

    id_pair = list(zip(indices, matched_cols))

    print(id_pair)

    # c_cols = c_df.columns.tolist()
    # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
    #
    # n_cols = n_df.columns.tolist()
    #
    # key_pair = list(product(key_cols, n_cols))
    #
    # print(key_pair)
    #
    # indices = list(range(len(key_pair)))
    # id_pair = list(zip(indices, key_pair))
    #
    # print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []
    for idx, col in id_pair:
        s1 = c_df[col[0]]
        s2 = n_df[col[1]]

        abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_n_func)
        res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

        result.append(res)

    pool.close()
    pool.join()

    del c_df
    del n_df
    gc.collect()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    del result
    del tups
    gc.collect()

    return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_grp_cat_key_n_feature_mulp(c_df, n_df):
    c_cols = c_df.columns.tolist()
    key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]

    n_cols = n_df.columns.tolist()

    key_pair = list(product(key_cols, n_cols))

    print(key_pair)

    indices = list(range(len(key_pair)))
    id_pair = list(zip(indices, key_pair))

    print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []
    for idx, col in id_pair:
        s1 = c_df[col[0]]
        s2 = n_df[col[1]]

        abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_n_func)
        res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

        result.append(res)

    pool.close()
    pool.join()

    del c_df
    del n_df
    gc.collect()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    del result
    del tups
    gc.collect()

    return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_grp_cat_key_n_feature(c_df, n_df):
    c_cols = c_df.columns.tolist()
    key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]

    n_cols = n_df.columns.tolist()
    res = []
    for key in key_cols:

        gc.collect()

        for n_col in n_cols:
            df = pd.DataFrame({key: c_df[key], n_col: n_df[n_col]})
            # tmp = df.groupby(key).agg({n_col: ['mean', 'sum', 'max', 'min']})
            tmp = df.groupby(key).agg({n_col: ['mean']})

            tmp.columns = tmp.columns.map(lambda a: f"n_{key}_grp_{a[1].upper()}.{a[0]}")

            tmp.reset_index(inplace=True)
            # print(tmp.head())

            df = df.merge(tmp, how='left')
            # print(df.head())
            df.drop([key, n_col], axis=1, inplace=True)
            # print(df.head())

            res.append(df)

    c_two_order_group_n = pd.concat(res, axis=1, sort=False)

    print(c_two_order_group_n.columns.tolist())

    del res

    gc.collect()

    return c_two_order_group_n


def c_two_order_cnt_key_func(idx, cols, s1, s2):
    tmp = s1.astype(str) + ',' + s2.astype(str)
    s_1_2 = tmp.map(tmp.value_counts())
    s_1_2_df = s_1_2.to_frame('-'.join(cols) + '_count')

    del tmp, s1, s2
    gc.collect()

    return (idx, s_1_2_df)


@timeit
def generate_two_order_cnt_cat_all_comb_feature_mulp_new_add_select(c_df, config, cols_selected):
    cols_name_c = [col for col in c_df.columns.tolist() if 'c_0' not in col]

    if cols_selected is None:

        matched_cols = list(combinations(cols_name_c, 2))

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # key_table_pair = []
        #
        # for relation in config['relations']:
        #     key_table_pair.append((relation['key'][0], relation['table_A']))
        #     key_table_pair.append((relation['key'][0], relation['table_B']))
        #
        # key_table_pair = sorted(set(key_table_pair))
        #
        # print(key_table_pair)
        #
        # c_cols = c_df.columns.tolist()
        #
        # matched_cols = []
        #
        # for pair in key_table_pair:
        #     if 'main' in pair[1]:
        #         c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #     elif pair[0] in c_cols:
        #
        #         c_cols_xtable = [col for col in c_cols if pair[1] in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #
        #     else:
        #         pass
        #         # key = [col for col in c_cols if pair[0] in col]
        #         # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
        #         # matched_cols = matched_cols + list(product(key, c_cols_xtable))
        #
        # print(matched_cols)
        #
        # indices = list(range(len(matched_cols)))
        #
        # id_pair = list(zip(indices, matched_cols))
        #
        # print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        matched_cols = list(combinations(cols_name_c, 2))

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # key_table_pair = []
        #
        # for relation in config['relations']:
        #     key_table_pair.append((relation['key'][0], relation['table_A']))
        #     key_table_pair.append((relation['key'][0], relation['table_B']))
        #
        # key_table_pair = sorted(set(key_table_pair))
        #
        # print(key_table_pair)
        #
        # c_cols = c_df.columns.tolist()
        #
        # matched_cols = []
        #
        # for pair in key_table_pair:
        #     if 'main' in pair[1]:
        #         c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #     elif pair[0] in c_cols:
        #
        #         c_cols_xtable = [col for col in c_cols if pair[1] in col]
        #         matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        #
        #     else:
        #         pass
        #         # key = [col for col in c_cols if pair[0] in col]
        #         # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
        #         # matched_cols = matched_cols + list(product(key, c_cols_xtable))
        #
        # print(matched_cols)
        #
        # indices = list(range(len(matched_cols)))
        #
        # id_pair = list(zip(indices, matched_cols))
        #
        # print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_count' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_cnt_cat_key_feature_mulp_new_add_select(c_df, config, cols_selected):
    if cols_selected is None:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del c_df
        del tups
        del result
        gc.collect()

        if len(result_dfs) <= 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            if '-'.join(col) + '_count' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del c_df
        del tups
        del result
        gc.collect()

        if len(result_dfs) == 0:
            return None

        return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_cnt_cat_key_feature_mulp_new(c_df, config):
    key_table_pair = []

    for relation in config['relations']:
        key_table_pair.append((relation['key'][0], relation['table_A']))
        key_table_pair.append((relation['key'][0], relation['table_B']))

    key_table_pair = sorted(set(key_table_pair))

    print(key_table_pair)

    c_cols = c_df.columns.tolist()

    matched_cols = []

    for pair in key_table_pair:
        if 'main' in pair[1]:
            c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
            matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        elif pair[0] in c_cols:

            c_cols_xtable = [col for col in c_cols if pair[1] in col]
            matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

        else:
            pass
            # key = [col for col in c_cols if pair[0] in col]
            # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
            # matched_cols = matched_cols + list(product(key, c_cols_xtable))

    print(matched_cols)

    indices = list(range(len(matched_cols)))

    id_pair = list(zip(indices, matched_cols))

    print(id_pair)

    # c_cols = c_df.columns.tolist()
    #
    # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
    # no_key_cols = [col for col in c_cols if 'c_0' not in col]
    #
    # key_pair = list(product(key_cols, no_key_cols))
    # print(key_pair)
    #
    # indices = list(range(len(key_pair)))
    # id_pair = list(zip(indices, key_pair))
    #
    # print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []
    for idx, col in id_pair:
        s1 = c_df[col[0]]
        s2 = c_df[col[1]]

        abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
        res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

        result.append(res)

    pool.close()
    pool.join()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_cnt_cat_key_feature_mulp(c_df):
    c_cols = c_df.columns.tolist()

    key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
    no_key_cols = [col for col in c_cols if 'c_0' not in col]

    key_pair = list(product(key_cols, no_key_cols))
    print(key_pair)

    indices = list(range(len(key_pair)))
    id_pair = list(zip(indices, key_pair))

    print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []
    for idx, col in id_pair:
        s1 = c_df[col[0]]
        s2 = c_df[col[1]]

        abortable_func = partial(abortable_worker, c_two_order_cnt_key_func)
        res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

        result.append(res)

    pool.close()
    pool.join()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    return pd.concat(result_dfs, axis=1, sort=False)


from scipy import stats


def c_two_order_grpmode_key_func(idx, cols, s1, s2):
    df = pd.DataFrame({cols[0]: s1, cols[1]: s2})
    tmp = df.groupby(cols[0])[cols[1]].agg(lambda x: stats.mode(x)[0]).reset_index()

    # tmp = df.groupby(cols[0])[cols[1]].agg(lambda x: x.value_counts().index[0]).reset_index()
    tmp.columns = [cols[0], '-'.join(cols) + '-grpmode']
    df = df.merge(tmp, how='left')

    df.drop([cols[0], cols[1]], axis=1, inplace=True)
    # print(df.head(10))
    # print(df.dtypes)
    del tmp, s1, s2
    gc.collect()

    return (idx, df)


@timeit
def generate_two_order_grpmode_cat_key_feature_mulp_new_add_select(c_df, config, cols_selected):
    if cols_selected is None:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_grpmode_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        del c_df
        gc.collect()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del result
        del tups
        gc.collect()

        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            # flag_list = []
            # if 'table' not in col[1]:
            #     for e in cols_selected:
            #         if col[0] in e and col[1] in e and 'table' not in e:
            #             flag_list.append(col)
            # else:
            #     for e in cols_selected:
            #         if col[0] in e and col[1] in e:
            #             flag_list.append(col)
            #
            #

            if '-'.join(col) + '-grpmode' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_grpmode_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        del c_df
        gc.collect()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del result
        del tups
        gc.collect()

        return pd.concat(result_dfs, axis=1, sort=False)


def c_two_order_grpcnt_key_func(idx, cols, s1, s2):
    df = pd.DataFrame({cols[0]: s1, cols[1]: s2})
    tmp = df.groupby(cols[0])[cols[1]].nunique().reset_index()
    tmp.columns = [cols[0], '-'.join(cols) + '-grpcnt']
    df = df.merge(tmp, how='left')

    df.drop([cols[0], cols[1]], axis=1, inplace=True)

    del tmp, s1, s2
    gc.collect()

    return (idx, df)


@timeit
def generate_two_order_grpcnt_cat_key_feature_mulp_new_add_select(c_df, config, cols_selected):
    if cols_selected is None:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_func)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

            result.append(res)

        pool.close()
        pool.join()

        del c_df
        gc.collect()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del result
        del tups
        gc.collect()

        if len(result_dfs) <= 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)

    else:

        key_table_pair = []

        for relation in config['relations']:
            key_table_pair.append((relation['key'][0], relation['table_A']))
            key_table_pair.append((relation['key'][0], relation['table_B']))

        key_table_pair = sorted(set(key_table_pair))

        print(key_table_pair)

        c_cols = c_df.columns.tolist()

        matched_cols = []

        for pair in key_table_pair:
            if 'main' in pair[1]:
                c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
            elif pair[0] in c_cols:

                c_cols_xtable = [col for col in c_cols if pair[1] in col]
                matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

            else:
                pass
                # key = [col for col in c_cols if pair[0] in col]
                # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
                # matched_cols = matched_cols + list(product(key, c_cols_xtable))

        print(matched_cols)

        indices = list(range(len(matched_cols)))

        id_pair = list(zip(indices, matched_cols))

        print(id_pair)

        # c_cols = c_df.columns.tolist()
        #
        # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
        # no_key_cols = [col for col in c_cols if 'c_0' not in col]
        #
        # key_pair = list(product(key_cols, no_key_cols))
        # print(key_pair)
        #
        # indices = list(range(len(key_pair)))
        # id_pair = list(zip(indices, key_pair))
        #
        # print(id_pair)

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []
        for idx, col in id_pair:

            # flag_list = []
            # if 'table' not in col[1]:
            #     for e in cols_selected:
            #         if col[0] in e and col[1] in e and 'table' not in e:
            #             flag_list.append(col)
            # else:
            #     for e in cols_selected:
            #         if col[0] in e and col[1] in e:
            #             flag_list.append(col)
            #
            #

            if '-'.join(col) + '-grpcnt' in cols_selected:
                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_func)
                res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

                result.append(res)

        pool.close()
        pool.join()

        del c_df
        gc.collect()

        tups = []

        for res in result:
            r = res.get()
            tups.append(r)

        tups.sort(key=lambda a: a[0])

        result_dfs = []

        for r in tups:
            result_dfs.append(r[1])

        del result
        del tups
        gc.collect()
        if len(result_dfs) <= 0:
            return None
        return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_grpcnt_cat_key_feature_mulp_new(c_df, config):
    key_table_pair = []

    for relation in config['relations']:
        key_table_pair.append((relation['key'][0], relation['table_A']))
        key_table_pair.append((relation['key'][0], relation['table_B']))

    key_table_pair = sorted(set(key_table_pair))

    print(key_table_pair)

    c_cols = c_df.columns.tolist()

    matched_cols = []

    for pair in key_table_pair:
        if 'main' in pair[1]:
            c_cols_xtable = [col for col in c_cols if 'table' not in col and pair[0] not in col]
            matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))
        elif pair[0] in c_cols:

            c_cols_xtable = [col for col in c_cols if pair[1] in col]
            matched_cols = matched_cols + list(product([pair[0]], c_cols_xtable))

        else:
            pass
            # key = [col for col in c_cols if pair[0] in col]
            # c_cols_xtable = [col for col in c_cols if pair[1] in col and key[0] not in col]
            # matched_cols = matched_cols + list(product(key, c_cols_xtable))

    print(matched_cols)

    indices = list(range(len(matched_cols)))

    id_pair = list(zip(indices, matched_cols))

    print(id_pair)

    # c_cols = c_df.columns.tolist()
    #
    # key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
    # no_key_cols = [col for col in c_cols if 'c_0' not in col]
    #
    # key_pair = list(product(key_cols, no_key_cols))
    # print(key_pair)
    #
    # indices = list(range(len(key_pair)))
    # id_pair = list(zip(indices, key_pair))
    #
    # print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []
    for idx, col in id_pair:
        s1 = c_df[col[0]]
        s2 = c_df[col[1]]

        abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_func)
        res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

        result.append(res)

    pool.close()
    pool.join()

    del c_df
    gc.collect()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    del result
    del tups
    gc.collect()

    return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_grpcnt_cat_key_feature_mulp(c_df):
    c_cols = c_df.columns.tolist()

    key_cols = [col for col in c_cols if 'c_0' in col and 'table' not in col]
    no_key_cols = [col for col in c_cols if 'c_0' not in col]

    key_pair = list(product(key_cols, no_key_cols))
    print(key_pair)

    indices = list(range(len(key_pair)))
    id_pair = list(zip(indices, key_pair))

    print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []
    for idx, col in id_pair:
        s1 = c_df[col[0]]
        s2 = c_df[col[1]]

        abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_func)
        res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))

        result.append(res)

    pool.close()
    pool.join()

    del c_df
    gc.collect()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    del result
    del tups
    gc.collect()

    return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def generate_two_order_grpcnt_cat_key_feature(c_df):
    c_cols = c_df.columns.tolist()

    key_cols = [col for col in c_cols if 'c_0' in col]
    no_key_cols = [col for col in c_cols if 'c_0' not in col]

    res = []
    for key in key_cols:
        for other in no_key_cols:
            df = pd.DataFrame({key: c_df[key], other: c_df[other]})

            tmp = df.groupby(key)[other].nunique().reset_index()

            tmp.columns = [key, '-'.join([key, other]) + '-grpcnt']

            df = df.merge(tmp, how='left')
            df.drop([key, other], axis=1, inplace=True)

            res.append(df)

    c_two_order_grpcnt = pd.concat(res, axis=1, sort=False)

    print(c_two_order_grpcnt.columns.tolist())

    return c_two_order_grpcnt


@timeit
def generate_two_order_cnt_cat_key_feature(c_df, c_two_order_cols):
    c_cols = c_df.columns.tolist()

    key_cols = [col for col in c_cols if 'c_0' in col]
    no_key_cols = [col for col in c_cols if 'c_0' not in col]

    # tmp = c_df[key_cols[0]].astype(str) + '000' + c_df[key_cols[1]].astype(str)
    # s_1_2 = tmp.map(tmp.value_counts())
    # c_two_order_cnt_1 = s_1_2.to_frame('-'.join(key_cols) + '_count')
    #
    #
    #
    # tmp = c_df['c_table_2.c_2'].astype(str) + '000' + c_df['c_table_1.c_1'].astype(str)
    # s_1_2 = tmp.map(tmp.value_counts())
    # c_two_order_cnt_2 = s_1_2.to_frame('-'.join(['c_table_2.c_2', 'c_table_1.c_1']) + '_count')
    #
    # c_two_order_cnt = pd.concat([c_two_order_cnt_1, c_two_order_cnt_2], axis=1,sort=False)

    res = []
    for key in key_cols:
        for other in no_key_cols:
            tmp = c_df[key].astype(str) + '000' + c_df[other].astype(str)
            s_1_2 = tmp.map(tmp.value_counts())
            s_1_2_df = s_1_2.to_frame('-'.join([key, other]) + '_count')

            # encoder = MinMaxScaler(feature_range=(0, 100))
            # new_arr = encoder.fit_transform(s_1_2_df[['-'.join([key, other]) + '_count']])
            #
            # new_df = pd.DataFrame(new_arr)
            # new_df.columns = ['-'.join([key, other]) + '_count']

            res.append(s_1_2_df)

    c_two_order_cnt = pd.concat(res, axis=1, sort=False)

    return c_two_order_cnt


@timeit
def func_to_n(n_df, config):
    n_cols = n_df.columns.tolist()

    tables_name = [tables_name for tables_name, value in config['tables'].items()]

    dict_table_n = {}

    for table_name in tables_name:
        if 'main' in table_name:
            dict_table_n[table_name] = [col for col in n_cols if 'table' not in col]
        else:
            dict_table_n[table_name] = [col for col in n_cols if table_name in col]

    print(dict_table_n)
    for table_name, n_cols in dict_table_n.items():
        n_df[f'n_sum_{table_name}'] = n_df[n_cols].sum(axis=1)
        # n_df[f'n_mul_{table_name}'] = n_df[n_cols].prod(axis=1)
    # print(n_df.head(10))

    return n_df
    # return pd.concat([n_df, tmp_n_minus_mean], axis=1), n_minus_mean_cols

    # if y is not None:
    #
    #     n_cols = n_df.columns.tolist()
    #
    #     sampled_indices = sample_indices(y)
    #     metric_1 = gen_mutual_info(n_df[n_cols].iloc[sampled_indices, :], y[sampled_indices])
    #     metric_1_dict = dict(zip(n_cols, metric_1))
    #
    #
    #
    #     tmp_n_minus_mean = n_df - n_df.mean().values
    #
    #     tmp_n_minus_mean.columns = tmp_n_minus_mean.columns.map(lambda col: col+"-mean")
    #
    #
    #     metric_2 = gen_mutual_info(tmp_n_minus_mean[tmp_n_minus_mean.columns.tolist()].iloc[sampled_indices, :], y[sampled_indices])
    #     metric_2_dict = dict(zip(tmp_n_minus_mean.columns.tolist(), metric_2))
    #
    #     print(metric_1_dict)
    #     print(metric_2_dict)
    #
    #     n_minus_mean_cols = [e_2 > e_1 for e_1, e_2 in zip(metric_1, metric_2)]
    #
    #     print(tmp_n_minus_mean.columns.tolist())
    #
    #     print(n_minus_mean_cols)
    #
    #     tmp_n_minus_mean = tmp_n_minus_mean.loc[:, n_minus_mean_cols]
    #
    #     print(tmp_n_minus_mean.columns.tolist())
    #
    #     tables_name = [tables_name for tables_name, value in config['tables'].items()]
    #
    #     dict_table_n = {}
    #
    #     for table_name in tables_name:
    #         if 'main' in table_name:
    #             dict_table_n[table_name] = [col for col in n_cols if 'table' not in col]
    #         else:
    #             dict_table_n[table_name] = [col for col in n_cols if table_name in col]
    #
    #     print(dict_table_n)
    #     for table_name, n_cols in dict_table_n.items():
    #         n_df[f'n_sum_{table_name}'] = n_df[n_cols].sum(axis=1)
    #         # print(n_df[n_cols])
    #         # print(n_df[f'n_sum_{table_name}'])
    #     print(n_df.head(10))
    #     return pd.concat([n_df, tmp_n_minus_mean], axis=1), n_minus_mean_cols
    #
    #
    #     # list_n_cols = []
    #     #
    #     # for table_name, value in config['tables'].items():
    #     #     dict_n_cols = {}
    #     #     for col in value['type'].keys():
    #     #         if 'n_' in col:
    #     #             dict_n_cols[col] = table_name + ',' + col
    #     #     if(len(dict_n_cols) >= 1):
    #     #         list_n_cols.append(dict_n_cols)
    #     # print(list_n_cols)
    #     #
    #     # for t_n_cols_dict in list_n_cols:
    #     #     n_cols_pair = list(t_n_cols_dict.values())
    #     #     table_name = n_cols_pair[0].split(',')[0]
    #     #     n_cols_1 = n_cols_pair[0].split(',')[1]
    #     #
    #     #     if 'main' in table_name:
    # else:
    #
    #     n_cols = n_df.columns.tolist()
    #
    #     tmp_n_minus_mean = n_df - n_df.mean().values
    #
    #     tmp_n_minus_mean.columns = tmp_n_minus_mean.columns.map(lambda col: col+"-mean")
    #
    #     tmp_n_minus_mean = tmp_n_minus_mean.loc[:, n_minus_mean_cols]
    #
    #     tables_name = [tables_name for tables_name, value in config['tables'].items()]
    #
    #     dict_table_n = {}
    #
    #     for table_name in tables_name:
    #         if 'main' in table_name:
    #             dict_table_n[table_name] = [col for col in n_cols if 'table' not in col]
    #         else:
    #             dict_table_n[table_name] = [col for col in n_cols if table_name in col]
    #
    #     print(dict_table_n)
    #     for table_name, n_cols in dict_table_n.items():
    #         n_df[f'n_sum_{table_name}'] = n_df[n_cols].sum(axis=1)
    #         # print(n_df[n_cols])
    #         # print(n_df[f'n_sum_{table_name}'])
    #     print(n_df.head(10))
    #
    #     return pd.concat([n_df, tmp_n_minus_mean], axis=1), n_minus_mean_cols


@timeit
def generate_length_of_mv_feature(mv_df):
    mv_cols = mv_df.columns.tolist()
    mv_cols_len = [m + '_length' for m in mv_cols]
    mv_df[mv_cols_len] = mv_df[mv_cols].applymap(lambda a: len(a))
    mv_df.drop(mv_cols, axis=1, inplace=True)
    mv_df['mv_len_sum'] = mv_df.sum(axis=1)
    return mv_df


@timeit
def generate_two_order_n_groupby_cat_feature(c_df, n_df, c_two_order_n_groupby_cat_cols, y):
    return None, None


@timeit
def generate_two_order_group_cnt_cat_feature(c_df, c_two_order_group_cnt_cols, y, threshold):
    cols_name_c = c_df.columns.tolist()
    # cols_name_c = [col for col in cols_name_c if 'table' not in col]

    if y is not None:

        sampled_indices = sample_indices(y)
        metric = gen_mutual_info(c_df[cols_name_c].iloc[sampled_indices, :], y[sampled_indices])

        metric_dict = dict(zip(cols_name_c, metric))

        combs = list(combinations(cols_name_c, 2))

        two_order_value_dict = {}
        for cols in combs:
            two_order_value_dict[cols] = metric_dict[cols[0]] + metric_dict[cols[1]]
        sorted_two_order_value_dict = dict(sorted(two_order_value_dict.items(), key=lambda d: d[1]))

        print(sorted_two_order_value_dict)

        combs = list(sorted_two_order_value_dict.keys())

        indices = list(range(len(combs)))
        id_combs = list(zip(indices, combs))

        print(id_combs, "id and combs +++++++++++++++++++++++++++++++")

        copy_id_combs = copy.deepcopy(id_combs)
        # random.shuffle(copy_id_combs)
        # random.seed(1024)

        print(copy_id_combs, "copy_id_combs +++++++++++++++++++++++++")

        # sampled_indices = sample_indices(y)
        #
        # metric = gen_mutual_info(c_df[cols_name_c].iloc[sampled_indices, :], y[sampled_indices])
        #
        # metric_dict = dict(zip(cols_name_c, metric))
        #
        # combs = list(combinations(cols_name_c, 2))
        #
        # indices = list(range(len(combs)))
        #
        # id_combs = list(zip(indices, combs))
        #
        # print(id_combs, "id and combs +++++++++++++++++++++++++++++++")
        #
        # copy_id_combs = copy.deepcopy(id_combs)
        # random.shuffle(copy_id_combs)
        # random.seed(1024)

        batch_num = 8

        start = time.time()
        time_out = 40

        tups = []

        while time.time() - start < time_out:
            if not copy_id_combs:
                break

            small_batch = copy_id_combs[-batch_num:]

            del copy_id_combs[-batch_num:]

            print(small_batch, "small batch ###########################")

            ctx = mp.get_context('forkserver')

            pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

            result = []

            for idx, col in small_batch:
                base_metric_1 = metric_dict[col[0]]
                base_metric_2 = metric_dict[col[1]]
                max_metric = max(base_metric_1, base_metric_2)

                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_group_cnt_func)
                res = pool.apply_async(abortable_func,
                                       args=(idx, col, s1, s2, y, max_metric, sampled_indices, threshold))
                result.append(res)
                # print("end++++++++++++++++++++++++++")

            pool.close()
            pool.join()
            tup = []
            for res in result:
                r = res.get()
                tup.append(r)

            tup = [t for t in tup if t[2] is not None]
            tups = tups + tup
            elapsed = time.time()
            print(elapsed - start, "time elapse &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del s1
        del s2
        del result
        del tups
        gc.collect()

        print(len(selected_feat_dfs))
        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols

    else:

        indices = list(range(len(c_two_order_group_cnt_cols)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, c_two_order_group_cnt_cols))

        for idx, col in idx_cols:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_group_cnt_func_predict)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)
        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del s1
        del s2
        del result
        del tups

        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols


def c_two_order_group_cnt_func_predict(idx, cols, s1, s2):
    df = pd.DataFrame({cols[0]: s1, cols[1]: s2})
    tmp = df.groupby(cols[0])[cols[1]].nunique().reset_index()

    tmp.columns = [cols[0], '_'.join(cols) + '-grpcnt']

    df = df.merge(tmp, how='left')
    df.drop([cols[0], cols[1]], axis=1, inplace=True)

    encoder = MinMaxScaler(feature_range=(0, 100))
    new_arr = encoder.fit_transform(df[['_'.join(cols) + '-grpcnt']])

    new_df = pd.DataFrame(new_arr)
    new_df.columns = ['-'.join(cols) + '-grpcnt']

    del df
    del tmp
    del encoder
    del new_arr

    gc.collect()

    return (idx, cols, new_df)


def c_two_order_group_cnt_func(idx, cols, s1, s2, y, base_metric, sampled_indices, threshold):
    df1 = pd.DataFrame({cols[0]: s1, cols[1]: s2})

    tmp1 = df1.groupby(cols[0])[cols[1]].nunique().reset_index()

    tmp1.columns = [cols[0], '-'.join(cols) + '-grpcnt']

    df1 = df1.merge(tmp1, how='left')

    df1.drop([cols[0], cols[1]], axis=1, inplace=True)

    reverse_cols = [cols[1], cols[0]]

    df2 = pd.DataFrame({reverse_cols[0]: s2, reverse_cols[1]: s1})
    tmp2 = df2.groupby(reverse_cols[0])[reverse_cols[1]].nunique().reset_index()
    tmp2.columns = [reverse_cols[0], '-'.join(reverse_cols) + '-grpcnt']

    df2 = df2.merge(tmp2, how='left')
    df2.drop(reverse_cols, axis=1, inplace=True)

    metric_1 = gen_mutual_info(df1.iloc[sampled_indices, :], y[sampled_indices])[0]
    metric_2 = gen_mutual_info(df2.iloc[sampled_indices, :], y[sampled_indices])[0]

    print(threshold, metric_1, metric_2, base_metric, threshold > max(metric_1, metric_2, base_metric),
          max(metric_1, metric_2) > base_metric,
          " threshold +++++++++++++++++++++++")

    if threshold > max(metric_1, metric_2, base_metric):
        del df1
        del tmp1
        del df2
        del tmp2

        gc.collect()

        return (idx, cols, None)

    if max(metric_1, metric_2) > threshold:

        print("encoder+++++++++++++++++++++++++++++")
        if metric_1 > metric_2:

            encoder = MinMaxScaler(feature_range=(0, 100))
            new_arr = encoder.fit_transform(df1[['-'.join(cols) + '-grpcnt']])

            new_df = pd.DataFrame(new_arr)
            new_df.columns = ['-'.join(cols) + '-grpcnt']

            del tmp1
            del df1
            del df2
            del tmp2
            del new_arr
            del encoder

            gc.collect()

            return (idx, cols, new_df)

        else:

            encoder = MinMaxScaler(feature_range=(0, 100))
            new_arr = encoder.fit_transform(df2[['-'.join(reverse_cols) + '-grpcnt']])

            new_df = pd.DataFrame(new_arr)
            new_df.columns = ['-'.join(reverse_cols) + '-grpcnt']

            del df1
            del df2
            del tmp1
            del tmp2
            del encoder
            del new_arr

            gc.collect()

            return (idx, reverse_cols, new_df)
    else:

        del df1
        del tmp1
        del df2
        del tmp2

        gc.collect()
        return (idx, cols, None)


@timeit
def generate_two_order_cnt_cat_feature(c_df, c_two_order_cols, y, threshold):
    cols_name_c = c_df.columns.tolist()
    # cols_name_c = [col for col in cols_name_c if 'table' not in col]
    if y is not None:

        sampled_indices = sample_indices(y)
        metric = gen_mutual_info(c_df[cols_name_c].iloc[sampled_indices, :], y[sampled_indices])

        metric_dict = dict(zip(cols_name_c, metric))

        combs = list(combinations(cols_name_c, 2))

        two_order_value_dict = {}
        for cols in combs:
            two_order_value_dict[cols] = metric_dict[cols[0]] + metric_dict[cols[1]]
        sorted_two_order_value_dict = dict(sorted(two_order_value_dict.items(), key=lambda d: d[1]))

        print(sorted_two_order_value_dict)

        combs = list(sorted_two_order_value_dict.keys())

        indices = list(range(len(combs)))
        id_combs = list(zip(indices, combs))

        print(id_combs, "id and combs +++++++++++++++++++++++++++++++")

        copy_id_combs = copy.deepcopy(id_combs)
        # random.shuffle(copy_id_combs)
        # random.seed(1024)

        print(copy_id_combs, "copy_id_combs +++++++++++++++++++++++++")

        # sampled_indices = sample_indices(y)
        # metric = gen_mutual_info(c_df[cols_name_c].iloc[sampled_indices, :], y[sampled_indices])
        #
        # metric_dict = dict(zip(cols_name_c, metric))
        #
        # dict_tmp = sorted(metric_dict.items(), key=lambda d: d[1], reverse=True)
        # print(dict_tmp, "metric_dict sorted ssssssssssssssssssssssssssssssssssssssssssssss"
        #                 "ssssssssssssssssssssssssssssssssssssssssssssssssssssssss"
        #                 "sssssssssssssssssssssssssssssssssssssssssss")
        #
        # print(metric_dict, "metric_dict +++++++++++++++++++++++++++++++")
        #
        # combs = list(combinations(cols_name_c, 2))
        # indices = list(range(len(combs)))
        # id_combs = list(zip(indices, combs))
        #
        # print(id_combs, "id and combs +++++++++++++++++++++++++++++++")
        #
        # copy_id_combs = copy.deepcopy(id_combs)
        # random.shuffle(copy_id_combs)
        # random.seed(1024)

        batch_num = 8

        # start = datetime.datetime.now()
        start = time.time()
        time_out = 40

        tups = []

        while time.time() - start < time_out:
            if not copy_id_combs:
                break

            small_batch = copy_id_combs[-batch_num:]
            del copy_id_combs[-batch_num:]

            print(small_batch, "small batch ###########################")

            ctx = mp.get_context('forkserver')

            pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

            result = []

            for idx, col in small_batch:
                base_metric_1 = metric_dict[col[0]]
                base_metric_2 = metric_dict[col[1]]
                max_metric = max(base_metric_1, base_metric_2)

                s1 = c_df[col[0]]
                s2 = c_df[col[1]]

                abortable_func = partial(abortable_worker, c_two_order_cnt_func)
                res = pool.apply_async(abortable_func,
                                       args=(idx, col, s1, s2, y, max_metric, sampled_indices, threshold))
                del s1
                del s2
                gc.collect()
                result.append(res)

            pool.close()
            pool.join()

            tup = []
            for res in result:
                r = res.get()
                tup.append(r)

            tup = [t for t in tup if t[2] is not None]

            tups = tups + tup

            elapsed = time.time()
            print(elapsed - start, "time elapse &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])

            # file_to_new_df_name = t[2]
            #
            # new_df = pd.read_pickle(file_to_new_df_name)
            # print("read new_df###################################\n")
            selected_feat_dfs.append(t[2])

        # os.system('rm c_two_order_cnt_*.pkl')

        del tups
        del c_df
        del result

        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols

    else:

        indices = list(range(len(c_two_order_cols)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, c_two_order_cols))

        for idx, col in idx_cols:
            s1 = c_df[col[0]]
            s2 = c_df[col[1]]

            abortable_func = partial(abortable_worker, c_two_order_cnt_func_predict)
            res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del s1
        del s2
        del tups
        del result

        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols


def c_two_order_cnt_func_predict(idx, cols, s1, s2):
    tmp = s1.astype(str) + '000' + s2.astype(str)
    s_1_2 = tmp.map(tmp.value_counts())
    s_1_2_df = s_1_2.to_frame('-'.join(cols) + '_count')

    encoder = MinMaxScaler(feature_range=(0, 100))
    new_arr = encoder.fit_transform(s_1_2_df[['-'.join(cols) + '_count']])

    new_df = pd.DataFrame(new_arr)
    new_df.columns = ['-'.join(cols) + '_count']

    del tmp
    del s1
    del s2
    del s_1_2
    del s_1_2_df
    gc.collect()

    return (idx, cols, new_df)


def c_two_order_cnt_func(idx, cols, s1, s2, y, base_metric, sampled_indices, threshold):
    # df1 = pd.DataFrame({cols[0]:s1})
    # df2 = pd.DataFrame({cols[1]:s2})

    tmp = s1.astype(str) + '000' + s2.astype(str)
    s_1_2 = tmp.map(tmp.value_counts())
    s_1_2_df = s_1_2.to_frame('-'.join(cols) + '_count')

    metric = gen_mutual_info(s_1_2_df.iloc[sampled_indices, :], y[sampled_indices])[0]

    # print(threshold, metric, base_metric, threshold > max(metric, base_metric),metric > base_metric,cols,
    #       "threshold metric base++++++++++++++++++++++++")
    if threshold > max(metric, base_metric):
        del tmp
        del s1
        del s2
        del s_1_2
        del s_1_2_df
        # s1, s2, tmp, s_1_2, s_1_2_df = None, None, None, None, None

        gc.collect()

        return (idx, cols, None)

    if metric > threshold:

        # print("retrun df+++++++++++++++")
        encoder = MinMaxScaler(feature_range=(0, 100))
        new_arr = encoder.fit_transform(s_1_2_df[['-'.join(cols) + '_count']])

        new_df = pd.DataFrame(new_arr)
        new_df.columns = ['-'.join(cols) + '_count']
        # new_df_to_file_name = 'c_two_order_cnt_' + '-'.join(cols) + '_' + str(idx) + '.pkl'
        #
        # print(new_df_to_file_name, "###########################\n")
        # new_df.to_pickle(new_df_to_file_name)

        del s1
        del s2
        del tmp
        del s_1_2
        del s_1_2_df
        del encoder
        # del new_df
        # s1, s2, tmp, s_1_2, s_1_2_df, encoder, new_df = None, None, None, None, None, None, None
        gc.collect()

        return (idx, cols, new_df)
    else:

        del tmp
        del s1
        del s2
        del s_1_2
        del s_1_2_df
        # s1, s2, tmp, s_1_2, s_1_2_df = None, None, None, None, None

        gc.collect()

        return (idx, cols, None)


def c_one_order_cnt_func(idx, col_name, c_col_df, base_metric, y, sampled_indices, threshold):
    df = pd.DataFrame({col_name: c_col_df})
    df[f'cnt_{col_name}'] = df[col_name].map(df[col_name].value_counts())
    df.drop(col_name, axis=1, inplace=True)

    metric = gen_mutual_info(df.iloc[sampled_indices, :], y[sampled_indices])[0]
    print(threshold, metric, base_metric, threshold > max(metric, base_metric), metric > base_metric,
          "threshold +++++++++++++++")
    if threshold > max(metric, base_metric):
        del df
        gc.collect()
        return (idx, col_name, None)

    if metric > threshold:

        encoder = MinMaxScaler(feature_range=(0, 100))
        new_arr = encoder.fit_transform(df[[f'cnt_{col_name}']])

        new_df = pd.DataFrame(new_arr)
        new_df.columns = [f'cnt_{col_name}']

        del df
        del encoder
        del new_arr
        gc.collect()
        return (idx, col_name, new_df)
    else:
        del df
        gc.collect()
        return (idx, col_name, None)


def c_one_order_cnt_func_predict(idx, col_name, c_col_df):
    df = pd.DataFrame({col_name: c_col_df})
    df[f'cnt_{col_name}'] = df[col_name].map(df[col_name].value_counts())
    df.drop(col_name, axis=1, inplace=True)

    encoder = MinMaxScaler(feature_range=(0, 100))
    new_arr = encoder.fit_transform(df[[f'cnt_{col_name}']])

    new_df = pd.DataFrame(new_arr)
    new_df.columns = [f'cnt_{col_name}']

    del df
    del encoder
    del new_arr
    gc.collect()

    return (idx, col_name, new_df)


def sample_indices(y):
    pos_index = np.where(y.ravel() == 1)[0].tolist()
    neg_index = np.where(y.ravel() == 0)[0].tolist()

    sample_num = min(len(pos_index), len(neg_index))
    sample_num = min(sample_num, 10000)
    p_indics = random.sample(pos_index, sample_num)
    n_indics = random.sample(neg_index, sample_num)
    return p_indics + n_indics


def gen_mutual_info(X, y):
    metric = mutual_info_classif(X, y, n_neighbors=60, discrete_features='auto', random_state=1024)

    del X
    gc.collect()

    return metric.tolist()


def c_one_order_cnt_func_new(idx, col_name, c_col_df):
    df = pd.DataFrame({col_name: c_col_df})
    df[f'cnt_{col_name}'] = df[col_name].map(df[col_name].value_counts())
    df.drop(col_name, axis=1, inplace=True)

    # encoder = MinMaxScaler(feature_range=(0, 100))
    # new_arr = encoder.fit_transform(df[[f'cnt_{col_name}']])
    # new_df = pd.DataFrame(new_arr)
    # new_df.columns = [f'cnt_{col_name}']
    #
    # del df
    # del encoder
    # del new_arr
    # gc.collect()
    #
    # return (idx, col_name, new_df)

    del c_col_df
    gc.collect()
    return (idx, col_name, df)

    # if metric > threshold:
    #
    #     encoder = MinMaxScaler(feature_range=(0, 100))
    #     new_arr = encoder.fit_transform(df[[f'cnt_{col_name}']])
    #
    #     new_df = pd.DataFrame(new_arr)
    #     new_df.columns = [f'cnt_{col_name}']
    #
    #     del df
    #     del encoder
    #     del new_arr
    #     gc.collect()
    #     return (idx, col_name, new_df)
    # else:
    #     del df
    #     gc.collect()
    #     return (idx, col_name, None)


@timeit
def generate_one_order_cnt_cat_feature_mulp_new_add_select(c_df, cols_selected):
    cols_name_c = c_df.columns.tolist()

    if cols_selected is None:

        indices = list(range(len(cols_name_c)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, cols_name_c))

        for idx, col in idx_cols:
            c_col_df = c_df[col]

            abortable_func = partial(abortable_worker, c_one_order_cnt_func_new)
            res = pool.apply_async(abortable_func,
                                   args=(idx, col, c_col_df))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del tups

        del result
        del c_df
        gc.collect()
        print(len(selected_feat_dfs))

        if len(selected_feat_dfs) <= 0:
            return None

        return pd.concat(selected_feat_dfs, axis=1, sort=False)

    else:

        indices = list(range(len(cols_name_c)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, cols_name_c))

        for idx, col in idx_cols:
            if f'cnt_{col}' in cols_selected:
                c_col_df = c_df[col]
                abortable_func = partial(abortable_worker, c_one_order_cnt_func_new)
                res = pool.apply_async(abortable_func, args=(idx, col, c_col_df))
                result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del tups
        del result
        del c_df
        gc.collect()
        print(len(selected_feat_dfs))
        if len(selected_feat_dfs) <= 0:
            return None
        return pd.concat(selected_feat_dfs, axis=1, sort=False)


@timeit
def generate_one_order_cnt_cat_feature_add_select(c_df, c_one_order_cols, y, threshold, cols_selected):
    cols_name_c = c_df.columns.tolist()

    if cols_selected is None:

        sampled_indices = sample_indices(y)
        metric = gen_mutual_info(c_df[cols_name_c].iloc[sampled_indices, :], y[sampled_indices])

        metric_dict = dict(zip(cols_name_c, metric))

        indices = list(range(len(cols_name_c)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, cols_name_c))

        for idx, col in idx_cols:
            c_col_df = c_df[col]
            base_metric = metric_dict[col]
            abortable_func = partial(abortable_worker, c_one_order_cnt_func)
            res = pool.apply_async(abortable_func,
                                   args=(idx, col, c_col_df, base_metric, y, sampled_indices, threshold))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        idx_cols = dict(idx_cols)

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del tups
        del result
        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols

    else:

        indices = list(range(len(c_one_order_cols)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, c_one_order_cols))

        for idx, col in idx_cols:
            if f'cnt_{col}' in cols_selected:
                c_col_df = c_df[col]
                abortable_func = partial(abortable_worker, c_one_order_cnt_func_predict)
                res = pool.apply_async(abortable_func, args=(idx, col, c_col_df))
                result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del tups
        del result
        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols


@timeit
def generate_one_order_cnt_cat_feature(c_df, c_one_order_cols, y, threshold):
    cols_name_c = c_df.columns.tolist()
    print(cols_name_c, "ccccccccccccccccccccccccccccccccc")

    if y is not None:

        sampled_indices = sample_indices(y)
        metric = gen_mutual_info(c_df[cols_name_c].iloc[sampled_indices, :], y[sampled_indices])

        metric_dict = dict(zip(cols_name_c, metric))

        indices = list(range(len(cols_name_c)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, cols_name_c))

        for idx, col in idx_cols:
            c_col_df = c_df[col]
            base_metric = metric_dict[col]
            abortable_func = partial(abortable_worker, c_one_order_cnt_func)
            res = pool.apply_async(abortable_func,
                                   args=(idx, col, c_col_df, base_metric, y, sampled_indices, threshold))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        idx_cols = dict(idx_cols)

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del tups
        del result
        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols

    else:

        indices = list(range(len(c_one_order_cols)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, c_one_order_cols))

        for idx, col in idx_cols:
            c_col_df = c_df[col]
            abortable_func = partial(abortable_worker, c_one_order_cnt_func_predict)
            res = pool.apply_async(abortable_func, args=(idx, col, c_col_df))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append(r)

        tups = [t for t in tups if t[2] is not None]

        tups.sort(key=lambda t: t[0])

        selected_feat_dfs = []
        selected_cols = []

        for t in tups:
            selected_cols.append(t[1])
            selected_feat_dfs.append(t[2])

        del tups
        del result
        gc.collect()
        print(len(selected_feat_dfs))

        return pd.concat(selected_feat_dfs, axis=1, sort=False), selected_cols


'''
    if c_one_order_cols is None:

        indices = list(range(len(cols_name_c)))

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        result = []

        idx_cols = list(zip(indices, cols_name_c))

        for idx, col in idx_cols:
            c_col_df = c_df[col]
            abortable_func = partial(abortable_worker, c_one_order_cnt_func)
            res = pool.apply_async(abortable_func, args=(idx, col, c_col_df))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            tups.append((r[0], r[2]))
        tups = [t for t in type if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            c_df_one_order_cnt = pd.concat(stck, axis=1)
        else:
            pass

        del c_df
        del stck
        gc.collect()

        return c_df_one_order_cnt, cols_name_c

    else:
        pass
'''


@timeit
def t_minus_t(t_df, config):
    table_t_dict = {}

    t_df_cols_to_drop = t_df.columns.tolist()
    # t_df_key_cols = [col for col in t_df_cols_to_drop if 't_0' in col and 'table' not in col]
    t_df_key_cols = [col for col in t_df_cols_to_drop if 't_0' in col]
    print(t_df_cols_to_drop)

    for table_name, value in config['tables'].items():

        print(value)

        t_cols_list_tmp = [col for col in value['type'].keys()
                           if 't_' in col and 't_0' not in col]

        t_cols_list = []

        for col in t_df_cols_to_drop:

            for t_col in t_cols_list_tmp:
                if t_col in col and table_name in col:
                    t_cols_list.append(col)

        t_cols_list = t_cols_list + t_df_key_cols
        if len(t_cols_list) >= 2:
            table_t_dict[table_name] = list(permutations(t_cols_list, 2))

        print(t_cols_list)
        print(table_t_dict)

    if len(table_t_dict) >= 1:

        for table_name, t_pair_list in table_t_dict.items():
            for cols in t_pair_list:
                tmp_s = t_df[cols[0]] - t_df[cols[1]]
                t_df['-minus-'.join(cols) + '_d'] = tmp_s.dt.days
                t_df['-minus-'.join(cols) + '_h'] = (tmp_s / np.timedelta64(1, 'h')).astype(int)
                # t_df.drop(['-minus-'.join(cols)], axis=1, inplace=True)
                # t_df[['-minus-'.join(cols) + '_d', '-minus-'.join(cols) + '_h']] = t_df['-minus-'.join(cols)].dt.components[['days','hours']]
        # print(t_df['t_table_1.t_1-minus-t_table_1.t_2'])
        return t_df.drop(t_df_cols_to_drop, axis=1)

    else:
        return None


@timeit
def time_to_int(t_df):
    for c in t_df.columns.tolist():
        # if c.startswith('t_0'):
        if 't_0' in c:
            t_df[c + '_day'] = t_df[c].dt.day
            t_df[c + '_day_of_week'] = t_df[c].dt.dayofweek
            t_df[c + '_hour'] = t_df[c].dt.hour
            t_df[c + '_day_and_hour'] = t_df[c + '_day'].astype(str) + t_df[c + '_hour'].astype(str)
            t_df[c + '_day_and_hour'] = t_df[c + '_day_and_hour'].astype(int)
            t_df[c + '_day_of_week_and_hour'] = t_df[c + '_day_of_week'].astype(str) + t_df[c + '_hour'].astype(str)
            t_df[c + '_day_of_week_and_hour'] = t_df[c + '_day_of_week_and_hour'].astype(int)
            t_df[c] = t_df[c].dt.hour
        else:
            t_df[c] = t_df[c].dt.dayofweek
    return t_df


def get_dummy_encode_func(idx, col, df):
    df = pd.DataFrame({col: df})
    df[col] = df[col].map(lambda x: int(x))
    return (idx, df)


@timeit
def get_dummy_encode_new(c_df):
    cols_name_c = c_df.columns.tolist()

    indices = list(range(len(cols_name_c)))
    id_pair = list(zip(indices, cols_name_c))
    print(id_pair)

    ctx = mp.get_context('forkserver')

    pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

    result = []

    for idx, col in id_pair:
        df = c_df[col]

        abortable_func = partial(abortable_worker, get_dummy_encode_func)

        res = pool.apply_async(abortable_func, args=(idx, col, df))

        result.append(res)

    pool.close()
    pool.join()

    del c_df
    gc.collect()

    tups = []

    for res in result:
        r = res.get()
        tups.append(r)

    tups.sort(key=lambda a: a[0])

    result_dfs = []

    for r in tups:
        result_dfs.append(r[1])

    del result
    del tups
    gc.collect()

    if len(result_dfs) <= 0:
        return None
    return pd.concat(result_dfs, axis=1, sort=False)


@timeit
def get_dummy_encode(c_df):
    cols_name_c = c_df.columns.tolist()
    for col in cols_name_c:
        c_df[col] = c_df[col].map(lambda x: int(x))
    return c_df


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)

    try:
        out = res.get(timeout)
        return out
    except mp.TimeoutError:
        print("[toy_hyper_paramter_tune] Aborting due to timeout")
        p.terminate()
        return (None, 0, None)


def mv_encoder_func_1(idx, col_name, mv_col):
    col_name = col_name + '+1'
    df = pd.DataFrame({col_name: mv_col})
    # print(df.head(10))
    # df[col_name] = df[col_name].map(lambda a: ','.join(sorted(a.split(','))))
    df[col_name] = df[col_name].map(lambda a: ','.join(a.split(',')[0:1]))
    # print(df.head(10))
    encoder = OrdinalEncoder(cols=[col_name])
    encoded_mv_col = encoder.fit_transform(df)
    del mv_col
    del df
    gc.collect()
    return (idx, col_name, encoded_mv_col, encoder)


def mv_encoder_func_2(idx, col_name, mv_col):
    col_name = col_name + '+2'

    df = pd.DataFrame({col_name: mv_col})
    # print(df.head(10))
    # df[col_name] = df[col_name].map(lambda a: ','.join(sorted(a.split(','))))
    df[col_name] = df[col_name].map(lambda a: ','.join(a.split(',')[0:2]))
    # print(df.head(10))
    encoder = OrdinalEncoder(cols=[col_name])
    encoded_mv_col = encoder.fit_transform(df)
    del mv_col
    del df
    gc.collect()
    return (idx, col_name, encoded_mv_col, encoder)


def mv_encoder_func(idx, col_name, mv_col):
    df = pd.DataFrame({col_name: mv_col})
    # print(df.head(10))
    # df[col_name] = df[col_name].map(lambda a: ','.join(sorted(a.split(','))))
    df[col_name] = df[col_name].map(lambda a: ','.join(a.split(',')[0:3]))
    # print(df.head(10))
    encoder = OrdinalEncoder(cols=[col_name])
    encoded_mv_col = encoder.fit_transform(df)
    del mv_col
    del df
    gc.collect()
    return (idx, col_name, encoded_mv_col, encoder)


def mv_encoder_func_predict(idx, col_name, mv_col, enc):
    df = pd.DataFrame({col_name: mv_col})

    # print(df.head(10))
    # df[col_name] = df[col_name].map(lambda a: ','.join(sorted(a.split(','))))
    df[col_name] = df[col_name].map(lambda a: ','.join(a.split(',')[0:3]))
    # print(df.head(10))

    encoded_mv_col = enc.transform(df)
    del mv_col
    del df
    gc.collect()

    return (idx, col_name, encoded_mv_col, enc)


@timeit
def label_encode_mv_as_cat_mulprocess_add_select(mv_df, mv_encs, cols_selected):
    cols_name_mv = mv_df.columns.tolist()

    if cols_selected is None:

        mv_encoders = {}
        contex = mp.get_context('forkserver')
        pool = contex.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        indices = list(range(len(cols_name_mv)))
        result = []

        for idx, col in zip(indices, cols_name_mv):
            mv_col = mv_df[col]
            abortable_func = partial(abortable_worker, mv_encoder_func)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del tups
            del stck
            gc.collect()

            return mv_df_processed, mv_encoders
        else:
            return None, None
    else:

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)
        indices = list(range(len(cols_name_mv)))
        result = []
        mv_encoders = {}

        for idx, col in zip(indices, cols_name_mv):

            if col in cols_selected:
                enc = mv_encs[col]
                mv_col = mv_df[col]

                abortable_func = partial(abortable_worker, mv_encoder_func_predict)
                res = pool.apply_async(abortable_func, args=(idx, col, mv_col, enc))
                result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del tups
            del stck
            gc.collect()

            return mv_df_processed, mv_encoders
        else:
            return None, None


@timeit
def label_encode_mv_as_cat_mulprocess_1(mv_df, mv_encs=None):
    cols_name_mv = mv_df.columns.tolist()

    if mv_encs is None:

        mv_encoders = {}
        contex = mp.get_context('forkserver')
        pool = contex.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        indices = list(range(len(cols_name_mv)))
        result = []

        for idx, col in zip(indices, cols_name_mv):
            mv_col = mv_df[col]
            abortable_func = partial(abortable_worker, mv_encoder_func_1)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del stck
            gc.collect()

            return mv_df_processed
        else:
            return None
    else:

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)
        indices = list(range(len(cols_name_mv)))
        result = []
        mv_encoders = {}

        for idx, col in zip(indices, cols_name_mv):
            print(type(mv_encs))
            enc = mv_encs[col]
            mv_col = mv_df[col]

            abortable_func = partial(abortable_worker, mv_encoder_func_predict)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col, enc))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del stck
            gc.collect()

            return mv_df_processed
        else:
            return None


@timeit
def label_encode_mv_as_cat_mulprocess_2(mv_df, mv_encs=None):
    cols_name_mv = mv_df.columns.tolist()

    if mv_encs is None:

        mv_encoders = {}
        contex = mp.get_context('forkserver')
        pool = contex.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        indices = list(range(len(cols_name_mv)))
        result = []

        for idx, col in zip(indices, cols_name_mv):
            mv_col = mv_df[col]
            abortable_func = partial(abortable_worker, mv_encoder_func_2)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del stck
            gc.collect()

            return mv_df_processed
        else:
            return None
    else:

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)
        indices = list(range(len(cols_name_mv)))
        result = []
        mv_encoders = {}

        for idx, col in zip(indices, cols_name_mv):
            print(type(mv_encs))
            enc = mv_encs[col]
            mv_col = mv_df[col]

            abortable_func = partial(abortable_worker, mv_encoder_func_predict)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col, enc))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del stck
            gc.collect()

            return mv_df_processed
        else:
            return None


@timeit
def label_encode_mv_as_cat_mulprocess(mv_df, mv_encs=None):
    cols_name_mv = mv_df.columns.tolist()
    print(cols_name_mv)
    print("@" * 100)

    if mv_encs is None:

        mv_encoders = {}
        contex = mp.get_context('forkserver')
        pool = contex.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)

        indices = list(range(len(cols_name_mv)))
        result = []

        for idx, col in zip(indices, cols_name_mv):
            mv_col = mv_df[col]
            abortable_func = partial(abortable_worker, mv_encoder_func)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del stck
            gc.collect()

            return mv_df_processed
        else:
            return None
    else:

        ctx = mp.get_context('forkserver')

        pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)
        indices = list(range(len(cols_name_mv)))
        result = []
        mv_encoders = {}

        for idx, col in zip(indices, cols_name_mv):
            print(type(mv_encs))
            enc = mv_encs[col]
            mv_col = mv_df[col]

            abortable_func = partial(abortable_worker, mv_encoder_func_predict)
            res = pool.apply_async(abortable_func, args=(idx, col, mv_col, enc))
            result.append(res)

        pool.close()
        pool.join()

        tups = []
        for res in result:
            r = res.get()
            mv_encoders[cols_name_mv[r[0]]] = r[3]
            tups.append((r[0], r[2]))

        tups = [t for t in tups if t[0] is not None]
        tups.sort(key=lambda t: t[0])

        stck = [t[1] for t in tups]

        if len(stck) > 0:
            mv_df_processed = pd.concat(stck, axis=1, sort=False)

            del mv_df
            del stck
            gc.collect()

            return mv_df_processed
        else:
            return None


@timeit
def transform_numerical(df, config):
    cats = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX) and 'COUNT' not in c]
    numericals = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX) and 'MEAN' not in c and 'SUM' not in c]
    # print(cats, "==================")
    # print(numericals, "============numericals")
    # cats = random.sample(cats, int(len(cats) / 2))
    # numericals = random.sample(numericals, int(len(numericals) / 2))
    print(cats, '============cats')
    print(numericals, '=============numericals')
    for cat in cats:
        df_tmp = df.groupby(cat)[numericals].agg('mean')
        # print(list(df_tmp.columns),'++++')
        new_columns_mean = [f'({cat})_({c}).mean' for c in numericals]
        df_tmp.columns = new_columns_mean
        # print(list(df_tmp.columns),'----')
        df = df.join(df_tmp, on=cat)
        # print(list(df.columns), '************')
        new_columns_minus = [f'({cat})_({c}).mean_minus' for c in numericals]
        df[new_columns_minus] = df[numericals] - df[new_columns_mean].values
        # print(list(df.columns), '&&&&&&&&&&&')
    print(list(df.columns), '/////////////////')
    return df


@timeit
def transform_datetime(df, config):
    # test

    list_time_col = []
    print(config['tables'], '############################')
    for table_name, value in config['tables'].items():
        #         print(value['type'])
        dict_time_col = {}
        for col in value['type'].keys():
            if 't_' in col and 't_0' not in col:
                #             if 't_' in col:
                dict_time_col[col] = table_name + ',' + col
        if (len(dict_time_col) >= 1):
            list_time_col.append(dict_time_col)

    print(list_time_col)

    #         for key1, value1 in value.items():
    #             print(key1, value1)
    #             for key2, value2 in value1.items():
    #                 print(key2, value2)

    def func_length(x):
        return len(x.split(','))

    for t_col_dict in list_time_col:
        #         t_col_dict = len(t_col_dict)

        if len(t_col_dict) < 2:
            t_cols_pair = list(t_col_dict.values())
            table_name = t_cols_pair[0].split(',')[0]
            t_cols_1 = t_cols_pair[0].split(',')[1]
            if 'main' in table_name:
                t_cols_1_f = None
                for col in df:
                    if 'table' not in col and t_cols_1 in col:
                        t_cols_1_f = col
                diff_time_cols_name = f'(t_01) - ({t_cols_1_f})'
                df[diff_time_cols_name] = df['t_01'] - df[t_cols_1_f]
                df[diff_time_cols_name] = df[diff_time_cols_name].dt.days
                numerical_cols = [n for n in df if 'n_' in n and 'table' not in n]
                df[[f'{col}/{diff_time_cols_name}' for col in numerical_cols]] = df[numerical_cols].div(
                    df[diff_time_cols_name], axis=0)
                mv_cols_len = [m for m in df if 'm_' in m and 'table' not in m and 'length' in m]
                df[[f'{col}/{diff_time_cols_name}' for col in mv_cols_len]] = df[mv_cols_len].div(
                    df[diff_time_cols_name], axis=0)
            else:
                t_cols_1_f = None
                for col in df:
                    if table_name in col and t_cols_1 in col:
                        t_cols_1_f = col
                if t_cols_1_f != None:

                    numerical_cols = [n for n in df if 'n_' in n and table_name in n]
                    mv_cols_len = [m for m in df if 'm_' in m and table_name in m and 'length' in m]
                    # print(numerical_cols)
                    # print(mv_cols_len)
                    if len(numerical_cols) != 0 or len(mv_cols_len) != 0:
                        diff_time_cols_name = f'(t_01) - ({t_cols_1_f})'
                        df[diff_time_cols_name] = df['t_01'] - df[t_cols_1_f]
                        df[diff_time_cols_name] = df[diff_time_cols_name].dt.days
                        # numerical_cols = [n for n in df if 'n_' in n and table_name in n]
                        df[[f'{col}/{diff_time_cols_name}' for col in numerical_cols]] = df[numerical_cols].div(
                            df[diff_time_cols_name], axis=0)
                        # mv_cols_len = [m for m in df if 'm_' in m and table_name in m and 'length' in m]
                        df[[f'{col}/{diff_time_cols_name}' for col in mv_cols_len]] = df[mv_cols_len].div(
                            df[diff_time_cols_name], axis=0)
        else:
            t_cols_pair = list(t_col_dict.values())
            table_name = t_cols_pair[0].split(',')[0]
            t_cols_1 = t_cols_pair[0].split(',')[1]
            t_cols_2 = t_cols_pair[1].split(',')[1]
            t_cols_1_f = None
            t_cols_2_f = None

            for col in df:
                if table_name in col and t_cols_1 in col:
                    t_cols_1_f = col
                if table_name in col and t_cols_2 in col:
                    t_cols_2_f = col

            diff_time_cols_name = f'({t_cols_1_f})-({t_cols_2_f})'
            df[diff_time_cols_name] = df[t_cols_1_f] - df[t_cols_2_f]
            df[diff_time_cols_name] = df[diff_time_cols_name].dt.days
            numerical_cols = [n for n in df if 'n_' in n and table_name in n]
            df[[f'{col}/{diff_time_cols_name}' for col in numerical_cols]] = df[numerical_cols].div(
                df[diff_time_cols_name], axis=0)
            mv_cols_len = [m for m in df if 'm_' in m and table_name in m and 'length' in m]
            #         mv_cols_len = [m + '-length' for m in mv_cols]
            print(mv_cols_len)
            #         df[mv_cols_len] = df[mv_cols].applymap(func_length)
            df[[f'{col}/{diff_time_cols_name}' for col in mv_cols_len]] = df[mv_cols_len].div(df[diff_time_cols_name],
                                                                                              axis=0)

    # submit
    # list_time_col = []
    # # print(config['tables'], '############################')
    # for table_name, value in config['tables'].items():
    #     #         print(value['type'])
    #     dict_time_col = {}
    #     for col in value['type'].keys():
    #         if 't_' in col and 't_0' not in col:
    #             dict_time_col[col] = table_name + ',' + col
    #     if (len(dict_time_col) >= 2):
    #         list_time_col.append(dict_time_col)
    #
    # # print(list_time_col)
    # #         for key1, value1 in value.items():
    # #             print(key1, value1)
    # #             for key2, value2 in value1.items():
    # #                 print(key2, value2)
    # if len(list_time_col) > 0:
    #     for t_col_dict in list_time_col:
    #         t_cols_pair = list(t_col_dict.values())
    #         table_name = t_cols_pair[0].split(',')[0]
    #         t_cols_1 = t_cols_pair[0].split(',')[1]
    #         t_cols_2 = t_cols_pair[1].split(',')[1]
    #         for col in df:
    #             if table_name in col and t_cols_1 in col:
    #                 t_cols_1_f = col
    #             if table_name in col and t_cols_2 in col:
    #                 t_cols_2_f = col
    #
    #         diff_time_cols_name = f'({t_cols_1_f})-({t_cols_2_f})'
    #         df[diff_time_cols_name] = df[t_cols_1_f] - df[t_cols_2_f]
    #         df[diff_time_cols_name] = df[diff_time_cols_name].dt.days
    #         numerical_cols = [n for n in df if 'n_' in n and table_name in n]
    #         df[[f'{col}/diff_time' for col in numerical_cols]] = df[numerical_cols].div(df[diff_time_cols_name], axis=0)
    #         mv_cols_len = [m for m in df if 'm_' in m and table_name in m and 'length' in m]
    #         df[[f'{col}/diff_time' for col in mv_cols_len]] = df[mv_cols_len].div(df[diff_time_cols_name], axis = 0)

    # time_cols = [t for t in df if 't_' in t and 't_0' not in t]
    # print(time_cols)
    # print(config['tables'], "########################")
    # if len(time_cols) >= 2:
    #     diff_time_cols_name = f'({time_cols[0]})-({time_cols[1]})'
    #     df[diff_time_cols_name] = df[time_cols[0]] - df[time_cols[1]]
    #     df[diff_time_cols_name] = df[diff_time_cols_name].dt.days
    #     numerical_cols = [n for n in df if 'n_' in n and 'table_1' in n]
    #     df[[f'{col}/diff_time' for col in numerical_cols]] = df[numerical_cols].div(df[diff_time_cols_name], axis=0)
    # print(list(df.columns), "time ============================")
    # print(df[[diff_time_cols_name]])

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        # df.drop(c, axis=1, inplace=True)
        if c.startswith('t_0'):
            df[c] = df[c].dt.hour
        else:
            df[c] = df[c].dt.dayofweek
        # df[f'{c}_m'] = df[c].dt.dayofweek


@timeit
def transform_categorical_hash(df, y=None, two_order_cols=None, two_group_cols=None):
    cats = [c for c in df if c.startswith(
        CONSTANT.CATEGORY_PREFIX) and 'SUM' not in c and 'COUNT' not in c and 'MEAN' not in c and 'UNIQUE' not in c]
    for c in cats:
        df[c] = df[c].apply(lambda x: int(x))
        df[f'cnt_{c}'] = df[c].map(df[c].value_counts())
    '''
    if two_order_cols is None:
        cats = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX) and 'SUM' not in c and 'COUNT' not in c and 'MEAN'not in c and 'UNIQUE' not in c]
        for c in cats:
            df[c] = df[c].apply(lambda x: int(x))
            df[f'cnt_{c}'] = df[c].map(df[c].value_counts())

        pos_index = np.where(y.ravel() == 1)[0].tolist()
        neg_index = np.where(y.ravel() == 0)[0].tolist()

        sample_num = min(int(len(pos_index) / 10), int(len(neg_index) / 10))

        p = random.sample(pos_index, sample_num)
        n = random.sample(neg_index, sample_num)
        sample_index = p + n
        metric = mutual_info_classif(df[cats].iloc[sample_index, :], list(y[sample_index]))
        metric_dic = dict(zip(cats, metric))

        combs = list(combinations(cats, 2))

        combs_sample = random.sample(combs, int(len(combs) / 10))
        two_order_cols_buffer = []
        for col in combs_sample:
            d0 = metric_dic[col[0]]
            d1 = metric_dic[col[1]]
            max_dependence = max(d0, d1)
            s1 = df[col[0]]
            s2 = df[col[1]]
            tmp = s1.astype(str) + '0000' + s2.astype(str)
            ss_cnt_v = tmp.map(tmp.value_counts())
            ss_cnt_v_df = ss_cnt_v.to_frame('tmp')

            value = ss_cnt_v_df.values
            metrix_ss = mutual_info_classif(value[sample_index, :], list(y[sample_index]))
            if metrix_ss[0] > max_dependence:
                df['-'.join(col) + 'cnt'] = ss_cnt_v
                two_order_cols_buffer.append(col)

        # combs_sample_grp = random.sample(combs, int(len(combs) / 10))
        # # print("still alive")
        # two_group_buffer = []
        # for col in combs_sample_grp:
        #     # print("still alive")
        #     d0 = metric_dic[col[0]]
        #     d1 = metric_dic[col[1]]
        #     max_dependence_2 = max(d0, d1)
        #     s1 = df[col[0]]
        #     s2 = df[col[1]]
        #
        #     df1 = pd.concat([s1, s2], axis=1)
        #     tmp = df1.groupby(col[0])[col[1]].nunique().reset_index()
        #     tmp.columns = [col[0], '-'.join(col) + '.grb']
        #     df1 = df1.merge(tmp, how='left')
        #     #         df1 = df1.drop(col, axis = 1)
        #
        #     if y is not None:
        #         reverse_cols = [col[1], col[0]]
        #         df2 = pd.concat([s2, s1], axis=1)
        #         tmp2 = df2.groupby(col[1])[col[0]].nunique().reset_index()
        #         tmp2.columns = [col[1], '-'.join(reverse_cols) + '.grb']
        #         df2 = df2.merge(tmp2, how='left')
        #         #             df2 = df2.drop(reverse_cols, axis = 1)
        #
        #         new_feat1 = df1.values
        #         new_feat2 = df2.values
        #
        #         dependence1 = mutual_info_classif(new_feat1[sample_index, :], list(y[sample_index]))
        #         dependence2 = mutual_info_classif(new_feat2[sample_index, :], list(y[sample_index]))
        #
        #         #             max_dep_new = max(dependence1, dependence2)
        #         if max(dependence1[0], dependence2[0]) > max_dependence_2:
        #             if dependence1[0] > dependence2[0]:
        #                 df['-'.join(col) + '.grb'] = df1['-'.join(col) + '.grb']
        #                 two_group_buffer.append(col)
        #             else:
        #                 df['-'.join(reverse_cols) + '.grb'] = df2['-'.join(reverse_cols) + '.grb']
        #                 two_group_buffer.append(reverse_cols)
        #         else:
        #             pass

        # print(list(df.columns))


    else:
        cats = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX) and 'SUM' not in c and 'COUNT' not in c and 'MEAN'not in c and 'UNIQUE' not in c]
        for c in cats:
            df[c] = df[c].apply(lambda x: int(x))
            df[f'cnt_{c}'] = df[c].map(df[c].value_counts())

        for col in two_order_cols:
            s1 = df[col[0]]
            s2 = df[col[1]]
            tmp = s1.astype(str) + '0000' + s2.astype(str)
            ss_cnt_v = tmp.map(tmp.value_counts())
            df['-'.join(col) + 'cnt'] = ss_cnt_v
        # for col in two_group_cols:
        #     s1 = df[col[0]]
        #     s2 = df[col[1]]
        #     df1 = pd.concat([s1, s2], axis=1)
        #     tmp = df1.groupby(col[0])[col[1]].nunique().reset_index()
        #     tmp.columns = [col[0], '-'.join(col) + '.grb']
        #     df1 = df1.merge(tmp, how='left')
        #     df['-'.join(col) + '.grb'] = df1['-'.join(col) + '.grb']

        # print(list(df.columns))
    '''
    # mv = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    # encoder = OrdinalEncoder(cols=mv)
    # res = encoder.fit_transform(df)
    # df.drop(mv, )
    # print(df['m_table_2.m_1'])
    # for c in mv:
    #     df[c] = df[c].apply(lambda x: int(x))

    # for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
    #     df[c] = df[c].apply(lambda x: int(x.split(',')[0]))

    mv = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    mv_cols_len = [m + '_length' for m in mv]
    df[mv_cols_len] = df[mv].applymap(lambda a: len(a.split(',')))

    encoder = OrdinalEncoder(cols=mv)
    df1 = encoder.fit_transform(df)
    df[mv] = df1[mv]

    # print(list(df.columns))

    try:
        return two_order_cols_buffer, None
    except:
        return None, None
