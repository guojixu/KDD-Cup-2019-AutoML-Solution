# import os
#
# # try:
# #     import hyperopt
# # except ImportError:
# #     os.system("pip3 install hyperopt")
# #
# # try:
# #     import lightgbm
# # except ImportError:
# #     os.system("pip3 install lightgbm")
# #
# # try:
# #
# #     import pandas as pd
# #     print("#########################################")
# #     print(pd.__version__)
# #
# #     if pd.__version__ not in '0.24.2':
# #         os.system("pip3 install pandas==0.24.2")
# #         import pandas as pd
# #         print(pd.__version__)
# #
# #     print(pd.__version__)
# #     print("#########################################")
# #
# #
# #     assert pd.__version__ == '0.24.2'
# #     # if pd.__version__ not in '0.24.2':
# #     #     os.system("pip3 install pandas==0.24.2")
# #     #     import pandas as pd
# #     #     print(pd.__version__)
# #
# #     print("#########################################")
# #
# # except AssertionError:
# #     os.system("pip3 install pandas==0.24.2")
# #     import pandas as pd
# #     print(pd.__version__)
# #     print("*"*40)
# #
# #
# # try:
# #     from category_encoders import OrdinalEncoder
# # except ImportError:
# #     os.system("pip3 install category_encoders")
#
#
# try:
#     # print("#"*500)
#     with open('flag', 'r') as f:
#         f.read()
#
# except:
#
#     os.system("pip3 install hyperopt")
#     os.system("pip3 install lightgbm")
#     os.system("pip3 install pandas==0.24.2")
#     os.system("pip3 install category_encoders")
#     os.system("pip3 install seaborn")
#     with open('flag', 'w') as f:
#         f.write("installed")
#     # print("*"*500)
#
#
#
#
#
# # os.system("pip3 install hyperopt")
# # os.system("pip3 install lightgbm")
# # os.system("pip3 install pandas==0.24.2")
# # os.system("pip3 install category_encoders")
# # os.system("pip3 install seaborn")
#
#
#
# import copy
# import numpy as np
# import pandas as pd
#
# # print("#########################################")
# # print(pd.__version__)
# # print("#########################################")
#
# from automl import predict, train, validate
# from CONSTANT import MAIN_TABLE_NAME
# from merge import merge_table
# from preprocess import clean_df, clean_tables, feature_engineer
# from util import Config, log, show_dataframe, timeit
#
# import gc
# # print("not import")
#
# from util import feature_selector
# from util import data_sample_new
# # from select import feature_selector
# # from util import log, timeit
# # from FeatureSelector import FeatureSelector
# #
# # @timeit
# # def feature_selector(X, y):
# #     fs = FeatureSelector(data=X, labels=y)
# #     # fs.identify_all(
# #     #     selection_params={
# #     #         'task': 'classification',
# #     #         'eval_metric': 'auc',
# #     #         # "num_threads": 4,
# #     #         'missing_threshold': 0.6,
# #     #         'correlation_threshold': 0.7,
# #     #         'cumulative_importance': 0.9
# #     #     }
# #     # )
# #     fs.identify_single_unique()
# #     print(fs.ops)
# #
# #
# #     X = fs.remove(methods=['single_unique'], keep_one_hot=False)
# #     cols_selected = X.columns.tolist()
# #     return X, cols_selected
# #     # if len(fs.ops['single_unique']) < 10:
# #     #     X = fs.remove(methods=['single_unique'], keep_one_hot=False)
# #     #     cols_selected = X.columns.tolist()
# #     #     return X, cols_selected
# #     # else:
# #     #     cols_selected = X.columns.tolist()
# #     #     return X_tmp, cols_selected
#
#
# # print("import module")
#
# class Model:
#     def __init__(self, info):
#         self.config = Config(info)
#         self.tables = None
#         self.two_order_cols = None
#         self.two_group_cols = None
#         self.mv_encs = None
#         self.c_one_order_cols = None
#         self.c_two_order_cols = None
#         self.c_two_order_group_cnt_cols = None
#         self.c_two_order_n_groupby_cat_cols = None
#         self.cols_selected = None
#         self.n_minus_mean_cols = None
#     @timeit
#     def fit(self, Xs, y, time_ramain):
#         self.tables = copy.deepcopy(Xs)
#
#         # # clean_tables(Xs)
#         # X = merge_table(Xs, self.config)
#         # clean_df(X)
#         # X, \
#         # self.two_order_cols, \
#         # self.two_group_cols, \
#         # self.mv_encs, \
#         # self.c_one_order_cols, \
#         # self.c_two_order_cols, \
#         # self.c_two_order_group_cnt_cols, \
#         # self.c_two_order_n_groupby_cat_cols, \
#         # self.n_minus_mean_cols \
#         # = feature_engineer(X, self.config, y,
#         #                    two_order_cols=self.two_order_cols,
#         #                    two_group_cols=self.two_group_cols,
#         #                    mv_encs=self.mv_encs,
#         #                    c_one_order_cols=self.c_one_order_cols,
#         #                    c_two_order_cols= self.c_two_order_cols,
#         #                    c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
#         #                    c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
#         #                    n_minus_mean_cols=self.n_minus_mean_cols,
#         #                    cols_selected=self.cols_selected
#         #                    )
#         # print(list(X.columns))
#         # #
#         # # unique_counts = X.nunique()
#         # #
#         # # tmp = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
#         # #
#         # # sorted_X = tmp.sort_values('nunique')
#         # #
#         # # for i in sorted_X.index:
#         # #     print(sorted_X.loc[[i], :])
#         # #
#         # X, self.cols_selected = feature_selector(X, y)
#         #
#         # gc.collect()
#
#
#
#
#         # user small X to determinal feature preocess
#
#         print("########################################################################\n"
#               "#               use part of data to select features                    #\n"
#               "########################################################################\n")
#
#         # clean_tables(Xs)
#         X = merge_table(Xs, self.config)
#         clean_df(X)
#
#         sample_num = int(len(y) / 10)
#
#         part_X, part_y = data_sample_new(X, y, sample_num)
#         # part_X, part_y = X, y
#         # print(type(X), type(part_X))
#         # print(type(y), type(part_y))
#         # print(part_X.head())
#         # print(part_y)
#
#         part_X = part_X.reset_index(drop=True)
#         part_y = part_y.reset_index(drop=True)
#         tmp_part_X, \
#         self.two_order_cols, \
#         self.two_group_cols, \
#         self.mv_encs, \
#         self.c_one_order_cols, \
#         self.c_two_order_cols, \
#         self.c_two_order_group_cnt_cols, \
#         self.c_two_order_n_groupby_cat_cols, \
#         self.n_minus_mean_cols \
#             = feature_engineer(part_X, self.config, part_y,
#                                two_order_cols=self.two_order_cols,
#                                two_group_cols=self.two_group_cols,
#                                mv_encs=self.mv_encs,
#                                c_one_order_cols=self.c_one_order_cols,
#                                c_two_order_cols=self.c_two_order_cols,
#                                c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
#                                c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
#                                n_minus_mean_cols=self.n_minus_mean_cols,
#                                cols_selected=self.cols_selected
#                                )
#
#         #
#         # unique_counts = X.nunique()
#         #
#         # tmp = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
#         #
#         # sorted_X = tmp.sort_values('nunique')
#         #
#         # for i in sorted_X.index:
#         #     print(sorted_X.loc[[i], :])
#         #
#
#         tmp_part_X_d, self.cols_selected = feature_selector(tmp_part_X, part_y)
#
#         self.mv_encs = None
#
#         del tmp_part_X
#         del tmp_part_X_d
#         del part_X
#         del part_y
#
#
#         gc.collect()
#
#
#
#         ############################## use all data ###########################################
#
#
#         print("########################################################################\n"
#               "#              after select feature use  all of data to train          #\n"
#               "########################################################################\n")
#
#         # clean_tables(Xs)
#         X, \
#         self.two_order_cols, \
#         self.two_group_cols, \
#         self.mv_encs, \
#         self.c_one_order_cols, \
#         self.c_two_order_cols, \
#         self.c_two_order_group_cnt_cols, \
#         self.c_two_order_n_groupby_cat_cols, \
#         self.n_minus_mean_cols \
#         = feature_engineer(X, self.config, y=None,
#                            two_order_cols=self.two_order_cols,
#                            two_group_cols=self.two_group_cols,
#                            mv_encs=self.mv_encs,
#                            c_one_order_cols=self.c_one_order_cols,
#                            c_two_order_cols= self.c_two_order_cols,
#                            c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
#                            c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
#                            n_minus_mean_cols=self.n_minus_mean_cols,
#                            cols_selected=self.cols_selected
#                            )
#         print(list(X.columns))
#         #
#         # unique_counts = X.nunique()
#         #
#         # tmp = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
#         #
#         # sorted_X = tmp.sort_values('nunique')
#         #
#         # for i in sorted_X.index:
#         #     print(sorted_X.loc[[i], :])
#         #
#
#         # X, self.cols_selected = feature_selector(X, y)
#
#         gc.collect()
#
#         X = X[self.cols_selected]
#
#
#
#         print(self.cols_selected)
#         print(X.columns.tolist())
#         print(len(X.columns.tolist()))
#
#         gc.collect()
#
#         train(X, y, self.config)
#
#     @timeit
#     def predict(self, X_test, time_remain):
#
#         Xs = self.tables
#         main_table = Xs[MAIN_TABLE_NAME]
#         main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
#         # main_table = pd.concat([X_test], keys=['test'])
#
#         main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
#         Xs[MAIN_TABLE_NAME] = main_table
#
#         # clean_tables(Xs)
#         X = merge_table(Xs, self.config)
#         clean_df(X)
#
#         X, \
#         self.two_order_cols, \
#         self.two_group_cols, \
#         self.mv_encs, \
#         self.c_one_order_cols, \
#         self.c_two_order_cols, \
#         self.c_two_order_group_cnt_cols, \
#         self.c_two_order_n_groupby_cat_cols, \
#         self.n_minus_mean_cols \
#         = feature_engineer(X, self.config,
#                            two_order_cols=self.two_order_cols,
#                            two_group_cols=self.two_group_cols,
#                            mv_encs=self.mv_encs,
#                            c_one_order_cols=self.c_one_order_cols,
#                            c_two_order_cols=self.c_two_order_cols,
#                            c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
#                            c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
#                            n_minus_mean_cols=self.n_minus_mean_cols,
#                            cols_selected=self.cols_selected
#                            )
#
#         X = X[self.cols_selected]
#
#         print(X.columns.tolist())
#         print(self.cols_selected)
#
#         X = X[X.index.str.startswith("test")]
#         X.index = X.index.map(lambda x: int(x.split('_')[1]))
#         X.sort_index(inplace=True)
#
#
#         gc.collect()
#
#         result = predict(X, self.config)
#
#         return pd.Series(result)

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

import os

# try:
#     import hyperopt
# except ImportError:
#     os.system("pip3 install hyperopt")
#
# try:
#     import lightgbm
# except ImportError:
#     os.system("pip3 install lightgbm")
#
# try:
#
#     import pandas as pd
#     print("#########################################")
#     print(pd.__version__)
#
#     if pd.__version__ not in '0.24.2':
#         os.system("pip3 install pandas==0.24.2")
#         import pandas as pd
#         print(pd.__version__)
#
#     print(pd.__version__)
#     print("#########################################")
#
#
#     assert pd.__version__ == '0.24.2'
#     # if pd.__version__ not in '0.24.2':
#     #     os.system("pip3 install pandas==0.24.2")
#     #     import pandas as pd
#     #     print(pd.__version__)
#
#     print("#########################################")
#
# except AssertionError:
#     os.system("pip3 install pandas==0.24.2")
#     import pandas as pd
#     print(pd.__version__)
#     print("*"*40)
#
#
# try:
#     from category_encoders import OrdinalEncoder
# except ImportError:
#     os.system("pip3 install category_encoders")


try:
    # print("#"*500)
    with open('flag', 'r') as f:
        f.read()

except:

    os.system("apt install libdpkg-perl -y")
    os.system("pip3 install hyperopt")
    os.system("pip3 install lightgbm")
    os.system("pip3 install pandas==0.24.2")
    os.system("pip3 install category_encoders==2.0.0")
    # os.system("pip3 install seaborn")
    with open('flag', 'w') as f:
        f.write("installed")
    # print("*"*500)

# os.system("pip3 install hyperopt")
# os.system("pip3 install lightgbm")
# os.system("pip3 install pandas==0.24.2")
# os.system("pip3 install category_encoders")
# os.system("pip3 install seaborn")


import copy
import numpy as np
import pandas as pd

# print("#########################################")
# print(pd.__version__)
# print("#########################################")

from automl_kdd import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer
from util import Config, log, show_dataframe, timeit

import gc
# print("not import")

from util import feature_selector
from util import data_sample_new
from util import MyTimer

import time


# from select import feature_selector
# from util import log, timeit
# from FeatureSelector import FeatureSelector
#
# @timeit
# def feature_selector(X, y):
#     fs = FeatureSelector(data=X, labels=y)
#     # fs.identify_all(
#     #     selection_params={
#     #         'task': 'classification',
#     #         'eval_metric': 'auc',
#     #         # "num_threads": 4,
#     #         'missing_threshold': 0.6,
#     #         'correlation_threshold': 0.7,
#     #         'cumulative_importance': 0.9
#     #     }
#     # )
#     fs.identify_single_unique()
#     print(fs.ops)
#
#
#     X = fs.remove(methods=['single_unique'], keep_one_hot=False)
#     cols_selected = X.columns.tolist()
#     return X, cols_selected
#     # if len(fs.ops['single_unique']) < 10:
#     #     X = fs.remove(methods=['single_unique'], keep_one_hot=False)
#     #     cols_selected = X.columns.tolist()
#     #     return X, cols_selected
#     # else:
#     #     cols_selected = X.columns.tolist()
#     #     return X_tmp, cols_selected


# print("import module")

class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.two_order_cols = None
        self.two_group_cols = None
        self.mv_encs = None
        self.c_one_order_cols = None
        self.c_two_order_cols = None
        self.c_two_order_group_cnt_cols = None
        self.c_two_order_n_groupby_cat_cols = None
        self.cols_selected = None
        self.n_minus_mean_cols = None
        self.y = None
        self.mytimer = None

    @timeit
    def fit(self, Xs, y, time_ramain):

        self.tables = copy.deepcopy(Xs)
        self.y = copy.deepcopy(y)
        # # clean_tables(Xs)
        # X = merge_table(Xs, self.config)
        # clean_df(X)
        # X, \
        # self.two_order_cols, \
        # self.two_group_cols, \
        # self.mv_encs, \
        # self.c_one_order_cols, \
        # self.c_two_order_cols, \
        # self.c_two_order_group_cnt_cols, \
        # self.c_two_order_n_groupby_cat_cols, \
        # self.n_minus_mean_cols \
        # = feature_engineer(X, self.config, y,
        #                    two_order_cols=self.two_order_cols,
        #                    two_group_cols=self.two_group_cols,
        #                    mv_encs=self.mv_encs,
        #                    c_one_order_cols=self.c_one_order_cols,
        #                    c_two_order_cols= self.c_two_order_cols,
        #                    c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
        #                    c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
        #                    n_minus_mean_cols=self.n_minus_mean_cols,
        #                    cols_selected=self.cols_selected
        #                    )
        # print(list(X.columns))
        # #
        # # unique_counts = X.nunique()
        # #
        # # tmp = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        # #
        # # sorted_X = tmp.sort_values('nunique')
        # #
        # # for i in sorted_X.index:
        # #     print(sorted_X.loc[[i], :])
        # #
        # X, self.cols_selected = feature_selector(X, y)
        #
        # gc.collect()

        # user small X to determinal feature preocess

        # self.mytimer = MyTimer()
        # self.mytimer.remain = self.config['time_budget'] - 10
        # self.mytimer.total = self.config['time_budget'] - 10

        # with self.mytimer.time_limit('Before train'):
        #
        #
        #     print("########################################################################\n"
        #           "#               use part of data to select features                    #\n"
        #           "########################################################################\n")
        #
        #     # clean_tables(Xs)
        #     X = merge_table(Xs, self.config)
        #     clean_df(X)
        #
        #     big_df_memory = X.memory_usage().sum()
        #     big_df_len = X.shape[0]
        #
        #     sample_num = int(len(y) / 10)
        #
        #     part_X, part_y = data_sample_new(X, y, sample_num)
        #
        #     del X
        #     del y
        #     gc.collect()
        #     # part_X, part_y = X, y
        #     # print(type(X), type(part_X))
        #     # print(type(y), type(part_y))
        #     # print(part_X.head())
        #     # print(part_y)
        #
        #     part_X = part_X.reset_index(drop=True)
        #     part_y = part_y.reset_index(drop=True)
        #     tmp_part_X, \
        #     self.two_order_cols, \
        #     self.two_group_cols, \
        #     self.mv_encs, \
        #     self.c_one_order_cols, \
        #     self.c_two_order_cols, \
        #     self.c_two_order_group_cnt_cols, \
        #     self.c_two_order_n_groupby_cat_cols, \
        #     self.n_minus_mean_cols, \
        #     max_numb_cols_to_select \
        #         = feature_engineer(part_X, self.config, part_y,
        #                            two_order_cols=self.two_order_cols,
        #                            two_group_cols=self.two_group_cols,
        #                            mv_encs=self.mv_encs,
        #                            c_one_order_cols=self.c_one_order_cols,
        #                            c_two_order_cols=self.c_two_order_cols,
        #                            c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
        #                            c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
        #                            n_minus_mean_cols=self.n_minus_mean_cols,
        #                            cols_selected=self.cols_selected,
        #                            big_df_memory=big_df_memory,
        #                            big_df_len=big_df_len,
        #                            )
        #
        #
        #
        #
        #
        #     #
        #     # unique_counts = X.nunique()
        #     #
        #     # tmp = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        #     #
        #     # sorted_X = tmp.sort_values('nunique')
        #     #
        #     # for i in sorted_X.index:
        #     #     print(sorted_X.loc[[i], :])
        #     #
        #
        #     tmp_part_X_d, self.cols_selected = feature_selector(tmp_part_X, part_y, max_numb_cols_to_select=max_numb_cols_to_select)
        #
        #     print("#" * 50)
        #     print(part_X.memory_usage())
        #
        #     print(tmp_part_X_d.memory_usage())
        #
        #     part_X_mem_use_b = part_X.memory_usage().sum()
        #     tmp_part_X_mem_use_b = tmp_part_X.memory_usage().sum()
        #     tmp_part_X_d_mem_use_b = tmp_part_X_d.memory_usage().sum()
        #     print(part_X_mem_use_b)
        #     print(tmp_part_X_d_mem_use_b)
        #     print(tmp_part_X_d_mem_use_b / part_X_mem_use_b)
        #     print(tmp_part_X_mem_use_b / part_X_mem_use_b)
        #
        #     part_X_mem_use_g = part_X.memory_usage().sum() / (1024 ** 3)
        #     tmp_part_X__d_mem_use_g = tmp_part_X_d.memory_usage().sum() / (1024 ** 3)
        #     print(part_X_mem_use_g)
        #     print(tmp_part_X__d_mem_use_g)
        #     print(tmp_part_X__d_mem_use_g / part_X_mem_use_g)
        #     print("#" * 50)
        #
        #
        #     self.mv_encs = None
        #
        #     del tmp_part_X
        #     del tmp_part_X_d
        #     del part_X
        #     del part_y
        #
        #     gc.collect()
        #
        #     ############################## use all data ###########################################
        #
        #     print("########################################################################\n"
        #           "#              after select feature use  all of data to train          #\n"
        #           "########################################################################\n")

            # clean_tables(Xs)
            # X, \
            # self.two_order_cols, \
            # self.two_group_cols, \
            # self.mv_encs, \
            # self.c_one_order_cols, \
            # self.c_two_order_cols, \
            # self.c_two_order_group_cnt_cols, \
            # self.c_two_order_n_groupby_cat_cols, \
            # self.n_minus_mean_cols \
            # = feature_engineer(X, self.config, y=None,
            #                    two_order_cols=self.two_order_cols,
            #                    two_group_cols=self.two_group_cols,
            #                    mv_encs=self.mv_encs,
            #                    c_one_order_cols=self.c_one_order_cols,
            #                    c_two_order_cols= self.c_two_order_cols,
            #                    c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
            #                    c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
            #                    n_minus_mean_cols=self.n_minus_mean_cols,
            #                    cols_selected=self.cols_selected
            #                    )
            # print(list(X.columns))
            # #
            # # unique_counts = X.nunique()
            # #
            # # tmp = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
            # #
            # # sorted_X = tmp.sort_values('nunique')
            # #
            # # for i in sorted_X.index:
            # #     print(sorted_X.loc[[i], :])
            # #

            # # X, self.cols_selected = feature_selector(X, y)

            # gc.collect()

            # X = X[self.cols_selected]

            # print(self.cols_selected)
            # print(X.columns.tolist())
            # print(len(X.columns.tolist()))

            # gc.collect()

            # train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        time_1 = time.time()

        Xs = self.tables
        main_table_tmp = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table_tmp, X_test], keys=['train', 'test'])
        # main_table = pd.concat([X_test], keys=['test'])

        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        del main_table_tmp
        del X_test
        gc.collect()

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        del Xs
        gc.collect()

##############################################################################3
##############################################################################3

        print("########################################################################\n"
              "#              select feature                                          #\n"
              "########################################################################\n")


        X_to_select = X[X.index.str.startswith("train")]

        big_df_memory = X_to_select.memory_usage().sum()
        big_df_len = X_to_select.shape[0]


        sample_num = int(len(self.y) / 10)

        part_X, part_y = data_sample_new(X_to_select, self.y, sample_num)

        del X_to_select
        # del y
        gc.collect()

        part_X = part_X.reset_index(drop=True)
        part_y = part_y.reset_index(drop=True)
        tmp_part_X, \
        self.two_order_cols, \
        self.two_group_cols, \
        self.mv_encs, \
        self.c_one_order_cols, \
        self.c_two_order_cols, \
        self.c_two_order_group_cnt_cols, \
        self.c_two_order_n_groupby_cat_cols, \
        self.n_minus_mean_cols, \
        max_numb_cols_to_select, \
        fe_model \
            = feature_engineer(part_X, self.config, part_y,
                               two_order_cols=self.two_order_cols,
                               two_group_cols=self.two_group_cols,
                               mv_encs=self.mv_encs,
                               c_one_order_cols=self.c_one_order_cols,
                               c_two_order_cols=self.c_two_order_cols,
                               c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
                               c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
                               n_minus_mean_cols=self.n_minus_mean_cols,
                               cols_selected=self.cols_selected,
                               big_df_memory=big_df_memory,
                               big_df_len=big_df_len,
                               fe_model=None
                               )

        tmp_part_X_d, self.cols_selected = feature_selector(tmp_part_X, part_y, max_numb_cols_to_select=max_numb_cols_to_select)

        # print("#" * 50)
        # print(part_X.memory_usage())
        #
        # print(tmp_part_X_d.memory_usage())
        #
        # part_X_mem_use_b = part_X.memory_usage().sum()
        # tmp_part_X_mem_use_b = tmp_part_X.memory_usage().sum()
        # tmp_part_X_d_mem_use_b = tmp_part_X_d.memory_usage().sum()
        # print(part_X_mem_use_b)
        # print(tmp_part_X_d_mem_use_b)
        # print(tmp_part_X_d_mem_use_b / part_X_mem_use_b)
        # print(tmp_part_X_mem_use_b / part_X_mem_use_b)
        #
        # part_X_mem_use_g = part_X.memory_usage().sum() / (1024 ** 3)
        # tmp_part_X__d_mem_use_g = tmp_part_X_d.memory_usage().sum() / (1024 ** 3)
        # print(part_X_mem_use_g)
        # print(tmp_part_X__d_mem_use_g)
        # print(tmp_part_X__d_mem_use_g / part_X_mem_use_g)
        # print("#" * 50)


        self.mv_encs = None

        del tmp_part_X
        del tmp_part_X_d
        del part_X
        del part_y

        gc.collect()

        print("########################################################################\n"
              "#              after select feature use  all of data to train          #\n"
              "########################################################################\n")
##############################################################################3
##############################################################################3


        X, \
        self.two_order_cols, \
        self.two_group_cols, \
        self.mv_encs, \
        self.c_one_order_cols, \
        self.c_two_order_cols, \
        self.c_two_order_group_cnt_cols, \
        self.c_two_order_n_groupby_cat_cols, \
        self.n_minus_mean_cols, \
        max_numb_cols_to_select, \
        fe_model \
            = feature_engineer(X, self.config,
                               two_order_cols=self.two_order_cols,
                               two_group_cols=self.two_group_cols,
                               mv_encs=self.mv_encs,
                               c_one_order_cols=self.c_one_order_cols,
                               c_two_order_cols=self.c_two_order_cols,
                               c_two_order_group_cnt_cols=self.c_two_order_group_cnt_cols,
                               c_two_order_n_groupby_cat_cols=self.c_two_order_n_groupby_cat_cols,
                               n_minus_mean_cols=self.n_minus_mean_cols,
                               cols_selected=self.cols_selected,
                               fe_model=fe_model
                               )

        X = X[self.cols_selected]

        print(X.columns.tolist())
        print(self.cols_selected)



        X_train = X[X.index.str.startswith("train")]
        X = X[X.index.str.startswith("test")]

        gc.collect()


        X_train.index = X_train.index.map(lambda x: int(x.split('_')[1]))
        X_train.sort_index(inplace=True)
        gc.collect()


        time_2 = time.time()

        time_left_to_train = time_remain - (time_2 - time_1)
        tmp_time = time_left_to_train
        run_flag = True
        a_time = 0
        train_count = 0

        train_num = 0
        run_num = 1
        # while run_flag:
        change_flag = True
        print(tmp_time)
        while run_num > 0:

            for i in range(1):
                t_1 = time.time()
                part_X, part_y = data_sample_for_train(X_train, self.y)
                print("*" * 10)
                print(len(part_y))
                print("*" * 10)
                train(part_X, part_y, self.config)
                t_2 = time.time()

                a_time = t_2 - t_1

                time_left_to_train = time_left_to_train - a_time
                print('a_time: ', a_time, 'time_left_to_train: ', time_left_to_train)

                if tmp_time / a_time > 60:
                    if change_flag:
                        run_num = 25
                        print("###25###")
                elif tmp_time / a_time < 5 and tmp_time > 3 * a_time:
                    if change_flag:
                        run_num = 2
                        print("###2###")

                elif time_left_to_train <= 3 * a_time:
                        run_num = 0
                        print("###stop###")

                elif time_left_to_train < 50:
                    run_num = 0
                    print("###stop###")

                else:
                    if change_flag:
                        run_num = 3
                        print("###3###")

                change_flag = False
                run_num = run_num - 1



                # if a_time * 5 + 30 >= time_left_to_train:
                #     run_flag = False
                # train_count = train_count + 1
                # if train_count > 25:
                #     run_flag = False
                # if train_count < 4:
                #     run_flag = True
                # if time_left_to_train / a_time < 3:
                #     run_flag = False

        # train(X_train, self.y, self.config)

        gc.collect()

        del X_train
        gc.collect()


        # X = X[X.index.str.startswith("test")]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)

        gc.collect()

        result = predict(X, self.config)

        return pd.Series(result)


import random
def sample_indices(y):
    pos_index = np.where(y.ravel() == 1)[0].tolist()
    neg_index = np.where(y.ravel() == 0)[0].tolist()

    sample_num = min(len(pos_index), len(neg_index))

    if len(pos_index) < len(neg_index) and 2 * sample_num < len(neg_index):

        p_indices = random.sample(pos_index, sample_num)
        n_indices = random.sample(neg_index, 2 * sample_num)

    elif len(pos_index) > len(neg_index) and 2 * sample_num < len(pos_index):
        p_indices = random.sample(pos_index, 2 * sample_num)
        n_indices = random.sample(neg_index, sample_num)

    else:
        p_indices = random.sample(pos_index, sample_num)
        n_indices = random.sample(neg_index, sample_num)

    return p_indices + n_indices


def data_sample_for_train(X, y):

    sampled_indices = sample_indices(y)

    return X.iloc[sampled_indices, :], y[sampled_indices]


