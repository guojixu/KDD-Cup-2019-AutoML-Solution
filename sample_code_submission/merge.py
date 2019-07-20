import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import CONSTANT
from util import Config, Timer, log, timeit

NUM_OP = [np.std, np.mean]

def bfs(root_name, graph, tconfig):
    tconfig[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@timeit
def join(u, v, v_name, key, type_):

    def func_rename(x):
        if 'mv' in x[1]:
            return f'{CONSTANT.MULTI_CAT_PREFIX}{x[1]}({x[0]})'
        elif 'cat_union' in x[1]:
            return f'{CONSTANT.MULTI_CAT_PREFIX}{x[1]}({x[0]})'
        elif 'cat_last' in x[1]:
            return f'{CONSTANT.CATEGORY_PREFIX}{x[1]}({x[0]})'
        elif CONSTANT.NUMERICAL_PREFIX in x[0]:
            return f"{CONSTANT.NUMERICAL_PREFIX}{x[1]}({x[0]})"
        elif CONSTANT.CATEGORY_PREFIX in x[0]:
            return f"{CONSTANT.CATEGORY_PREFIX}{x[1]}({x[0]})"

    if type_.split("_")[2] == 'many':
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key[0]
                     and not col.startswith(CONSTANT.TIME_PREFIX)
                     # and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)
        }
        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(func_rename)
        # v.columns = v.columns.map(lambda a:
        #         f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]})")
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)


from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from preprocess import abortable_worker
import multiprocessing as mp

pool_maxtasksperchild = 100

# @timeit
# def temporal_join_mulp(u, v, v_name, key, time_col, type_):
#
#     if type_.split("_")[2] == 'many':
#         timer = Timer()
#
#         if isinstance(key, list):
#             assert len(key) == 1
#             key = key[0]
#
#         tmp_u = u[[time_col, key]]
#         timer.check("select")
#         # print(tmp_u)
#
#         tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
#         timer.check("concat")
#         # print(tmp_u)
#
#         rehash_key = f'rehash_{key}'
#         tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
#         timer.check("rehash_key")
#         # print(tmp_u)
#
#         tmp_u.sort_values(time_col, inplace=True)
#         timer.check("sort")
#         # print(tmp_u)
#
#         # tmp_u = tmp_u.groupby(key).fillna(method='ffill')
#         # timer.check("group & ffill")
#
#         ############################################################
#         # new_groupby
#
#         tmp_u_grp_df = tmp_u.groupby(key)
#
#         indices = list(range(len(tmp_u)))
#
#         id_pair = list(zip(indices, tmp_u))
#
#         ctx = mp.get_context('forkserver')
#
#         pool = ctx.Pool(processes=4, maxtasksperchild=pool_maxtasksperchild)
#         result = []
#
#         for idx, grp in id_pair:
#             s1 = c_df[col[0]]
#             s2 = c_df[col[1]]
#
#             abortable_func = partial(abortable_worker, c_two_order_grpcnt_key_func)
#             res = pool.apply_async(abortable_func, args=(idx, col, s1, s2))
#
#             result.append(res)
#
#
#         pool.close()
#         pool.join()
#
#
#         ############################################################
#
#
#
#         # print(tmp_u)
#
#         tmp_u = tmp_u.loc['u']
#         # print(tmp_u)
#         tmp_u.sort_index(inplace=True)
#         # print(tmp_u)
#         needed_cols = [col for col in tmp_u if col != key and col != rehash_key and col != time_col]
#         tmp_u = tmp_u[needed_cols]
#         tmp_u.columns = tmp_u.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}_TMJOIN.{a}")
#         timer.check("get tmp_u to cnocat")
#         # print(tmp_u)
#
#         if tmp_u.empty:
#             log("empty tmp_u, return u")
#             return u
#
#         ret = pd.concat([u, tmp_u], axis=1, sort=False)
#         timer.check("final concat")
#         del tmp_u
#     else:
#         v = v.set_index(key)
#         v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}_TOJOIN.{a}")
#         ret = u.join(v, on=key)
#
#     return ret




@timeit
def temporal_join(u, v, v_name, key, time_col, type_):

    if type_.split("_")[2] == 'many':
        timer = Timer()

        if isinstance(key, list):
            assert len(key) == 1
            key = key[0]

        tmp_u = u[[time_col, key]]
        timer.check("select")
        # print(tmp_u)

        tmp_u = pd.concat([v, tmp_u], keys=['v', 'u'], sort=False)

        # tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)

        timer.check("concat")
        # print(tmp_u)

        rehash_key = f'rehash_{key}'

        hash_max_tmp = tmp_u[key].nunique()
        if hash_max_tmp > 3000:
            tmp = min(int(hash_max_tmp / 10), 3000)
            hash_max = tmp - tmp % 100
        else:
            hash_max = hash_max_tmp - hash_max_tmp % 100
        # hash_max = hash_max_tmp % 10000

        print("#" * 20)
        print(hash_max)
        print("#" * 20)

        tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % hash_max)

        # tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
        timer.check("rehash_key")
        # print(tmp_u)

        tmp_u.sort_values(time_col, inplace=True)
        timer.check("sort")
        # print(tmp_u)

        tmp_u = tmp_u.groupby(rehash_key).fillna(method='ffill')
        timer.check("group & ffill")
        # print(tmp_u)

        tmp_u = tmp_u.loc['u']
        # print(tmp_u)
        tmp_u.sort_index(inplace=True)
        # print(tmp_u)
        needed_cols = [col for col in tmp_u if col != key and col != rehash_key and col != time_col]
        tmp_u = tmp_u[needed_cols]
        tmp_u.columns = tmp_u.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}_TMJOIN.{a}")
        timer.check("get tmp_u to cnocat")
        # print(tmp_u)

        if tmp_u.empty:
            log("empty tmp_u, return u")
            return u

        ret = pd.concat([u, tmp_u], axis=1, sort=False)
        timer.check("final concat")
        del tmp_u
    else:
        v = v.set_index(key)
        v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}_TOJOIN.{a}")
        ret = u.join(v, on=key)

    return ret


    # timer = Timer()
    #
    # if isinstance(key, list):
    #     assert len(key) == 1
    #     key = key[0]
    #
    # tmp_u = u[[time_col, key]]
    # timer.check("select")
    #
    # tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    # timer.check("concat")
    #
    # rehash_key = f'rehash_{key}'
    # tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    # timer.check("rehash_key")
    #
    # tmp_u.sort_values(time_col, inplace=True)
    # timer.check("sort")
    #
    # agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
    #              and not col.startswith(CONSTANT.TIME_PREFIX)
    #              and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}
    #
    # tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    # timer.check("group & rolling & agg")
    #
    # tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    # timer.check("reset_index")
    #
    # tmp_u.columns = tmp_u.columns.map(lambda a:
    #     f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")
    #
    # if tmp_u.empty:
    #     log("empty tmp_u, return u")
    #     return u
    #
    # ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    # timer.check("final concat")
    #
    # del tmp_u
    #
    # return ret
    # return u

def dfs(u_name, config, tables, graph):
    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            continue

        if config['time_col'] in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--t {v_name}")
            # u = temporal_join(u, v, v_name, key, config['time_col'], type_)
            u = join(u, v, v_name, key, type_)
        else:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)
