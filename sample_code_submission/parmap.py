import multiprocessing
import gc
import numpy as np


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


import random


def _data_sample(y, sampled=[]):
    pos_index = np.where(y == 1)[0]
    neg_index = np.where(y == 0)[0]
    neg_index = list(set(neg_index) - set(sampled))

    neg_sample_num = min(int(2.5 * len(pos_index)), len(neg_index))
    sample_neg_index = random.sample(neg_index, neg_sample_num)
    return list(pos_index), list(sample_neg_index)
