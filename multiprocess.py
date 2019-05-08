import torch
import time
from numba import *
import torch.multiprocessing as mp
import timeit


def function(x):
    res = []
    for i in range(N):
        out = x * mask[i].type(torch.float32)
        out = out[~mask[i]]
        out = linear[i](out)
        res.append(out)

@jit
def function2(x):
    res = []
    for i in range(N):
        out = x * mask[i].type(torch.float32)
        out = out[~mask[i]]
        out = linear[i](out)
        res.append(out)

@jit
def each(x, mask, linear):
    out = x * mask.type(torch.float32)
    out = out[~mask]
    out = linear(out)
    return out


if __name__ == '__main__':

    N = 50
    x = torch.randn(100, 100)
    mask, linear, x_list = [], [], []
    for i in range(N):
        x_list.append(x)
        m = (torch.randn(100, 100) > 0.1)
        mask.append(m)
        size = x[~m]
        linear.append(torch.nn.Linear(size.shape[0], 8 ** 2))

    ##### Method 1 #################
    start = time.time()
    function(x)

    print(time.time()-start)

    ##### Method 2 #################
    start = time.time()
    function2(x)
    print(time.time()-start)


    ##### Method 3 #################
    res = []
    for i in range(N):
        res.append(each(x_list[i], mask[i], linear[i]))

    print(time.time()-start)