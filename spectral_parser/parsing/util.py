from numba import njit, int64
from numba.core.types import UniTuple
import numpy as np


@njit(int64(int64, int64, int64))
def hash_forward(a, b, c):
    return (a << 40) ^ (b << 20) ^ c

@njit(UniTuple(int64, 3)(int64,))
def hash_backward(h):
    # 20bits mask, 1048575 = 2^20-1
    mask = 1048575
    c = h & mask
    b = (h >> 20) & mask
    a = (h >> 40) & mask
    return (a, b, c)


@njit
def Tjk(a, b, c, T1j, T1k):
    ans = np.zeros(len(a))
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                ans[i] += a[i] * b[j] * c[k] * T1j[j] * T1k[k]
    return ans


@njit
def Tij(a, b, c, T1i, T1j):
    ans = np.zeros(len(c))
    for k in range(len(c)):
        for i in range(len(a)):
            for j in range(len(b)):
                ans[k] += a[i] * b[j] * c[k] * T1i[i] * T1j[j]
    return ans


@njit
def Tik(a, b, c, T1i, T1k):
    ans = np.zeros(len(b))
    for j in range(len(b)):
        for i in range(len(a)):
            for k in range(len(c)):
                ans[j] += a[i] * b[j] * c[k] * T1i[i] * T1k[k]
    return ans
