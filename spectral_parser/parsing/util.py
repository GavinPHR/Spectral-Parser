from numba import njit, int64
from numba.core.types import UniTuple
import numpy as np

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


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
def Tj(T2ij, T1j):
    ans = np.zeros(T2ij.shape[0])
    for i in range(T2ij.shape[0]):
        for j in range(T2ij.shape[1]):
            ans[i] += T2ij[i][j] * T1j[j]
    return ans

@njit
def Ti(T2ij, T1i):
    ans = np.zeros(T2ij.shape[1])
    for j in range(T2ij.shape[1]):
        for i in range(T2ij.shape[0]):
            ans[j] += T2ij[i][j] * T1i[i]
    return ans

@njit
def Tjk(T3ijk, T1j, T1k):
    ans = np.zeros(T3ijk.shape[0])
    for i in range(T3ijk.shape[0]):
        for j in range(T3ijk.shape[1]):
            for k in range(T3ijk.shape[2]):
                ans[i] += T3ijk[i][j][k] * T1j[j] * T1k[k]
    return ans


@njit
def Tij(T3ijk, T1i, T1j):
    ans = np.zeros(T3ijk.shape[2])
    for k in range(T3ijk.shape[2]):
        for i in range(T3ijk.shape[0]):
            for j in range(T3ijk.shape[1]):
                ans[k] += T3ijk[i][j][k] * T1i[i] * T1j[j]
    return ans


@njit
def Tik(T3ijk, T1i, T1k):
    ans = np.zeros(T3ijk.shape[1])
    for j in range(T3ijk.shape[1]):
        for i in range(T3ijk.shape[0]):
            for k in range(T3ijk.shape[2]):
                ans[j] += T3ijk[i][j][k] * T1i[i] * T1k[k]
    return ans
