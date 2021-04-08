from numba.core import types
from numba.typed import Dict, List
from numba import njit
from parsing.util import hash_forward, hash_backward, Tij, Tik, Tjk
import numpy as np
import warnings

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


warnings.simplefilter('ignore')


@njit
def fill_inside_base(inside, terminals, N, r1, r1_lookup, constrains):
    for i in range(N):
        for rule in r1_lookup[terminals[i]]:
            a, _, _ = hash_backward(rule)
            if a not in constrains[i][i]:
                continue
            inside[i][i][a] = r1[rule]


@njit
def fill_inside(inside, N, r3, r3_lookupC, constrains):
    for length in range(2, N + 1):
        for i in range(N - length + 1):
            j = i + length - 1
            for k in range(i, j):
                if len(inside[k + 1][j]) == 0 or len(inside[i][k]) == 0:
                    continue
                for c in inside[k+1][j]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if b not in inside[i][k]:
                            continue
                        if a not in constrains[i][j]:
                            continue
                        res = Tjk(r3[rule], inside[i][k][b], inside[k+1][j][c])
                        if a not in inside[i][j]:
                            inside[i][j][a] = res
                        else:
                            inside[i][j][a] += res


@njit
def fill_outside_base(outside, inside, N, pi):
    for nonterm, prob in pi.items():
        if nonterm not in inside[0][N-1]:
            continue
        outside[0][N-1][nonterm] = prob


@njit
def fill_outside(outside, inside, N, r3, r3_lookupC):
    for length in range(N - 1, 0, -1):
        for i in range(N - length + 1):
            j = i + length - 1
            if len(inside[i][j]) == 0:
                continue
            for k in range(i):
                if len(outside[k][j]) == 0 or len(inside[k][i - 1]) == 0:
                    continue
                for c in inside[i][j]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if a not in outside[k][j]:
                            continue
                        if b not in inside[k][i-1]:
                            continue
                        res = Tij(r3[rule], outside[k][j][a], inside[k][i-1][b])
                        if c not in outside[i][j]:
                            outside[i][j][c] = res
                        else:
                            outside[i][j][c] += res
            for k in range(j + 1, N):
                if len(outside[i][k]) == 0 or len(inside[j+1][k]) == 0:
                    continue
                for c in inside[j+1][k]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if a not in outside[i][k]:
                            continue
                        if b not in inside[i][j]:
                            continue
                        res = Tik(r3[rule], outside[i][k][a], inside[j+1][k][c])
                        if b not in outside[i][j]:
                            outside[i][j][b] = res
                        else:
                            outside[i][j][b] += res


float_array = types.float64[:]
@njit
def make_array_chart(N):
    outer = List()
    for i in range(N):
        inner = List()
        outer.append(inner)
        for j in range(N):
            d = Dict.empty(key_type=types.int64, value_type=float_array)
            inner.append(d)
    return outer

int_tuple = types.UniTuple(types.int64, 3)
@njit
def make_tuple_chart(N):
    outer = List()
    for i in range(N):
        inner = List()
        outer.append(inner)
        for j in range(N):
            d = Dict.empty(key_type=types.int64, value_type=int_tuple)
            inner.append(d)
    return outer

@njit
def make_chart(N):
    outer = List()
    for i in range(N):
        inner = List()
        outer.append(inner)
        for j in range(N):
            d = Dict.empty(key_type=types.int64, value_type=types.float64)
            inner.append(d)
    return outer

@njit
def fill_marginal(marginal, inside, outside, N):
    for length in range(1, N + 1):
        for i in range(N - length + 1):
            j = i + length - 1
            for nonterm, o_score in outside[i][j].items():
                if nonterm not in inside[i][j]:
                    continue
                score = abs(np.dot(o_score, inside[i][j][nonterm]))
                marginal[i][j][nonterm] = score

@njit
def get_parse_chart(marginal, N, r3_lookupC):
    # {A, (B, C, k)} k is the splitting point
    parse_chart = make_tuple_chart(N)
    score_chart = make_chart(N)
    for i in range(N):
        for nonterm, score in marginal[i][i].items():
            parse_chart[i][i][nonterm] = (-1, -1, -1)
            score_chart[i][i][nonterm] = score
    for length in range(2, N+1):
        for i in range(N - length + 1):
            j = i + length - 1
            for k in range(i, j):
                if len(parse_chart[k + 1][j]) == 0 or len(parse_chart[i][k]) == 0:
                    continue
                for c in parse_chart[k+1][j]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if b not in parse_chart[i][k]:
                            continue
                        if a not in marginal[i][j]:
                            continue
                        score = marginal[i][j][a] + score_chart[i][k][b] + score_chart[k + 1][j][c]
                        if a not in parse_chart[i][j] or score_chart[i][j][a] < score:
                            parse_chart[i][j][a] = (b, c, k)
                            score_chart[i][j][a] = score
    return parse_chart, score_chart


@njit
def get_parse_chart_(marginal, N, r3_lookupC):
    # {A, (B, C, k)} k is the splitting point
    parse_chart = List()
    for i in range(N):
        inner = List()
        for j in range(N):
            inner.append((-1, -1))
        parse_chart.append(inner)
    score_chart = List()
    for i in range(N):
        inner = List()
        for j in range(N):
            inner.append(0.0)
        score_chart.append(inner)
    for i in range(N):
        # parse_chart[i][i] = -1
        # score_chart[i][i] = 0
        for nonterm, score in marginal[i][i].items():
            if score > score_chart[i][i]:
                parse_chart[i][i] = (nonterm, -1)
                score_chart[i][i] = score
    for length in range(2, N+1):
        for i in range(N - length + 1):
            j = i + length - 1
            # parse_chart[i][j] = -1
            # score_chart[i][j] = 0
            for nonterm, score in marginal[i][j].items():
                if score > score_chart[i][j]:
                    parse_chart[i][j] = (nonterm, -1)
                    score_chart[i][j] = score
            best_score = 0
            best_split = -1
            for k in range(i, j):
                if score_chart[k + 1][j] == 0 or score_chart[i][k] == 0:
                    continue
                if score_chart[k + 1][j] + score_chart[i][k] > best_score:
                    best_score = score_chart[k + 1][j] + score_chart[i][k]
                    best_split = k
            if best_split == -1:
                score_chart[i][j] = 0
            score_chart[i][j] += best_score
            parse_chart[i][j] = (parse_chart[i][j][0], best_split)
    return parse_chart, score_chart

@njit
def constrained(terminals, r3, r1, pi, r3_lookupC, r1_lookup, constrains):
    N = len(terminals)
    inside = make_array_chart(N)
    outside = make_array_chart(N)
    marginal = make_chart(N)
    fill_inside_base(inside, terminals, N, r1,  r1_lookup, constrains)
    fill_inside(inside, N, r3, r3_lookupC, constrains)
    fill_outside_base(outside, inside, N, pi)
    fill_outside(outside, inside, N, r3, r3_lookupC)
    fill_marginal(marginal, inside, outside, N)
    return marginal
    # return get_parse_chart(marginal, N, r3_lookupC)
