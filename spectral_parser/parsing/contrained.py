from numba.core import types
from numba.typed import Dict, List
from numba import njit
from parsing.util import hash_backward, Tij, Tik, Tjk
import numpy as np
import warnings

warnings.simplefilter('ignore')


@njit
def fill_inside_base(inside, terminals, N, r1, r1_lookup, constrains, embeddings, total, I, O, inside_param, embed_param):
    for i in range(N):
        for rule in r1_lookup[terminals[i]]:
            a, _, _ = hash_backward(rule)
            if a not in constrains[i][i]:
                continue
            embed_param[i][i][0] = embeddings[i]
            inside_param[i][i][a] = I[a].dot(embeddings[i])
            if N - 1 == 0:
                vec = np.zeros(768)
                vec[0] = 1
                inside[i][i][a] = r1[rule] * O[a].dot(vec)
            else:
                inside[i][i][a] = r1[rule] * O[a].dot((total - embeddings[i]) / (N - 1))


@njit
def fill_inside(inside, N, r3, r3_lookupC, constrains, total, I, O, inside_param, outside_param, embed_param):
    for length in range(2, N + 1):
        for i in range(N - length + 1):
            j = i + length - 1
            i_embed, o_embed = np.empty(1), np.empty(1)
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
                        if len(embed_param[i][j]) == 0:
                            embed_param[i][j][0] = embed_param[i][k][0] + embed_param[k+1][j][0]
                            i_embed = embed_param[i][j][0] / length
                            if N - length == 0:
                                o_embed = np.zeros(768)
                                o_embed[0] = 1
                            else:
                                o_embed = (total - embed_param[i][j][0]) / (N - length)
                        if a not in inside_param[i][j]:
                            inside_param[i][j][a] = I[a].dot(i_embed)
                            outside_param[i][j][a] = O[a].dot(o_embed)
                        res = r3[rule] * Tjk(outside_param[i][j][a], inside_param[i][k][b], inside_param[k+1][j][c],  inside[i][k][b], inside[k+1][j][c])
                        if a not in inside[i][j]:
                            inside[i][j][a] = res
                        else:
                            inside[i][j][a] += res


@njit
def fill_outside_base(outside, inside, N, pi, inside_param):
    for nonterm, prob in pi.items():
        if nonterm not in inside[0][N-1]:
            continue
        outside[0][N-1][nonterm] = prob * inside_param[0][N-1][nonterm]


@njit
def fill_outside(outside, inside, N, r3, r3_lookupC, inside_param, outside_param):
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
                        res = r3[rule] * Tij(outside_param[k][j][a], inside_param[k][i-1][b], inside_param[i][j][c], outside[k][j][a], inside[k][i-1][b])
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
                        res = r3[rule] * Tik(outside_param[i][k][a], inside_param[i][j][b], inside_param[j+1][k][c], outside[i][k][a], inside[j+1][k][c])
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
def constrained(terminals, embeddings, total, r3, r1, pi, I, O, r3_lookupC, r1_lookup, constrains):
    N = len(terminals)
    inside = make_array_chart(N)
    outside = make_array_chart(N)
    inside_param = make_array_chart(N)
    outside_param = make_array_chart(N)
    embed_param = make_array_chart(N)
    marginal = make_chart(N)
    fill_inside_base(inside, terminals, N, r1,  r1_lookup, constrains, embeddings, total, I, O, inside_param, embed_param)
    fill_inside(inside, N, r3, r3_lookupC, constrains, total, I, O, inside_param, outside_param, embed_param)
    fill_outside_base(outside, inside, N, pi, inside_param)
    fill_outside(outside, inside, N, r3, r3_lookupC, inside_param, outside_param)
    fill_marginal(marginal, inside, outside, N)
    return marginal
