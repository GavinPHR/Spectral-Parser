from collections import Counter, defaultdict
from math import sqrt

import config
from preprocessing.transforms import transform_trees, inverse_transform_trees
from tqdm import tqdm
import numpy as np
from scipy.sparse import dok_matrix

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


inverse_transform_trees(config.train)

def outside(node, child, level=3):
    u = []
    if node.parent() and level != 1:
        u = outside(node.parent(), node, level-1)
    res = []
    if node[0] is child:
        res.append('()' + node.label() + '(' + node[1].label() + ')')
    else:
        res.append('(' + node[0].label() + ')' + node.label() + '()')
    for a in u:
        res.append(res[0] + '^' + a)
    return res

def inside(node):
    res = []
    if len(node) == 1:
        res.append(node.label() + ' ' + node[0].lower())
    else:
        # res.append(node.label() + ' (' + node[0].label() + ' ' + node[1].label() + ')')
        if len(node[0]) == 1:
            l = node[0].label() + ' ' + node[0][0].lower()
        else:
            l = node[0].label() + ' (' + node[0][0].label() + ' ' + node[0][1].label() + ')'
        res.append(node.label() + ' ((' + l + ') ' + node[1].label() + ')')
        if len(node[1]) == 1:
            r = node[1].label() + ' ' + node[1][0].lower()
        else:
            r = node[1].label() + ' (' + node[1][0].label() + ' ' + node[1][1].label() + ')'
        res.append(node.label() + ' (' + node[0].label() + ' (' + r + '))')
    return res

inside_count = defaultdict(Counter)
outside_count = defaultdict(Counter)
inside_idx = defaultdict(dict)
outside_idx = defaultdict(dict)
I_F, O_F = defaultdict(list), defaultdict(list)

def count_features(node):
    label = node.label()
    features = [inside_idx[label].setdefault(f, len(inside_idx[label])) for f in inside(node)]
    I_F[label].append(features)
    for f in features:
        inside_count[label][f] += 1
    if node.parent() is None:
        TOP = outside_idx[label].setdefault('TOP', len(outside_idx[label]))
        outside_count[label][TOP] += 1
        O_F[label].append([TOP])
    else:
        features = [outside_idx[label].setdefault(f, len(outside_idx[label])) for f in outside(node.parent(), node)]
        O_F[label].append(features)
        for f in features:
            outside_count[label][f] += 1

for tree in tqdm(config.train, desc='Counting features'):
    for node in tree.postorder():
        count_features(node)


def scale(M, c):
    return sqrt(M / (c + 5))


I, O = dict(), dict()

# Initialization
for nt, count in config.pcfg.nonterminals.items():
    nonterm = config.nonterminal_map[nt]
    I[nt] = dok_matrix((count, len(inside_count[nonterm])+1), dtype=np.float32)
    O[nt] = dok_matrix((count, len(outside_count[nonterm])+1), dtype=np.float32)

for nt, count in tqdm(config.pcfg.nonterminals.items(), desc='Constructing Sparse'):
    nonterm = config.nonterminal_map[nt]
    for i, fs in enumerate(I_F[nonterm]):
        for f in fs:
            I[nt][i, f] = scale(M=count, c=inside_count[nonterm][f])
    for i, fs in enumerate(O_F[nonterm]):
        for f in fs:
            O[nt][i, f] = scale(M=count, c=outside_count[nonterm][f])

for k, v in I.items():
    I[k] = v.tocsr()

for k, v in O.items():
    O[k] = v.tocsr()

transform_trees(config.train)

del inside_count, outside_count, inside_idx, outside_idx, I_F, O_F
config.I, config.O = I, O
