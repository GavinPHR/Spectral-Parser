from collections import defaultdict
import config
from nltk.tree import Tree
from tqdm import tqdm
import numpy as np

em = config.embedding_map
I, O = defaultdict(list), defaultdict(list)

def rec(node, vecs, total, N):
    if not isinstance(node[0], Tree):
        vec = vecs[id(node)]
        I[node.label()].append(vec)
        if N - 1 == 0:
            vec = np.zeros(768)
            vec[0] = 1
            O[node.label()].append(vec)
        else:
            O[node.label()].append((total - vec) / (N - 1))
        return 1, vec
    l, lvec = rec(node[0], vecs, total, N)
    r, rvec = rec(node[1], vecs, total, N)
    k = l + r
    vec = lvec + rvec
    I[node.label()].append(vec / k)
    if N - k == 0:
        vec = np.zeros(768)
        vec[0] = 1
        O[node.label()].append(vec)
    else:
        O[node.label()].append((total - vec) / (N - k))
    return k, vec

for tree in tqdm(config.train, desc='Collating features'):
    N = len(em[id(tree)])
    total = sum(em[id(tree)].values())
    rec(tree, em[id(tree)], total, N)

for k, v in I.items():
    I[k] = np.vstack(v)
for k, v in O.items():
    O[k] = np.vstack(v)

config.I_feature, config.O_feature = I, O
