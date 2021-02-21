from scipy.sparse.linalg import svds
import numpy as np
from tqdm import tqdm
import config

config.S = dict()
I, O = config.I, config.O
info = []
for nt, count in tqdm(config.pcfg.nonterminals.items(), desc='Doing SVDs'):
    sigma = I[nt].T.dot(O[nt]) / config.pcfg.nonterminals[nt]
    if nt in config.pcfg.preterminals:
        state = config.prestates
    else:
        state = config.instates
    u, s, vt = svds(sigma, k=min(min(sigma.shape) - 1, state))
    idx = np.argsort(s)[::-1]
    i = 1
    while i < len(idx) and s[idx[i]] > 0 and i < state:
        i += 1
    idx = idx[:i]
    info.append((config.nonterminal_map[nt], i))
    config.S[nt] = s[idx]
    s = np.reciprocal(s[idx]).reshape(-1, 1)
    I[nt] = I[nt].dot(u[:, idx])
    O[nt] = O[nt].dot((s*vt[idx]).T)
