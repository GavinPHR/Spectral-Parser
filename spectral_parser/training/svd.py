from scipy.linalg import svd
import numpy as np
from tqdm import tqdm
import config


states = dict()
for nt in config.pcfg.nonterminals:
    states[nt] = config.max_state

I, O = dict(), dict()
I_feature, O_feature = config.I_feature, config.O_feature
info = []
for nt, count in tqdm(config.pcfg.nonterminals.items(), desc='Doing SVDs'):
    state = states[nt]
    sigma = I_feature[nt].T.dot(O_feature[nt]) / config.pcfg.nonterminals[nt]
    u, s, vt = svd(sigma)
    i = 1
    while i < len(s) and s[i] > config.min_singular_value and i < state:
        i += 1
    info.append((config.nonterminal_map[nt], i))
    s = np.reciprocal(s[:i]).reshape(-1, 1)
    I[nt] = u.T[:i]
    O[nt] = s*vt[:i]
print(info)
config.I, config.O = I, O
