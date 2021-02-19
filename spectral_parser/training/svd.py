from scipy.sparse.linalg import svds
import numpy as np
from tqdm import tqdm
import config


states = dict()
for nt, count in config.pcfg.nonterminals.items():
    if count <= 10:
        states[nt] = 1
    elif count <= 100:
        states[nt] = 10
    elif count <= 1000:
        states[nt] = 20
    elif count <= 10000:
        states[nt] = 30
    else:
        states[nt] = 40

# I, O = dict(), dict()
# I_feature, O_feature = config.I_feature, config.O_feature
I, O = config.I, config.O
info = []
for nt, count in tqdm(config.pcfg.nonterminals.items(), desc='Doing SVDs'):
    state = states[nt]
    sigma = I[nt].T.dot(O[nt]) / config.pcfg.nonterminals[nt]
    u, s, vt = svds(sigma, k=min(min(sigma.shape) - 1, config.max_state))
    s = s[::-1]
    i = 1
    while i < len(s) and s[i] > 0.01 and i < config.max_state:
        i += 1
    info.append((config.nonterminal_map[nt], i))
    s = np.reciprocal(s[:i]).reshape(-1, 1)
    I[nt] = I[nt].dot(u[:, :i])
    O[nt] = O[nt].dot((s*vt[:i]).T)
print(info)
# config.I, config.O = I, O
