import numpy as np
from tqdm import tqdm
from collections import Counter
import config
from training.rule import Rule3, Rule2, Rule1

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'



class LPCFG:
    def __init__(self):
        self.rule1s = dict()
        self.rule3s = dict()
        self.pi = dict()
        self.populate()
        self.normalize_rules(self.rule3s)
        self.normalize_rules(self.rule1s)

    def populate(self):
        I, O = config.I, config.O
        pI, pO = Counter(), Counter()
        config.pI, config.pO = pI, pO
        for tree in tqdm(config.train, desc='Constructing L-PCFG'):
            for node in tree.postorder():
                if len(node) == 2:
                    a, b, c = node.label(), node[0].label(), node[1].label()
                    Zi = O[a][pO[a]]
                    pO[a] += 1
                    Yj = I[b][pI[b]]
                    pI[b] += 1
                    Yk = I[c][pI[c]]
                    pI[c] += 1
                    r = Rule3(a, b, c)
                    if r not in self.rule3s:
                        self.rule3s[r] = np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                    else:
                        self.rule3s[r] += np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                elif len(node) == 1:
                    a, x = node.label(), node[0]
                    Z = O[a][pO[a]]
                    pO[a] += 1
                    r = Rule1(a, x)
                    if r not in self.rule1s:
                        self.rule1s[r] = Z
                    else:
                        self.rule1s[r] += Z
                else:
                    raise RuntimeError
            a = node.label()
            if a not in self.pi:
                self.pi[a] = I[a][pI[a]]
            else:
                self.pi[a] += I[a][pI[a]]
            pI[a] += 1
        for a, param in self.pi.items():
            self.pi[a] = param / len(config.train)


    def normalize_rules(self, rules):
        pcfg = config.pcfg
        for rule, param in rules.items():
            rules[rule] = param / pcfg.nonterminals[rule.a]
