import numpy as np
from tqdm import tqdm
from collections import Counter
import config
from training.rule import Rule3, Rule1
from math import sqrt
from copy import deepcopy

class LPCFG_Smoothed:
    def __init__(self, optimize=False):
        self.rule1s = dict()
        self.rule3s = dict()
        self.pi = dict()
        self.Ei, self.Ej, self.Ek = dict(), dict(), dict()
        self.Eij, self.Eik, self.Ejk = dict(), dict(), dict()
        self.Eijk, self.Eax = dict(), dict()
        self.H, self.F = dict(), dict()
        self.populate()
        self.normalize()
        if not optimize:
            self.smooth()
            self.cleanup()

    def cleanup(self):
        del self.Ei, self.Ej, self.Ek
        del self.Eij, self.Eik, self.Ejk
        del self.Eijk, self.Eax
        del self.H, self.F

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
                    if r not in self.Eijk:
                        self.Eijk[r] = np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                        self.Eij[r] = np.einsum('i,j->ij', Zi, Yj)
                        self.Eik[r] = np.einsum('i,k->ik', Zi, Yk)
                        self.Ejk[r] = np.einsum('j,k->jk', Yj, Yk)
                        self.Ei[r] = deepcopy(Zi)
                        self.Ej[r] = deepcopy(Yj)
                        self.Ek[r] = deepcopy(Yk)
                    else:
                        self.Eijk[r] += np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                        self.Eij[r] += np.einsum('i,j->ij', Zi, Yj)
                        self.Eik[r] += np.einsum('i,k->ik', Zi, Yk)
                        self.Ejk[r] += np.einsum('j,k->jk', Yj, Yk)
                        self.Ei[r] += Zi
                        self.Ej[r] += Yj
                        self.Ek[r] += Yk
                    if a not in self.H:
                        self.H[a] = deepcopy(Zi)
                    else:
                        self.H[a] += Zi
                    if b not in self.F:
                        self.F[b] = deepcopy(Yj)
                    else:
                        self.F[b] += Yj
                    if c not in self.F:
                        self.F[c] = deepcopy(Yk)
                    else:
                        self.F[c] += Yk
                elif len(node) == 1:
                    a, x = node.label(), node[0]
                    Z = O[a][pO[a]]
                    pO[a] += 1
                    r = Rule1(a, x)
                    if r not in self.Eax:
                        self.Eax[r] = deepcopy(Z)
                    else:
                        self.Eax[r] += Z
                    if a not in self.H:
                        self.H[a] = deepcopy(Z)
                    else:
                        self.H[a] += Z
                else:
                    raise RuntimeError
            a = node.label()
            Y = I[a][pI[a]]
            pI[a] += 1
            if a not in self.pi:
                self.pi[a] = deepcopy(Y)
            else:
                self.pi[a] += Y
            if a not in self.F:
                self.F[a] = deepcopy(Y)
            else:
                self.F[a] += Y
        for a, param in self.pi.items():
            self.pi[a] = param / len(config.train)


    def normalize(self):
        pcfg = config.pcfg
        for rule, count in pcfg.rule3s_count.items():
            self.Eijk[rule] /= count
            self.Eij[rule] /= count
            self.Eik[rule] /= count
            self.Ejk[rule] /= count
            self.Ei[rule] /= count
            self.Ej[rule] /= count
            self.Ek[rule] /= count
        for rule, count in pcfg.rule1s_count.items():
            self.Eax[rule] /= count
        for nonterm, count in pcfg.nonterminals.items():
            self.F[nonterm] /= count
            self.H[nonterm] /= count

    def smooth(self):
        pcfg = config.pcfg
        for rule, count in pcfg.rule3s_count.items():
            e1 = self.Eijk[rule]
            eij, eik, ejk = self.Eij[rule], self.Eik[rule], self.Ejk[rule]
            ei, ej, ek = self.Ei[rule], self.Ej[rule], self.Ek[rule]
            e2 = (np.einsum('ij,k->ijk', eij, ek) + np.einsum('ik,j->ijk', eik, ej) + np.einsum('jk,i->ijk', ejk, ei))/3
            e3 = np.einsum('i,j,k->ijk', ei, ej, ek)
            hi, fj, fk = self.H[rule.a], self.F[rule.b], self.F[rule.c]
            e4 = np.einsum('i,j,k->ijk', hi, fj, fk)
            lambda_ = sqrt(count) / (config.C + sqrt(count))
            k = lambda_ * e3 + (1-lambda_) * e4
            e = lambda_ * e1 + (1-lambda_)*(lambda_ * e2 + (1-lambda_) * k)
            self.rule3s[rule] = pcfg.rule3s[rule] * e
        for rule, count in pcfg.rule1s_count.items():
            self.rule1s[rule] = pcfg.rule1s[rule] * self.Eax[rule]

