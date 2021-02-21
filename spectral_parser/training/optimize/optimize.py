import numpy as np
import config
from math import sqrt
from training.lpcfg_smoothed import LPCFG_Smoothed
from training.optimize.lpcfg_surrogate import LPCFG_Surrogate


class LPCFG_Optimize(LPCFG_Smoothed):

    def smooth(self, L, C):
        pcfg = config.pcfg
        for rule, count in pcfg.rule3s_count.items():
            e1 = self.Eijk[rule][:L[rule.a], :L[rule.b], :L[rule.c]]
            eij, eik, ejk = self.Eij[rule][:L[rule.a], :L[rule.b]], self.Eik[rule][:L[rule.a], :L[rule.c]], self.Ejk[rule][:L[rule.b], :L[rule.c]]
            ei, ej, ek = self.Ei[rule][:L[rule.a]], self.Ej[rule][:L[rule.b]], self.Ek[rule][:L[rule.c]]
            e2 = (np.einsum('ij,k->ijk', eij, ek) + np.einsum('ik,j->ijk', eik, ej) + np.einsum('jk,i->ijk', ejk, ei)) / 3
            e3 = np.einsum('i,j,k->ijk', ei, ej, ek)
            hi, fj, fk = self.H[rule.a][:L[rule.a]], self.F[rule.b][:L[rule.b]], self.F[rule.c][:L[rule.c]]
            e4 = np.einsum('i,j,k->ijk', hi, fj, fk)
            lambda_ = sqrt(count) / (C + sqrt(count))
            k = lambda_ * e3 + (1 - lambda_) * e4
            e = lambda_ * e1 + (1 - lambda_) * (lambda_ * e2 + (1 - lambda_) * k)
            config.lpcfg.rule3s[rule] = pcfg.rule3s[rule] * e
        for rule, count in pcfg.rule2s_count.items():
            e1 = self.Eij[rule][:L[rule.a], :L[rule.b]]
            ei, ej = self.Ei[rule][:L[rule.a]], self.Ej[rule][:L[rule.b]]
            e2 = np.einsum('i,j->ij', ei, ej)
            hi, fj = self.H[rule.a][:L[rule.a]], self.F[rule.b][:L[rule.b]]
            e3 = np.einsum('i,j->ij', hi, fj)
            lambda_ = sqrt(count) / (C + sqrt(count))
            e = lambda_ * e1 + (1 - lambda_) * (lambda_ * e2 + (1 - lambda_) * e3)
            config.lpcfg.rule2s[rule] = pcfg.rule2s[rule] * e
        for rule, count in pcfg.rule1s_count.items():
            config.lpcfg.rule1s[rule] = pcfg.rule1s[rule] * self.Eax[rule][:L[rule.a]]
        config.lpcfg.pi = self.pi

    @staticmethod
    def get_length(incutoff, instates, precutoff, prestates):
        L = dict()
        for nt in config.pcfg.nonterminals:
            if nt in config.pcfg.preterminals:
                cutoff, states = precutoff, prestates
            else:
                cutoff, states = incutoff, instates
            s = config.S[nt]
            i = 1
            while i < len(s) and s[i] > cutoff and i < states:
                i += 1
            L[nt] = i
        return L


    def opt(self, incutoff, instates, precutoff, prestates, C):
        instates, prestates = int(instates), int(prestates)
        config.lpcfg = LPCFG_Surrogate()
        self.smooth(self.get_length(incutoff, instates, precutoff, prestates), C)
        config.save()

        from parsing import parser
        import multiprocessing as mp
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            pass
        parser.parse_devset(config.dev_file)

        import subprocess
        subprocess.Popen(['cd', 'parsing'])
        subprocess.Popen(['python3', 'parser.py'])
        subprocess.Popen(['cd', '..'])
        process = subprocess.Popen(['./evalb', '-p', 'new.prm', config.dev_file, 'output/baseline_parse.txt'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = str(stdout)
        i = output.find('FMeasure       =  ')
        return float(output[i + 18:i + 23])
