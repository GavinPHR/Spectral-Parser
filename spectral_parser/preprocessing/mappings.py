"""
Nonterminal and terminal mappings as described in section
3.2.1 in my dissertation.

These data structures are two-way mappings:
you can index from integers to strings and vice versa.
"""
import collections

from tqdm import tqdm
import config

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


class NonterminalMap:

    def __init__(self, trees):
        self.nonterm2int = dict()
        self.int2nonterm = dict()
        self.populate(trees)

    def populate(self, trees):
        i = 0
        for tree in tqdm(trees, desc='Nonterminal mappings'):
            for node in tree.postorder():
                if node.label() not in self.nonterm2int:
                    self.nonterm2int[node.label()] = i
                    self.int2nonterm[i] = node.label()
                    i += 1
    
    def __getitem__(self, item):
        if type(item) == str:
            return self.nonterm2int[item]
        elif type(item) == int:
            return self.int2nonterm[item]
        else:
            raise RuntimeError('Item has incorrect type.')
    
    def __len__(self):
        return len(self.nonterm2int)

    def __contains__(self, item):
        if type(item) == str:
            return item in self.nonterm2int
        elif type(item) == int:
            return item in self.int2nonterm
        else:
            raise RuntimeError('Item has incorrect type.')


class TerminalMap:

    def __init__(self, trees, start_index):
        self.term2int = dict()
        self.int2term = dict()
        self.acc = start_index
        self.populate(trees)

    def populate(self, trees):
        term_count = collections.Counter()
        for tree in trees:
            term_count.update(tree.leaves())
        for term, count in term_count.items():
            if count <= config.terminal_cutoff:
                continue
            self.term2int[term] = self.acc
            self.int2term[self.acc] = term
            self.acc += 1

    def update_UNK(self, UNK):
        assert (type(UNK) == str)
        if UNK not in self.term2int:
            self.term2int[UNK] = self.acc
            self.int2term[self.acc] = UNK
            self.acc += 1

    def __getitem__(self, item):
        if type(item) == str:
            return self.term2int[item]
        elif type(item) == int:
            return self.int2term[item]
        else:
            raise RuntimeError('Item has incorrect type.')

    def __len__(self):
        return len(self.term2int)

    def __contains__(self, item):
        if type(item) == str:
            return item in self.term2int
        elif type(item) == int:
            return item in self.int2term
        else:
            raise RuntimeError('Item has incorrect type.')
