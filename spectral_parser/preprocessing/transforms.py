"""
Functions for transforming the labels in trees
from strings to integers, and vice versa.
"""
from tqdm import tqdm
import config
from preprocessing.unk import signature

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


def transform_trees(trees):
    """
    Transform trees to int labels.
    """
    nmap = config.nonterminal_map
    tmap = config.terminal_map
    for tree in tqdm(trees, desc='Transform from strs to ints'):
        i = 0
        for node in tree.postorder():
            if isinstance(node[0], str):
                # Replace the word by an UNK as described in
                # Chapter 6 of my dissertation
                if node[0] not in tmap:
                    tag = signature(node[0], i, node[0].lower() in tmap)
                    if tag in tmap:
                        node[0] = tmap[tag]
                    else:
                        tmap.update_UNK(tag)
                        node[0] = tmap[tag]
                else:
                    node[0] = tmap[node[0]]
                i += 1
            node.set_label(nmap[node.label()])


def inverse_transform_trees(trees):
    nmap = config.nonterminal_map
    tmap = config.terminal_map
    for tree in tqdm(trees, desc='Inverse transform from ints to strs'):
        for node in tree.postorder():
            node.set_label(nmap[node.label()])
            if isinstance(node[0], int):
                node[0] = tmap[node[0]]
