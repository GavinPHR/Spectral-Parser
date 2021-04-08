"""
Function to read in parse trees and putting them in
parented tree structure for easier feature extraction.

If you are using the PTB WSJ dataset, make sure it is
sanitized, this script only binarize and markovize the trees.

You need to rewrite this file if other datasets are used.
"""
from tqdm import tqdm
from nltk.tree import ParentedTree, Tree

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


class ParentedNormalTree(ParentedTree):

    def raw_label(self):
        """
        Returns the label of a node without any
        parent/sibling annotations.
        """
        label = self.label()
        i = label.rfind('+')
        if i != -1:
            return label[i + 1:]
        i = label.find('|')
        if i != -1:
            return label[:i + 2] + '>'
        i = label.find('^')
        if i != -1:
            return label[:i]
        return label

    def postorder(self, tree=None):
        """
        Generate the subtrees (non-terminals) in postorder.
        """
        if tree is None:
            tree = self
        for subtree in tree:
            if isinstance(subtree, Tree):
                yield from self.postorder(subtree)
        yield tree

    def preorder(self, tree=None):
        """
        Generate the subtrees (non-terminals) in postorder.
        """
        if tree is None:
            tree = self
        yield tree
        for subtree in tree:
            if isinstance(subtree, Tree):
                yield from self.preorder(subtree)

    def __hash__(self):
        return id(self)

def lower(tree):
    for node in tree.postorder():
        if not isinstance(node[0], Tree):
            node[0] = node[0].lower()

def vmarkov(tree):
    """
    Vertical markovization, slightly complicated as
    we want a (h=1, v<1) Markovization.
    See Chapter 5 of my dissertation.
    """
    stack = [(tree, None)]
    while stack:
        node, parent = stack.pop()
        if not isinstance(node[0], Tree):
            if '+' not in node.label():
                node.set_label(node.label() + '^<' + parent + '>')
            continue
        if parent is None:
            parent = node.raw_label()
            stack.append((node[0], parent))
            stack.append((node[1], parent))
            continue
        if '+' not in node.label() and '|' not in node.label():
            node.set_label(node.label() + '^<' + parent + '>')
            parent = node.raw_label()
        if '+' in node.label():
            parent = node.raw_label()
        stack.append((node[0], parent))
        stack.append((node[1], parent))


def read(file):
    trees = []
    with open(file, 'r') as f:
        length = sum(1 for line in f)
    with open(file, 'r') as f:
        for line in tqdm(f, total=length, desc='Reading files'):
            t = Tree.fromstring(line)
            t = t[0]  # Remove TOP
            t.chomsky_normal_form(factor='left', horzMarkov=1, vertMarkov=0)  # Binarization
            t.collapse_unary(collapsePOS=True, collapseRoot=True)  # Collapse ALL unary rules
            t = ParentedNormalTree.convert(t)
            vmarkov(t)
            trees.append(t)
    return trees
