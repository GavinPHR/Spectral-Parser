from tqdm import tqdm
from nltk.tree import ParentedTree, Tree
import config


class ParentedNormalTree(ParentedTree):

    def raw_label(self):
        label = self.label()
        i = label.rfind('+')
        if i != -1:
            return label[i + 1:]
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

def read(file):
    trees = []
    with open(file, 'r') as f:
        length = sum(1 for line in f)
    with open(file, 'r') as f:
        for line in tqdm(f, total=length, desc='Reading files'):
            t = Tree.fromstring(line)
            t = t[0]  # Remove TOP
            t.chomsky_normal_form(factor='left', horzMarkov=0, vertMarkov=0)  # Binarization
            t.collapse_unary(collapsePOS=True, collapseRoot=True)  # Collapse ALL unary rules
            t = ParentedNormalTree.convert(t)
            lower(t)  # Lower case the terminals
            trees.append(t)
    return trees
