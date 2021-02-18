from tqdm import tqdm
from nltk.tree import ParentedTree, Tree
import config

class TraversableTree(Tree):

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

    def __hash__(self):
        return id(self)


class ParentedNormalTree(ParentedTree):

    def raw_label(self):
        label = self.label()
        i = label.rfind('+')
        if i != -1:
            return label[i + 1:]
        i = label.find('|')
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

    # @classmethod
    # def convert(cls, tree):
    #     """
    #     Convert a tree between different subtypes of Tree.  ``cls`` determines
    #     which class will be used to encode the new tree.
    #
    #     :type tree: Tree
    #     :param tree: The tree that should be converted.
    #     :return: The new Tree.
    #     """
    #     if isinstance(tree, Tree):
    #         children = [cls.convert(child) for child in tree]
    #         node = cls(tree._label, children)
    #         node.head = tree.head
    #         return node
    #     else:
    #         return tree

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
            t = TraversableTree.fromstring(line)
            t = t[0]  # Remove TOP
            # for node in t.postorder():
            #     node.head = determineHead(node).label()
            # chomsky_normal_form_with_head(t)
            t.chomsky_normal_form(factor='left', horzMarkov=1, vertMarkov=0)  # Binarization
            t.collapse_unary(collapsePOS=True, collapseRoot=True)  # Collapse ALL unary rules
            t = ParentedNormalTree.convert(t)
            trees.append(t)
    return trees
