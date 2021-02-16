from nltk.tree import Tree
from BERT import embed
import config
from tqdm import tqdm
from collections import defaultdict

def retrieve():
    embedding_map = defaultdict(dict)
    for tree in tqdm(config.train, desc='Retrieving embeddings'):
        i = 0
        vecs = embed(tree.leaves())
        vec_map = embedding_map[id(tree)]
        for node in tree.postorder():
            if not isinstance(node[0], Tree):
                vec_map[id(node)] = vecs[i]
                i += 1
    config.embedding_map = embedding_map
