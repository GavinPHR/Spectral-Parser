"""
export LD_LIBRARY_PATH=~/venv/lib
"""
import config
config.load()
from tagger.perceptron import PerceptronTagger
tagger = PerceptronTagger()
import config
from nltk.tree import Tree
from tqdm import tqdm
from numba import njit
from numba.typed import List
from parsing.baseline import prune, get_parse_chart, make_chart
import os
import multiprocessing as mp
from parsing.contrained import constrained
import pickle
import numpy as np
from datetime import datetime

def transform_int2str(tree, sent, i=0):
    tree.set_label(config.nonterminal_map[int(tree.label())])
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        tree[0] = sent[i]
        return i + 1
    else:
        for subtree in tree:
            i = transform_int2str(subtree, sent, i)
    return i

# @njit
def recursive_build(parse_chart, score_chart, i, j, a=-1):
    if a == -1:
        assert(i == 0 and j == len(parse_chart) - 1)
        best_score = -1
        for candidate, score in score_chart[i][j].items():
            if score > best_score:
                best_score = score
                a = candidate
    b, c, k = parse_chart[i][j][a]
    assert(i == j or (i <= k and k < j))
    root = '('+str(a) + ' '
    if i != j:
        root += recursive_build(parse_chart, score_chart, i, k, b)
        root += recursive_build(parse_chart, score_chart, k+1, j, c)
    else:
        root += 'w'
    return root + ')'

def serialize(chart, terminals):
    N = len(chart)
    fname = str(hash(tuple(terminals)))
    pychart = [[dict() for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k, v in chart[i][j].items():
                pychart[i][j][k] = v
    with open(config.cache + fname, 'wb') as f:
        pickle.dump(pychart, f)
    # for testing
    with open(config.cache + 'map', 'a') as f:
        words = [config.terminal_map[x] for x in terminals]
        f.write(fname + ' ' + ' '.join(words) + '\n')

def deserialize(terminals):
    fname = str(hash(tuple(terminals)))
    with open(config.cache + fname, 'rb') as f:
        pychart = pickle.load(f)
    N = len(pychart)
    chart = make_chart(N)
    for i in range(N):
        for j in range(N):
            for k, v in pychart[i][j].items():
                chart[i][j][k] = v
    return chart
import os.path

# @njit
def get_charts(terminals, r3_p, r1_p, pi_p, r3_lookupC, r1_lookup, prune_cutoff, r3_f, r1_f, pi_f):
    if config.cache_prune_charts:
        fname = str(hash(tuple(terminals)))
        if os.path.exists(config.cache + fname):
            constrains = deserialize(terminals)
        else:
            constrains = prune(terminals, r3_p, r1_p, pi_p, r3_lookupC, r1_lookup, prune_cutoff)
            serialize(constrains, terminals)
    else:
        constrains = prune(terminals, r3_p, r1_p, pi_p, r3_lookupC, r1_lookup, prune_cutoff)
    if len(constrains[0][len(constrains) - 1]) == 0:
        return '()'

    # parse_chart, score_chart = get_parse_chart(constrains, len(constrains), r3_lookupC)
    # return recursive_build(parse_chart, score_chart, 0, len(parse_chart) - 1)

    marginal = constrained(terminals, r3_f, r1_f, pi_f, r3_lookupC, r1_lookup, constrains)
    parse_chart, score_chart = get_parse_chart(marginal, len(marginal), r3_lookupC)
    if len(parse_chart[0][len(parse_chart) - 1]) == 0:
        return '()'
    return recursive_build(parse_chart, score_chart, 0, len(parse_chart) - 1)

def process_wrapper(terminals):
    if terminals is None:
        return '()'
    if not config.numba_ready:
        from parsing import prepare_global_param
    return get_charts(List(terminals), config.rule3s_prune,
                        config.rule1s_prune,
                        config.pi_prune,
                        config.rule3s_lookupC,
                        config.rule1s_lookup,
                        config.prune_cutoff,
                        config.rule3s_full,
                        config.rule1s_full,
                        config.pi_full)
from preprocessing.unk import signature
def prepare_args(sent):
    # uncased = [w.lower() for w in sent]
    uncased = sent
    terminals = []
    for i, (wordC, POS) in enumerate(tagger.tag(uncased)):
        # word = wordC.lower()
        word = wordC
        if word not in config.terminal_map.term2int:
            # Fall back to POS tag
            # POS = signature(word, i, word.lower() in config.terminal_map.term2int)
            if POS not in config.terminal_map.term2int:
                terminals.append(config.terminal_map['NN'])
                # terminals.append(config.terminal_map['UNK'])
            else:
                terminals.append(config.terminal_map[POS])
        else:
            terminals.append(config.terminal_map[word])
    else:
        return terminals


def parse_devset(dev_file):
    sents = []
    with open(dev_file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            sents.append(tree.leaves())
    args = list(map(prepare_args, sents))
    cpu = os.cpu_count()
    # now = datetime.now().strftime("-%M-%H-%d-%m")
    now = ''
    with open(config.output_dir + 'parse' + now + '.txt', 'w') as f:
        with mp.Pool(cpu - 2) as pool:
            for i, tree_str in enumerate(tqdm(pool.imap(process_wrapper, args, chunksize=len(sents)//(cpu)), total=len(sents))):
                if tree_str == '()':
                    f.write('()\n')
                else:
                    tree = Tree.fromstring(tree_str)
                    transform_int2str(tree, sents[i])
                    tree.un_chomsky_normal_form()
                    parse = '(TOP ' + tree.pformat(margin=float('inf')) + ')'
                    f.write(parse + '\n')


if __name__ == '__main__':
    mp.set_start_method('fork')
    parse_devset(config.dev_file)
