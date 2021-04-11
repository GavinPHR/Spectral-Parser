"""
Global config file.
This file is imported in almost every other file.
"""
import os
import pickle

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


"""
File paths and output directory
"""
prefix = '/Users/phr/Desktop/Spectral-Parser/spectral_parser'
train_file = prefix + '/data/train.txt'
test_file = prefix + '/data/dev1.txt'
output_dir = prefix + '/output/'
cache = prefix + '/output/cache/'

# Create the output/cache directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Output directory created.')
if not os.path.exists(cache):
    os.makedirs(cache)
    print('Cache directory created.')

"""
Hyperparameters
"""
prestates = 16       # Number of latent states for preterminals
instates = 13        # Number of latent states for interminals
C = 12.2             # Smoothing parameter
terminal_cutoff = 5  # Word below this frequency are replaced by UNKs
prune_cutoff = 1e-5  # Marginals less than this threshold are pruned

# Number of CPUs to use for multi-processing
CPUs = os.cpu_count()
# Whether to cache prune charts, useful when tuning.
cache_prune_charts = True

"""
The varaibles below should not be manually set.
They are here to prevent lint.
"""
train = None
nonterminal_map = None
terminal_map = None
pcfg = None
I, O = None, None
S = None
lpcfg= None
lpcfg_optimize = None
rule3s_lookupC = None
rule1s_lookup = None
numba_ready = False
rule3s = None
rule1s = None
pi = None

def _save(obj, fname):
    with open(output_dir + fname, 'wb') as f:
        pickle.dump(obj, f)

def _load(fname):
    with open(output_dir + fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save():
    print('Saving parameters.')
    _save(nonterminal_map, 'nonterminal_map.p')
    _save(terminal_map, 'terminal_map.p')
    _save(pcfg, 'pcfg.p')
    _save(lpcfg, 'lpcfg.p')
    _save(rule3s_lookupC, 'rule3s_lookupC.p')
    _save(rule1s_lookup, 'rule1s_lookup.p')
    print('Done!')

def load():
    global nonterminal_map, terminal_map
    global pcfg, lpcfg
    global rule3s_lookupC, rule1s_lookup
    nonterminal_map = _load('nonterminal_map.p')
    terminal_map = _load('terminal_map.p')
    pcfg = _load('pcfg.p')
    lpcfg = _load('lpcfg.p')
    rule3s_lookupC = _load('rule3s_lookupC.p')
    rule1s_lookup = _load('rule1s_lookup.p')
