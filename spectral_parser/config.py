"""
Global config file.
This file is imported in almost every other file.
"""
import os

import torch  # Should remove torch dependency in the future

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


"""
File paths and output directory
"""
prefix = '/Users/phr/Desktop/Spectral-Parser/spectral_parser'
# prefix = '/afs/inf.ed.ac.uk/user/s17/s1757135/Spectral-Parser/spectral_parser'
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

def save():
    print('Saving parameters.')
    torch.save(nonterminal_map, output_dir+'nonterminal_map.pt')
    torch.save(terminal_map, output_dir + 'terminal_map.pt')
    torch.save(pcfg, output_dir + 'pcfg.pt')
    torch.save(lpcfg, output_dir + 'lpcfg.pt')
    torch.save(rule3s_lookupC, output_dir + 'rule3s_lookupC.pt')
    torch.save(rule1s_lookup, output_dir + 'rule1s_lookup.pt')
    print('Done!')

def load():
    global nonterminal_map, terminal_map
    global pcfg, lpcfg
    global rule3s_lookupC, rule1s_lookup
    nonterminal_map = torch.load(output_dir+'nonterminal_map.pt')
    terminal_map = torch.load(output_dir+'terminal_map.pt')
    pcfg = torch.load(output_dir+'pcfg.pt')
    lpcfg = torch.load(output_dir + 'lpcfg.pt')
    rule3s_lookupC = torch.load(output_dir+'rule3s_lookupC.pt')
    rule1s_lookup = torch.load(output_dir+'rule1s_lookup.pt')
