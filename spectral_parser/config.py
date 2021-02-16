prefix = '/Users/phr/Desktop/Spectral-Parser/spectral_parser'
# prefix = '/afs/inf.ed.ac.uk/user/s17/s1757135/Spectral-Parser/spectral_parser'
train_file = prefix+'/data/train.txt'
dev_file = prefix+'/data/dev.txt'
test_file = prefix+'/data/test.txt'
output_dir = prefix+'/output/'
cache = prefix+'/output/cache/'
tagger = prefix + '/tagger/tagger.pickle'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Output directory created.')

if not os.path.exists(cache):
    os.makedirs(cache)
    print('Cache directory created.')
import torch
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
max_state = 32
min_singular_value = 1
embedding_map = None
terminal_cutoff = 1
train = None
nonterminal_map = None
terminal_map = None
pcfg = None
I, O = None, None
rule3s_lookupC = None
rule1s_lookup = None

prune_cutoff = 1e-5
numba_ready = False
rule3s = None
rule1s = None
pi = None

def save():
    print('Saving parameters.')
    torch.save(nonterminal_map, output_dir+'nonterminal_map.pt')
    torch.save(terminal_map, output_dir + 'terminal_map.pt')
    torch.save(pcfg, output_dir + 'pcfg.pt')
    torch.save(I, output_dir + 'I.pt')
    torch.save(O, output_dir + 'O.pt')
    torch.save(rule3s_lookupC, output_dir + 'rule3s_lookupC.pt')
    torch.save(rule1s_lookup, output_dir + 'rule1s_lookup.pt')
    print('Done!')

def load():
    global nonterminal_map, terminal_map
    global pcfg, I, O
    global rule3s_lookupC, rule1s_lookup
    nonterminal_map = torch.load(output_dir+'nonterminal_map.pt')
    terminal_map = torch.load(output_dir+'terminal_map.pt')
    pcfg = torch.load(output_dir+'pcfg.pt')
    I, O = torch.load(output_dir+'I.pt'), torch.load(output_dir+'O.pt')
    rule3s_lookupC = torch.load(output_dir+'rule3s_lookupC.pt')
    rule1s_lookup = torch.load(output_dir+'rule1s_lookup.pt')
