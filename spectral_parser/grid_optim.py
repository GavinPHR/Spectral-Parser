import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg, optimize

config.prestates = 18
config.instates = 18
config.train = treebank_reader.read(config.train_file)
config.nonterminal_map = mappings.NonterminalMap(config.train)
config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
transforms.transform_trees(config.train)

config.pcfg = pcfg.PCFG()
import training.feature_extraction
import training.svd
config.lpcfg_optimize = optimize.LPCFG_Optimize()
import training.lookup

f = open('grid.logs', 'w')

prestates = list(range(10, 19))
instates = list(range(10, 19))
for p in prestates:
    for i in instates:
        score = config.lpcfg_optimize.opt(0, i, p, config.C)
        print('P=%d I=%d S=%.2f' % (p, i, score))
        f.write('P=%d I=%d S=%.2f\n' % (p, i, score))


f.close()
