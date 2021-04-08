import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg, lpcfg, lpcfg_smoothed

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


config.train = treebank_reader.read(config.train_file)
config.nonterminal_map = mappings.NonterminalMap(config.train)
config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
transforms.transform_trees(config.train)

config.pcfg = pcfg.PCFG()
import training.feature_extraction
import training.svd
# config.lpcfg = lpcfg.LPCFG()
config.lpcfg = lpcfg_smoothed.LPCFG_Smoothed()
import training.lookup

config.save()
