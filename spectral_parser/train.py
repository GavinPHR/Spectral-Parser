"""
Main entry point to train a L-PCFG
You should have set all the file paths and hyperparameters in config.py
The pipeline below follows Chapter 3 of my dissertation.
"""
import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg, lpcfg_smoothed

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


# Preprocessing
config.train = treebank_reader.read(config.train_file)
config.nonterminal_map = mappings.NonterminalMap(config.train)
config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
transforms.transform_trees(config.train)

# PCFG
config.pcfg = pcfg.PCFG()

# L-PCFG
import training.feature_extraction
import training.svd
config.lpcfg = lpcfg_smoothed.LPCFG_Smoothed()

# Saving parameters
import training.lookup
config.save()
