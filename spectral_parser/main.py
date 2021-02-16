import config
from preprocessing import mappings, transforms, treebank_reader, embeddings
from training import pcfg

config.train = treebank_reader.read(config.train_file)
embeddings.retrieve()
config.nonterminal_map = mappings.NonterminalMap(config.train)
config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
transforms.transform_trees(config.train)

config.pcfg = pcfg.PCFG()
import training.feature_extraction
import training.svd
import training.lookup

config.save()
