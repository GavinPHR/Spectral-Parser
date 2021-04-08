"""
Experimental file for Bayesian optimization
"""
import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg, optimize

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


config.max_state = 26
config.cutoff = 0
config.train = treebank_reader.read(config.train_file)
config.nonterminal_map = mappings.NonterminalMap(config.train)
config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
transforms.transform_trees(config.train)

config.pcfg = pcfg.PCFG()
import training.feature_extraction
import training.svd
config.lpcfg_optimize = optimize.LPCFG_Optimize()
import training.lookup


from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Parameters bounds
pbounds = {'cutoff': (0, 0.2),
           'instates': (10, 26),
           'prestates': (10, 26),
           'C': (0, 50)
           }

optimizer = BayesianOptimization(
    f=config.lpcfg_optimize.opt,
    pbounds=pbounds,
    verbose=2, 
    random_state=42,
)

# This is dangerous as it deletes the original file if you have one
# waiting for the next update that fixes this
logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# from bayes_opt.util import load_logs
# load_logs(optimizer, logs=["./logs.json"])

optimizer.probe(
    params={'cutoff': 0.01,
           'instates': 16,
           'prestates': 16,
           'C': 10}
)

optimizer.maximize(
    init_points=5,
    n_iter=500,
)

print(optimizer.max)
