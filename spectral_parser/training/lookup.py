"""
Reverse lookup for parsing.
"""
from collections import defaultdict

import config

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


pcfg = config.pcfg

rule3s_lookupC = defaultdict(list)
for rule in pcfg.rule3s:
    rule3s_lookupC[rule.c].append(rule)

rule1s_lookup = defaultdict(list)
for rule in pcfg.rule1s:
    rule1s_lookup[rule.x].append(rule)

config.rule3s_lookupC = rule3s_lookupC
config.rule1s_lookup = rule1s_lookup
