"""
Loading all the saved parameters and cast
them into Numba compatible types.
"""
import config
from numba.core import types
from numba.typed import Dict
import numpy as np
import numba

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


numba.config.THREADING_LAYER = 'safe'

def to_typed_dict_rule_tensor(untyped_d, dimension, pi=False):
    if dimension == 1:
        t = types.float64[:]
    elif dimension == 2:
        t = types.float64[:, :]
    elif dimension == 3:
        t = types.float64[:, :, :]
    typed_d = Dict.empty(key_type=types.int64, value_type=t)
    if pi:
        for nonterm, tensor, in untyped_d.items():
            typed_d[nonterm] = tensor.astype(np.float64)
    else:
        for rule, tensor in untyped_d.items():
            assert (hash(rule) not in typed_d)
            typed_d[hash(rule)] = tensor.astype(np.float64)
    return typed_d

def to_typed_dict_nonterm_rules(untyped_d):
    typed_d = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    for nonterm, rules in untyped_d.items():
        np_rules = np.array([hash(rule) for rule in rules], dtype=np.int64)
        typed_d[nonterm] = np_rules
    return typed_d

def to_typed_dict_rule_float(untyped_d, pi=False):
    typed_d = Dict.empty(key_type=types.int64, value_type=types.float64)
    if pi:
        for nonterm, prob in untyped_d.items():
            typed_d[nonterm] = prob
    else:
        for rule, prob in untyped_d.items():
            assert(hash(rule) not in typed_d)
            typed_d[hash(rule)] = prob
    return typed_d

# Boolean as dummy values
# config.pos_tags = Dict.empty(key_type=types.int64, value_type=types.boolean)
# for rule in config.rule1s_full:
#     config.pos_tags[rule.a] = False

config.rule3s_full = to_typed_dict_rule_tensor(config.lpcfg.rule3s, 3)
config.rule1s_full = to_typed_dict_rule_tensor(config.lpcfg.rule1s, 1)
config.pi_full = to_typed_dict_rule_tensor(config.lpcfg.pi, 1, pi=True)

config.rule3s_prune = to_typed_dict_rule_float(config.pcfg.rule3s)
config.rule1s_prune = to_typed_dict_rule_float(config.pcfg.rule1s)
config.pi_prune = to_typed_dict_rule_float(config.pcfg.pi, pi=True)

config.rule3s_lookupC = to_typed_dict_nonterm_rules(config.rule3s_lookupC)
config.rule1s_lookup = to_typed_dict_nonterm_rules(config.rule1s_lookup)

config.numba_ready = True
