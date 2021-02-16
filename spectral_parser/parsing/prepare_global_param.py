import config
from numba.core import types
from numba.typed import Dict
import numpy as np
import numba

numba.config.THREADING_LAYER = 'safe'

def to_typed_dict_nt_array(untyped_d):
    typed_d = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
    for nt, array in untyped_d.items():
        typed_d[nt] = array.astype(np.float64)
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


config.rule3s = to_typed_dict_rule_float(config.pcfg.rule3s)
config.rule1s = to_typed_dict_rule_float(config.pcfg.rule1s)
config.pi = to_typed_dict_rule_float(config.pcfg.pi)

config.I = to_typed_dict_nt_array(config.I)
config.O = to_typed_dict_nt_array(config.O)

config.rule3s_lookupC = to_typed_dict_nonterm_rules(config.rule3s_lookupC)
config.rule1s_lookup = to_typed_dict_nonterm_rules(config.rule1s_lookup)

config.numba_ready = True
