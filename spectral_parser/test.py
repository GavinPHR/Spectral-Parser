"""
Main entry point for parsing dev/test files
Remember to set the number of CPUs to use in config.py
"""
import multiprocessing as mp

import config
from parsing.parser import parse_devset

__author__ = 'Haoran Peng'
__email__ = 'gavinsweden@gmail.com'
__license__ = 'MIT'


mp.set_start_method('fork')
parse_devset(config.test_file)
