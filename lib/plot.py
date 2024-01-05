# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 03:47:52 2024

@author: rolly
"""
import numpy as np


def rotflip(input):
    return np.rot90(np.fliplr(input))