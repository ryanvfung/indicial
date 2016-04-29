# -*- coding: utf-8 -*-
"""
Indicial response - calls made after running modelling.py
"""


from modelling import *


# Initialise variables
d = {}


# Run automated curve fitting to the following files
fit_multiple([
    'b10s0M0_3',
    'b10s0M0_5',
    'b10s0M0_7',
    'b4s0M0_3',
    'b4s0M0_5',
    'b4s0M0_7',
    'b10s30M0_3',
    'b10s30M0_5',
    'b10s30M0_7',
    'b4s30M0_3',
    'b4s30M0_5',
    'b4s30M0_7'
], d)


# Run manual curve fitting for the following configurations
test_functions(d, 20, 0, 0.3)
test_functions(d, 8, 0, 0.3)
test_functions(d, 20, 30, 0.5)
test_functions(d, 20, 30, 0.3)
test_functions(d, 8, 30, 0.5)
test_functions(d, 8, 30, 0.3)
