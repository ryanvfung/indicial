# -*- coding: utf-8 -*-
"""
Indicial response - typical response functions for modelling

No functions intended for independent use
"""


import numpy as np


def exp1(t, a0, a1, b1):
    """exp1(t, a0, a1, b1)

    Evaluates the 1 term exponential:
        y(t) = a_0 - a_1 exp(-b_1 t)

    Parameters
    ----------
    a0 : float
        asymptotic value
    a1 : float
        linear coefficient
    b1 : float
        exponent

    """
    return a0 - a1 * np.exp(-b1 * t)


def exp1nc(t, a2, b2):
    """exp1nc(t, a2, b2)

    Evaluates the 1 term exponential:
        y(t) = a_2 exp(-b_2 t)
    (same as exp1 but without a constant - exp1 no constant)

    Parameters
    ----------
    a0 : float
        asymptotic value
    a2 : float
        linear coefficient
    b2 : float
        exponent

    """
    return -a2 * np.exp(-b2 * t)


def exp1x(t, a2, b2, c2):
    """exp1x(t, a2, b2, c2)

    Parameters
    ----------
    a0 : float
        asymptotic value
    a2 : float
        linear coefficient
    b2 : float
        exponent linear coefficient
    c2 : float
        exponent linear constant

    """
    return -a2 * np.exp(-b2 * (t-c2))


def exp2(t, a0, a1, b1, a2, b2):
    """exp2(t, a0, a1, b1, a2, b2)

    Evaluates the 2 term exponential:
        y(t) = a_0 - a_1 exp(-b_1 t) - a_2 exp(-b_2 t)

    Parameters
    ----------
    a0 : float
        asymptotic value
    a1 : float
        first term coefficient
    b1 : float
        first term exponent
    a2 : float
        second term coefficient
    b2 : float
        second term exponent

    """
    return a0 - a1 * np.exp(-b1 * t) - a2 * np.exp(-b2 * t)


def exp1cos1(t, a0, a1, b1, a2, b2):
    """exp1cos1(t, a0, a1, b1, a2, b2)

    Evaluates the exponential-cosine function:
        y(t) = a_0 - a_1 exp(-b_1 t) - a_2 cos(-b_2 t)

    Parameters
    ----------
    a0 : float
        asymptotic value
    a1 : float
        exponential term coefficient
    b1 : float
        exponential term exponent
    a2 : float
        cosine term coefficient
    b2 : float
        cosine term exponent

    """
    return a0 - a1 * np.exp(-b1 * t) - a2 * np.cos(-b2 * t)


def rational1(t, a0, a1, a2):
    """rational1(t, a0, a1, a2)

    Evaluates the following rational function:
                       a_1
        y(t) = a_0 - -------
                     t + a_2

    Parameters
    ----------
    a0 : float
        asymptotic value
    a1 : float
        numerator
    a2 : float
        denominator constant term coefficient

    """
    return a0 - a1/(t+a2)
