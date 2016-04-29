# -*- coding: utf-8 -*-
"""
Indicial response - response functions

No functions intended for independent use
"""


import numpy as np

from functions import exp1, exp2


def wagner(t):
    """wagner(t)

    Returns the Wagner function approximation:

    φ(s) = 1 - A_1 exp(-b_1 s) - A_2 exp(-b_2 s)

    where:
        A_1 = 0.165
        A_2 = 0.335
        b_1 = 0.0455
        b_2 = 0.3

    Parameters
    ----------
    s : int or float or np.array

    Returns
    -------
    np.array of floats

    """
    s = t*2  # converting t=Ut/c to s=Ut/b=Ut/(c/2)
    phi1 = 0.165
    phi2 = 0.335
    epsilon1 = 0.0455
    epsilon2 = 0.3
    return 1 - phi1*np.exp(-epsilon1*s) - phi2*np.exp(-epsilon2*s)


def wagner5(t):
    """wagner(t)

    Returns the Wagner function approximation at Mach number 0.5:

    φ(s) = A_0 - A_1 exp(-b_1 s) - A_2 exp(-b_2 s) - A_3 exp(-b_3 s)

    where:
        A_0 = 1.0
        A_1 = 0.406
        A_2 = 0.249
        A_3 = -0.773
        b_1 = 0.0753
        b_2 = 0.372
        b_3 = 1.89

    Parameters
    ----------
    s : int or float or np.array

    Returns
    -------
    np.array of floats

    """
    s = t*2  # converting t=Ut/c to s=Ut/b=Ut/(c/2)
    A_0 = 1.0
    A_1 = 0.406
    A_2 = 0.249
    A_3 = -0.773
    b_1 = 0.0753
    b_2 = 0.372
    b_3 = 1.89
    return A_0 - A_1*np.exp(-b_1*s) - A_2*np.exp(-b_2*s) - A_3*np.exp(-b_3*s)


def wagner7(t):
    """wagner(t)

    Returns the Wagner function approximation at Mach number 0.7:

    φ(s) = A_0 - A_1 exp(-b_1 s) - A_2 exp(-b_2 s) - A_3 exp(-b_3 s)

    where:
        A_0 = 1.4
        A_1 = 0.5096
        A_2 = 0.567
        A_3 = -0.5866
        b_1 = 0.0536
        b_2 = 0.357
        b_3 = 0.902

    Parameters
    ----------
    s : int or float or np.array

    Returns
    -------
    np.array of floats

    """
    s = t*2  # converting t=Ut/c to s=Ut/b=Ut/(c/2)
    A_0 = 1.4
    A_1 = 0.5096
    A_2 = 0.567
    A_3 = -0.5866
    b_1 = 0.0536
    b_2 = 0.357
    b_3 = 0.902
    return A_0 - A_1*np.exp(-b_1*s) - A_2*np.exp(-b_2*s) - A_3*np.exp(-b_3*s)


def wagner_gen(array):
    """wagner_gen(array)

    Returns an array of Wagner function approximation values for each input.

    Parameters
    ----------
    array : np.array of floats

    """
    output = np.zeros(len(array))
    for i in range(len(array)):
        output[i] = wagner(array[i])
    return output


def kussner(s):
    """kussner(s)

    Returns the Küssner function approximation by von Kárman and Sears:

    Ψ(s) = 1 - A_3 exp(-b_3 s) - A_4 exp(-b_4 s)

    where:
        A_3 = 0.5
        A_4 = 0.5
        b_3 = 0.130
        b_4 = 1

    Parameters
    ----------
    s : int

    """
    A3 = 0.5
    A4 = 0.5
    b3 = 0.130
    b4 = 1
    return 1 - A3*np.exp(-1*b3*s) - A4*np.exp(-1*b4*s)


def kussner_gen(array):
    """kussner_gen(array)

    Returns an array of Küssner function approximation values for each input.

    Parameters
    ----------
    array : np.array of floats

    """
    output = np.zeros(len(array))
    for i in range(len(array)):
        output[i] = kussner(array[i])
    return output


def onerad_ar20s0(t, M):
    """onerad_ar20s0(t, M)

    Returns the indicial response for:
        Aspect ratio: 20
        Sweep angle: 0 degrees

    Parameters
    ----------
    t : np.array of floats
        time values
    M : float
        Mach number

    """
    a0 = 8.5584*(M**2.) + 3.1637 * M + 5.5489
    a1 = 3.144849189*M + 1.429234632
    b1 = -0.389208609*M + 0.377970221
    return exp1(t, a0, a1, b1)


def onerad_ar8s0(t, M):
    """onerad_ar8s0(t, M)

    Returns the indicial response for:
        Aspect ratio: 8
        Sweep angle: 0 degrees

    Parameters
    ----------
    t : np.array of floats
        time values
    M : float
        Mach number

    """
    a0 = 5.1389*(M**2.) + 1.742 * M + 4.7201
    a1 = 0.01497710890485*M + 2.35205673511049
    b1 = -0.484716349547985*M + 0.493309211841525

    return exp1(t, a0, a1, b1)


def onerad_ar20s30(t, M):
    """onerad_ar20s30(t, M)

    Returns the indicial response for:
        Aspect ratio: 20
        Sweep angle: 30 degrees

    Parameters
    ----------
    t : np.array of floats
        time values
    M : float
        Mach number

    """
    a0 = 4.668*(M**2.) + 1.8844 * M + 5.0452
    a1 = 0.86023552205925*M + 2.14628032199648
    b1 = -0.24956366386698*M + 0.29478183193349

    return exp1(t, a0, a1, b1)


def onerad_ar8s30(t, M):
    """onerad_ar8s30(t, M)

    Returns the indicial response for:
        Aspect ratio: 8
        Sweep angle: 30 degrees

    Parameters
    ----------
    t : np.array of floats
        time values
    M : float
        Mach number

    """
    a0 = 2.7313*(M**2.) + 0.4291 * M + 4.0829
    a1 = -0.98552787925575*M + 2.79276393962787
    b1 = -0.5262607681456*M + 0.5431303840728

    return exp1(t, a0, a1, b1)


def onerad(t, M, AR, S):
    """onerad(M, AR, S):

    Returns the modelled indicial response for arbitrary aspect ratio and sweep
    angle of an OneraD wing.

    Parameters
    ----------
    t : np.array of floats
        time values
    M : float
        Mach number
    AR : float
        Aspect ratio
    S : float
        Sweep angle

    """
    c = {
        # Parameters for determining coefficients [a, b, c, d]
        # for (a*S + b)*AR + (c*S + d)
        'a0_M2': [
            -0.00411888888888889,
            0.284958333333333,
            -0.0473022222222222,
            2.85923333333333,
        ],
        'a0_M1': [
            -0.0000933333333333337,
            -0.118475,
            0.04451,
            -0.7942,
        ],
        'a0_M0': [
            0.000370833333333335,
            0.0690666666666666,
            -0.0242066666666667,
            4.16756666666667,
        ],
        'a1_M1': [
            -0.00356696855350598,
            0.260822673381429,
            -0.0048144178439722,
            -2.07160427814658,
        ],
        'a1_M0': [
            0.000767606904633126,
            -0.0769018419416104,
            0.00854938491351433,
            2.96727147064338,
        ],
        'b1_M1': [
            0.000503303787724389,
            0.00795897839148668,
            -0.00541124425504894,
            -0.548388176679879,
        ],
        'b1_M0': [
            -0.0003694710022937,
            -0.00961158260946484,
            0.00461647375939212,
            0.570201872717243,
        ],
        'a2_M2': [
            -0.00461111111111112,
            0.105916666666667,
            0.050488888888889,
            -46.0673333333333,
        ],
        'a2_M1': [
            0.00439722222222222,
            -0.169666666666667,
            -0.0492777777777775,
            73.0783333333333,
        ],
        'a2_M0': [
            -0.000802777777777782,
            0.0855,
            -0.013111111111111,
            -30.739,
        ],
        'b2_M2': [
            -0.0107305555555556,
            0.5345,
            -0.0864555555555556,
            20.367,
        ],
        'b2_M1': [
            0.00846944444444444,
            -0.50925,
            0.0439444444444445,
            -23.798,
        ],
        'b2_M0': [
            -0.00119194444444444,
            0.1228,
            0.00573888888888892,
            8.635,
        ],
    }

    def lin(key):
        p = c[key]
        return (p[0]*S+p[1])*AR+(p[2]*S+p[3])

    a0 = lin('a0_M2')*(M**2) + lin('a0_M1')*M + lin('a0_M0')
    a1 = lin('a1_M1')*M + lin('a1_M0')
    b1 = lin('b1_M1')*M + lin('b1_M0')
    a2 = lin('a2_M2')*(M**2) + lin('a2_M1')*M + lin('a2_M0')
    b2 = lin('b2_M2')*(M**2) + lin('b2_M1')*M + lin('b2_M0')

    return exp2(t, a0, a1, b1, a2, b2)
