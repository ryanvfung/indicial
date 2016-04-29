# -*- coding: utf-8 -*-
"""
Indicial response - typical forcing functions

No functions intended for independent use
"""


import numpy as np

from indicial import time_step_generator


def unit_impulse(t, step, initial_time):
    """unit_impulse(t, step, initial_time)

    Returns the unit impulse function for a given initial start time

    Parameters
    ----------
    t : int
        total time
    step : int
        steps per unit time
    initial_time : int or float
        initial time at which impulse occurs

    """
    array = np.zeros(t*step+1)
    array[time*step] = 1
    return array


def step_change(t, step, initial_time):
    """step_change(t, step, initial_time)

    Returns the unit step change function for a given initial start time

    Parameters
    ----------
    t : int
        total time
    step : int
        steps per unit time
    initial_time : int or float
        initial time at which step change occurs

    """
    array = np.zeros(t*step+1)
    for i in range(initial_time*step, t*step+1):
        array[i] = 1
    return array


def sine(t, step, amplitude, initial_time, wavelength, cycles):
    """sine(t, step, amplitude, initial_time, wavelength, cycles)

    Returns one or multiple sine waves starting from an initial time

    W_g(s) = (w_g/U) sin( (π/τ_g) s)

    Parameters
    ----------
    intial_time : int
        initial time at which sine waves begins
    wavelength : int
        gust gradient?
    cycles : int

    """
    x = time_step_generator(t, step)
    w = np.zeros(t*step+1)
    for i in range(t*step):
        if x[i] >= initial_time and x[i] <= initial_time + wavelength * cycles:
            w[i] = amplitude * np.sin(
                (2*np.pi/wavelength)*(x[i]-initial_time)
            )
    return w


def cosine(t, step, amplitude, initial_time, wavelength, cycles):
    """cosine(t, step, amplitude, initial_time, wavelength, cycles)

    Returns one or multiple cosine waves starting from an initial time

    W_g(s) = (w_g/U) sin( (π/τ_g) s)

    Parameters
    ----------
    intial_time : int
        initial time at which sine waves begins
    wavelength : int
        gust gradient?
    cycles : int

    """
    x = time_step_generator(t, step)
    w = np.zeros(t*step+1)
    for i in range(t*step):
        if x[i] >= initial_time and x[i] <= initial_time + wavelength * cycles:
            w[i] = amplitude * np.cos(
                (2*np.pi/wavelength)*(x[i]-initial_time)
            )
    return w


def one_minus_cosine(t, step, amplitude, initial_time, wavelength, cycles):
    """one_minus_cosine(t, step, amplitude, initial_time, wavelength, cycles)

    Returns one or more one-minus-cosine waves starting from an initial time

    W_g(s) = (w_g/2U) (1 - cos( (π/τ_g) s) )

    Parameters
    ----------
    intial_time : int
        initial time at which sine waves begins
    wavelength : int
        gust gradient?
    cycles : int

    """
    x = time_step_generator(t, step)
    w = np.zeros(t*step+1)
    for i in range(t*step):
        if x[i] >= initial_time and x[i] <= initial_time + wavelength * cycles:
            w[i] = amplitude * (1 - np.cos(
                (2*np.pi/wavelength)*(x[i]-initial_time)
            ))
    return w
