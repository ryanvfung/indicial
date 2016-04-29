# -*- coding: utf-8 -*-
"""
Indicial response

Run the following command in Spyder's IPython console
    runfile('folder/indicial.py', wdir='folder')
where the working directory replaces the placeholder keyword folder,
or the following command in the Python console:
    exec(open("folder/indicial.py").read())

Manually call each function as desired, usually 'indicial1'
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from time import process_time, strftime

from shapes import *
from responses import *
from modelling import extract_2d_data


def indicial2(t, f, H, filename=False):
    """indicial2(t, f, H, filename=False)

    Evaluates the system response y(t) using indicial theory, based on the
    forcing function f(t) and the indicial response to a unit impulse
    function H(t).

    Parameters
    ----------
    t : np.array of floats
        time steps
    f : np.array of floats
        forcing function
    H : np.array of floats
        indicial response to a unit impulse function

    Returns
    -------
    np.array of floats
        response to forcing function based on indicial theory

    """
    sys.stdout.write('indicial2 execution started\n')
    sys.stdout.flush()

    execution_start = process_time()

    # indicial start
    y = np.zeros(len(t))
    for i in range(len(t)):
        y[i] = duhamel2(i, t[0], f, H)
    # indicial end

    sys.stdout.write('indicial2 execution time: {0}\n'.format(
        process_time() - execution_start
    ))
    sys.stdout.flush()

    postprocess(t, f, H, y, filename)
    return y


def duhamel2(i, h, f, H):
    """duhamel2(i, h, f, H)

    Evaluates Duhamel's integral using Simpson's rule:

           t
    y(t) = ∫ f(τ) H(t-τ) dτ
           0

    Parameters
    ----------
    i : int
        integral upper limit as index of t
    h : float
        length of one time step
    f : np.array of floats
        forcing function
        array dimensions should be [1, t]
    H : np.array of floats
        indicial response function to a unit impulse function
        array dimensions should be [1, t]

    Returns
    -------
    np.array of floats
        response to forcing function at time t based on indicial theory

    """
    sum_all = 0.
    for j in range(i):
        sum_all += f[j]*H[i-j]
    sum_even = 0.
    for k in range(0, i, 2):
        sum_even += f[k]*H[i-(k)]
    final_terms = (f[0]*H[i] + f[i]*H[0])
    return h/3.*(4*sum_all - 2*sum_even - 3*final_terms)


def indicial1(t, f, A, df=0, dA=0, use_df=True, filename=False,
              post_process=False):
    """indicial1(t, f, A, df=0, dA=0, use_df=True, filename=False)

    Evaluates the system response y(t) using indicial theory, based on the
    forcing function f(t) and the indicial response to a unit step change
    function A(t).

    Parameters
    ----------
    t : np.array of floats
        time steps
    f : np.array of floats
        forcing function
    A : np.array of floats
        indicial response to a unit step change function

    Returns
    -------
    np.array of floats
        response to forcing function based on indicial theory

    """

    sys.stdout.write('indicial1 execution started\n')
    sys.stdout.flush()
    execution_start = process_time()

    # indicial start
    if len(A) < len(t):
        A = np.append(A, np.full(len(t)-len(A), A[-1]))
    if df is 0:
        df = derivative(f, t[0])
    y = np.zeros(len(t))
    for i in range(len(t)):
        y[i] = duhamel1(i, t[0], f, A, df, dA, use_df)
    # indicial end

    sys.stdout.write('indicial1 execution time: {0}\n'.format(
        process_time() - execution_start
    ))
    sys.stdout.flush()

    if post_process is True:
        postprocess(t, f, A, y, filename)

    return y


def duhamel1(i, h, f, A, df=0, dA=0, use_df=True):
    """duhamel1(i, h, f, A, df=0, dA=0, use_df=True)

    Evaluates Duhamel's integral using Simpson's rule:

                       t
    y(t) = f(0) A(t) + ∫ f'(τ) A(t-τ) dτ
                       0

    Parameters
    ----------
    i : int
        integral upper limit as index of t
    h : float
        length of one time step
    f : np.array of floats
        forcing function
        array dimensions should be [1, t]
    A : np.array of floats
        indicial response function to a unit step change function
        array dimensions should be [1, t]
    df : np.array of floats
        precomputed array of derivatives of f
    dA : np.array of floats
        precomputed array of derivatives of A
    use_df : bool
        True : use f for derivative term
        False: use A for derivative term

    Returns
    -------
    np.array of floats
        response to forcing function at time t based on indicial theory

    """
    sum_all = 0.
    sum_even = 0.
    for j in range(i):
        sum_all += df[j]*A[i-j]
    for k in range(0, i, 2):
        sum_even += df[k]*A[i-(k)]
    final_terms = (df[0]*A[i] + df[i]*A[0])
    return f[0]*A[i] + h/3.*(4*sum_all - 2*sum_even - 3*final_terms)


def derivative(f, h):
    """derivative(f, h)

    Evaluates the derivative using the 2nd order central difference formula:

            f(x+h) - f(x-h)
    f'(x) = ---------------
                  2h

    where h = 1/τ

    Parameters
    ----------
    f : np.array of floats
        function for numerical differentiation
    h : float
        length of one time step

    Returns
    -------
    np.array of floats

    """
    l = len(f)
    df = np.zeros(l)
    df[0] = (f[1]-f[0])/h
    df[l-1] = (f[l-1]-f[l-2])/h
    for i in range(1, l-1):
        df[i] = (f[i+1]-f[i-1])/(2*h)
    return df


def time_step_generator(time, step):
    """time_step_generator(time, step)

    Generates array of time step values.

    Parameters
    ----------
    time : int
        total time duration
    step : int
        steps per unit time

    Returns
    -------
    np.array of floats

    """

    return np.array(range(int(time*step+1)))[1:]/step


def postprocess(t, f, A, y, filename=False):
    """postprocess(t, f, A, y, filename=False)

    Plots the numerical data and saves it to a PNG and SVG file, saves the
    indicial function numerical data into a csv file.

    Parameters
    ----------
    t : np.array of floats
        time steps
    f : np.array of floats
        forcing function
    A : np.array of floats
        indicial response to a unit step change function
    y : np.array of floats
        indicial response to forcing function
    filename : bool
        True to customise output file name
        False will default to current datetime as file name

    """
    if filename is False:
        filename = strftime('%Y%m%dT%H%M%S')
    plt.figure(filename + '1')
    plot1, = plt.plot(t, y, 'b-', label='Response')
    plot2, = plt.plot(t, f, 'g--', label='Forcing function')
    plt.xlabel(r'$\tau$', fontsize='xx-large')
    plt.ylabel(r'$C_L, \alpha$', fontsize='xx-large')
    plt.legend(
        handles=[plot1, plot2],
        loc=4,  # bottom right
        fontsize='large',
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(
        'output/' + filename + '.png',
        format='png',
        dpi=150,
        bbox_inches='tight',
    )
    fig.savefig(
        'output/' + filename + '.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight',
    )

    plt.figure(filename + '2')
    plot1, = plt.plot(t, y, 'b-', label='Response')
    plt.xlabel(r'$\tau$', fontsize='xx-large')
    plt.ylabel(r'$C_L$', fontsize='xx-large')
    plt.legend(
        handles=[plot1],
        loc=4,  # bottom right
        fontsize='large',
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(
        'output/' + filename + '-output.png',
        format='png',
        dpi=96,
        bbox_inches='tight',
    )
    fig.savefig(
        'output/' + filename + '-output.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight',
    )

    plt.figure(filename + '3')
    plot1, = plt.plot(t, f, 'g--', label='Forcing function')
    plt.xlabel(r'$\tau$', fontsize='xx-large')
    plt.ylabel(r'$\alpha$', fontsize='xx-large')
    plt.legend(
        handles=[plot1],
        loc=4,  # bottom right
        fontsize='large',
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(
        'output/' + filename + '-input.png',
        format='png',
        dpi=96,
        bbox_inches='tight',
    )
    fig.savefig(
        'output/' + filename + '-input.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight',
    )

    csv = open('output/' + filename + '.csv', 'w')
    csv.write('Index,Time,Output,Forcing function,System response\n')
    for i in range(len(t)):
        csv.write('{i},{t},{y},{f},{A}\n'.format(
            i=i, t=t[i], y=y[i], f=f[i], A=A[i],
        ))
    csv.close()
    return
