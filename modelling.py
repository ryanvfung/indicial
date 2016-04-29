# -*- coding: utf-8 -*-
"""
Indicial response - modelling response functions

Run the following command in Spyder's IPython console
    runfile('folder/modelling.py', wdir='folder')
where the working directory replaces the placeholder keyword folder,
or the following command in the Python console:
    exec(open("folder/modelling.py").read())

Manually call each function as desired, usually 'fit'
"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from functions import *


def extract_2d_data(filename):
    """extract_2d_data(filename)

    Extracts 2D time-response value pairs in the format
        time, value
    from given file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    t : np.array of floats
        time values
    y : np.array of floats
        response values

    """
    datafile = open(filename)
    lines = datafile.readlines()
    datafile.close()
    length = len(lines)
    t = np.zeros(length)
    y = np.zeros(length)
    line_indices = range(length)
    try:
        # if first row is header-like, ignore it
        float(lines[0].split(',')[0])
    except ValueError:
        line_indices = range(1, length)
    for i in line_indices:
        values = lines[i].split(',')
        t[i] = float(values[0])
        y[i] = float(values[1])
    return t, y


def fit(filename, normalise=False):
    """fit(filename, normalise=False)

    Extracts 2D time-response data from a file, remove transient response data,
    curve fits it to exp1 and rational1.
    Saves the optimised parameters, covariance and standard deviation to a
    text file in the output folder.
    Plots the original data against the fitted curve and saves the graph plot
    data and image to the output folder.
    Returns generated data and curve fit data.

    Parameters
    ----------
    filename : str
        file name without file extension
    normalise : bool
        True to normalise values against asymptotic value
        False to leave data unchanged

    Returns
    -------
    data : dict
        dictionary containing the following:

        t : np.array of floats
            time values
        y : np.array of floats
            response values
        t2 : np.array of floats
            time values starting at response minimum
        y2 : np.array of floats
            response values starting at response minimum
        (p1, p3) : tuple of np.array
            optimal values of exp1 and rational 1 parameters
            (popt from scipy.optimise.curve_fit)
        (p1_cov, p3_cov) : tuple of np.array
            covariance of parameters
            (pcov from scipy.optimise.curve_fit)
        (p1_err, p3_err) : tuple of np.array
            standard deviation of parameters

    """
    t, y = extract_2d_data(filename + '.csv')

    # Split instantaneous and asymptotic parts of the response by truncating
    # the starting values until the response is at its minimum.
    argmin = y.argmin()
    t2, y2 = t[argmin:], y[argmin:]
    t3, y3 = t[:argmin], y[:argmin]

    if normalise is True:
        filename += '-norm'
        y2 = y2/y[-1]
        y = y/y[-1]

    plt.figure(filename)
    plot0a, = plt.plot(t2, y2, 'b-', label='CFD (steady-state)')
    plot0b, = plt.plot(t3, y3, 'c-', label='CFD (transient)')

    x = np.linspace(0, t[-1], 200)

    p1, p1_cov = curve_fit(exp1, t2, y2, p0=[max(y2), 3, 0.1])
    p1_err = np.sqrt(np.diag(p1_cov))
    fit1 = exp1(x, p1[0], p1[1], p1[2])
    plot1, = plt.plot(x, fit1, 'g--', label='Exponential type 1')

    fit1_t = exp1(t, p1[0], p1[1], p1[2])
    # error1, = plt.plot(t, y-fit1, 'g--', label='Error')
    p1a, p1a_cov = curve_fit(exp1nc, t, y-fit1_t)
    p1a_err = np.sqrt(np.diag(p1a_cov))
    fit1a = exp1nc(x, p1a[0], p1a[1])
    fit1a_t = exp1nc(t, p1a[0], p1a[1])
    plot1a, = plt.plot(t, fit1_t + fit1a_t, 'r--', label='Exponential type 2')
    # p1b, p1b_cov = curve_fit(rational1, t, y-fit1)
    # fit1b = rational1(t, p1b[0], p1b[1], p1b[2])
    # plot1b, = plt.plot(t, fit1+fit1b, 'y--', label='Error fit (rational)')

    error = [  # [at end, at minimum, standard deviation]
        1 - exp2(t2[-1], p1[0], p1[1], p1[2], p1a[0], p1a[1])/y2[-1],
        1 - exp2(t2[0], p1[0], p1[1], p1[2], p1a[0], p1a[1])/y2[0],
        np.std(np.array(
            [exp2(t, p1[0], p1[1], p1[2], p1a[0], p1a[1]), y]
        ))/y2[-1],
    ]
    values = [  # [impulsive value, minimum time, minimum value]
        y[0],
        t2[0],
        y2[0],
    ]

    p3, p3_cov = curve_fit(rational1, t, y, p0=[max(y2), 1, 0])
    p3_err = np.sqrt(np.diag(p3_cov))
    fit3 = rational1(x, p3[0], p3[1], p3[2])
    # plot3, = plt.plot(x, fit3, '--', label='Curve fit (rational1)')

    plt.xlabel(r'$t^*$', fontsize='xx-large')
    plt.ylabel(r'$C_L/\Delta\alpha\ \ [\mathrm{rad}^{-1}]$',
               fontsize='xx-large')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, 0, ymax])

    if y[-1] < 0.8*y[0]:
        loc = 1  # top right
    else:
        loc = 4  # bottom right
    plt.legend(
        handles=[
            plot0a,
            plot0b,
            plot1,
            plot1a,
            # plot1b,
            # plot3,
        ],
        loc=loc,  # bottom right
        fontsize='large',
    )

    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(
        'output/' + filename + '-fit.png',
        format='png',
        dpi=150,
        bbox_inches='tight',
    )
    fig.savefig(
        'output/' + filename + '-fit.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight',
    )

    csv = open('output/' + filename + '-fit.csv', 'w')
    csv.write('t,exp1,exp1+exp1nc\n')
    for i in range(len(x)):
        csv.write('{xi},{y1i},{y1ai}\n'.format(
            xi=x[i],
            y1i=fit1[i],
            y1ai=fit1a[i]+fit1[i],
        ))
    csv.close()
    results = open('output/' + filename + '-fit.txt', 'w')
    results.write('Curve fitting for {}\n'.format(filename))
    results.write('='*(18+len(filename)))
    results.write('\n\nFunctions\n---------\n')
    results.write('exp1:\n\t')
    results.write('y = {a0} - {a1} exp(-{b1} t)\n'.format(
        a0=p1[0], a1=p1[1], b1=p1[2],
    ))
    results.write('exp1 + exp1nc:\n\t')
    results.write('y = {a0} - {a1} exp(-{b1} t)\n'.format(
        a0=p1[0], a1=p1[1], b1=p1[2],
    ))
    results.write('\t    + {a1} exp(-{b1} t)\n'.format(
        a1=p1a[0], b1=p1a[1],
    ))
    results.write('rational1:\n\t')
    results.write('y = {a0} - {a1}/(t + {a2})\n'.format(
        a0=p3[0], a1=p3[1], a2=p3[2],
    ))
    results.write('\nStandard deviation\n------------------\n')
    results.write('exp1:\n\t{}\n'.format(p1_err))
    # file.write('exp1cos1:\n\t{}\n'.format(p2err))
    results.write('rational1:\n\t{}\n'.format(p3_err))
    results.close()
    data = {
        't': t,
        'y': y,
        't2': t2,
        'y2': y2,
        'p1': p1,
        'p3': p3,
        'p1a': p1a,
        'p1_cov': p1_cov,
        'p3_cov': p3_cov,
        'p1a_cov': p1a_cov,
        'p1_err': p1_err,
        'p3_err': p3_err,
        'p1a_err': p1a_err,
        'error': error,
        'values': values,
    }
    return data


def fit_multiple(filenames, data):
    """fit_multiple(filenames, data)

    Run 'fit' against multiple files consecutively.

    Parameters
    ----------
    filenames : list
    data : dict
        variable to store processed data to

    """
    output = []
    for filename in filenames:
        data[filename] = fit(filename)
    csv = open('output/fit_multiple.csv', 'w')
    csv.write('filename')
    csv.write(',exp1 a0,exp1 a1,exp1 b1')
    csv.write(',exp1nc a2,exp1nc b2')
    csv.write(',exp1 a0 stddev,exp1 a1 stddev,exp1 b1 stddev')
    csv.write(',exp1nc a2 stddev,exp1nc b2 stddev')
    csv.write(',error at end,error at min,stddev')
    csv.write(',impulsive val,time at min,value at min')
    csv.write('\n')
    for filename in filenames:
        csv.write(filename)
        csv.write(',{exp1_a0},{exp1_a1},{exp1_b1}'.format(
            exp1_a0=data[filename]['p1'][0],
            exp1_a1=data[filename]['p1'][1],
            exp1_b1=data[filename]['p1'][2],
        ))
        csv.write(',{exp1nc_a2},{exp1nc_b2}'.format(
            exp1nc_a2=data[filename]['p1a'][0],
            exp1nc_b2=data[filename]['p1a'][1],
        ))
        # Standard deviations
        csv.write(',{exp1_a0},{exp1_a1},{exp1_b1}'.format(
            exp1_a0=data[filename]['p1_err'][0],
            exp1_a1=data[filename]['p1_err'][1],
            exp1_b1=data[filename]['p1_err'][2],
        ))
        csv.write(',{exp1nc_a2},{exp1nc_b2}'.format(
            exp1nc_a2=data[filename]['p1a_err'][0],
            exp1nc_b2=data[filename]['p1a_err'][1],
        ))
        # Errors
        csv.write(',{err0},{err1},{err2}'.format(
            err0=data[filename]['error'][0],
            err1=data[filename]['error'][1],
            err2=data[filename]['error'][2],
        ))
        # Values
        csv.write(',{val0},{val1},{val2}'.format(
            val0=data[filename]['values'][0],
            val1=data[filename]['values'][1],
            val2=data[filename]['values'][2],
        ))
        csv.write('\n')
    csv.close()
    return data


def testfit(d, a0, a1, b1, a2=0, b2=0, filename='testfit'):
    """testfit(d, a0, a1, b1)

    Plots the exp1 function against provided response data, and saves the plot
    to output/testfit.png.

    Parameters
    ----------
    d : dict
        processed data source for one case (e.g.: from running fit_multiple)
    a0 : float
        asymptotic value
    a1 : float
        linear coefficient
    b1 : float
        exponent

    """
    plt.figure(filename)

    # Split instantaneous and asymptotic parts of the response by truncating
    # the starting values until the response is at its minimum.
    argmin = d['y'].argmin()
    t2, y2 = d['t'][argmin:], d['y'][argmin:]
    t3, y3 = d['t'][:argmin], d['y'][:argmin]

    plot0a, = plt.plot(t2, y2, 'b-', label='CFD (steady-state)')
    plot0b, = plt.plot(t3, y3, 'c-', label='CFD (transient)')
    x = np.linspace(0, d['t'][-1], 200)

    fit1 = exp1(d['t'], a0, a1, b1)
    # fit1 = rational1(d['t'], a0, a1, b1)
    plot1, = plt.plot(d['t'], fit1, 'g--', label='Exponential type 1')

    if a2 == 0 and b2 == 0:
        # error1, = plt.plot(t, y-fit1, 'g--', label='Error')
        p1a, p1a_cov = curve_fit(exp1nc, d['t'], d['y']-fit1)
        p1a_err = np.sqrt(np.diag(p1a_cov))
        fit1a = exp1nc(x, p1a[0], p1a[1])
        fit1a_t = exp1nc(d['t'], p1a[0], p1a[1])
        plot1a, = plt.plot(
            d['t'], fit1 + fit1a_t, 'r--',
            label='Exponential type 2',
        )

        error = [  # [at end, at minimum, standard deviation]
            1 - exp2(t2[-1], a0, a1, b1, p1a[0], p1a[1])/y2[-1],
            1 - exp2(t2[0], a0, a1, b1, p1a[0], p1a[1])/y2[0],
            np.std(np.array(
                [exp2(d['t'], a0, a1, b1, p1a[0], p1a[1]), d['y']]
            ))/y2[-1],
        ]
        values = [  # [impulsive value, minimum time, minimum value]
            d['y'][0],
            t2[0],
            y2[0],
        ]
    else:
        fit1a_t = exp1nc(d['t'], a2, b2)
        plot1a, = plt.plot(
            d['t'], fit1 + fit1a_t, 'r--',
            label='Exponential type 2',
        )

    plt.xlabel(r'$t^*$', fontsize='xx-large')
    plt.ylabel(r'$C_L/\Delta\alpha\ \ [\mathrm{rad}^{-1}]$',
               fontsize='xx-large')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, 0, ymax])

    if d['y'][-1] < 0.8*d['y'][0]:
        loc = 1  # top right
    else:
        loc = 4  # bottom right
    plt.legend(
        handles=[
            plot0a,
            plot0b,
            plot1,
            plot1a,
            # plot1b,
            # plot3,
        ],
        loc=loc,  # bottom right
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

    if a2 == 0 and b2 == 0:
        csv = open('output/' + filename + '.csv', 'w')
        csv.write('t,exp1,exp1+exp1nc')
        csv.write(' [y = {a0} - {a1} exp(-{b1} t)'.format(
            a0=a0, a1=a1, b1=b1,
        ))
        csv.write(' + {a1} exp(-{b1} t)]\n'.format(
            a1=p1a[0], b1=p1a[1],
        ))
        for i in range(len(x)):
            csv.write('{xi},{y1i},{y1ai}\n'.format(
                xi=x[i],
                y1i=fit1[i],
                y1ai=fit1a[i]+fit1[i],
            ))
        csv.close()

        data = {
            'p1a': p1a,
            'p1a_cov': p1a_cov,
            'p1a_err': p1a_err,
            'error': error,
        }
    else:
        data = {}

    return data


def test_functions(d, AR, s, M):
    """test_functions(ar, s, M)

    Run the testfit function for pre-defined combinations of AR, s, M.

    Parameters
    ----------
    d : dict
        processed data source (e.g.: from running fit_multiple)
    AR : int or float
        Aspect ratio
    s : int or float
        Sweep angle in degrees
    M : int or float
        Mach number

    """
    if AR == 20 and s == 0 and M == 0.3:
        source = 'b10s0M0_3'
        a0 = 5.37
        a1 = 2.37268938865577
        b1 = 0.261207637872903
        filename = 'b10s0M0_3-testfit'
    elif AR == 8 and s == 0 and M == 0.3:
        source = 'b4s0M0_3'
        a0 = 4.66
        a1 = 2.35654986778195
        b1 = 0.347894306977129
        filename = 'b4s0M0_3-testfit'
    elif AR == 20 and s == 30 and M == 0.5:
        source = 'b10s30M0_5'
        a0 = 5.27
        a1 = 2.5763980830261
        b1 = 0.17
        filename = 'b10s30M0_5-testfit'
    elif AR == 20 and s == 30 and M == 0.3:
        source = 'b10s30M0_3'
        a0 = 4.9
        a1 = 2.4
        b1 = 0.22
        filename = 'b10s30M0_3-testfit'
    elif AR == 8 and s == 30 and M == 0.5:
        source = 'b4s30M0_5'
        a0 = 4.55120254009252
        a1 = 2.3
        b1 = 0.28
        filename = 'b4s30M0_5-testfit'
    elif AR == 8 and s == 30 and M == 0.3:
        source = 'b4s30M0_3'
        a0 = 4.2
        a1 = 2.49710557585115
        b1 = 0.38525215362912
        filename = 'b4s30M0_3-testfit'
    else:
        raise ValueError

    data = testfit(d[source], a0, a1, b1, filename=filename)

    csv = open('output/' + filename + '.csv', 'w')
    csv.write('filename')
    csv.write(',exp1 a0,exp1 a1,exp1 b1')
    csv.write(',exp1nc a2,exp1nc b2')
    csv.write(',exp1nc a2 stddev,exp1nc b2 stddev')
    csv.write(',error at end,error at min,stddev')
    csv.write('\n')
    csv.write(filename)
    csv.write(',{exp1_a0},{exp1_a1},{exp1_b1}'.format(
        exp1_a0=a0,
        exp1_a1=a1,
        exp1_b1=b1,
    ))
    csv.write(',{exp1nc_a2},{exp1nc_b2}'.format(
        exp1nc_a2=data['p1a'][0],
        exp1nc_b2=data['p1a'][1],
    ))
    # Standard deviations
    csv.write(',{exp1nc_a2},{exp1nc_b2}'.format(
        exp1nc_a2=data['p1a_err'][0],
        exp1nc_b2=data['p1a_err'][1],
    ))
    # Errors
    csv.write(',{err0},{err1},{err2}'.format(
        err0=data['error'][0],
        err1=data['error'][1],
        err2=data['error'][2],
    ))
    csv.write('\n')
    csv.close()

    return
