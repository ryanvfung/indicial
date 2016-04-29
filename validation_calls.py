# -*- coding: utf-8 -*-
"""
Indicial response - calls made after running indicial.py
"""


from indicial import *


# Initialise variables
tau = {}
y_data = {}
y_indicial = {}
yw_indicial = {}
y2_indicial = {}
y_steps = {}
plots = {}


def perturbation(t):
    """perturbation(t)

    Returns an array of

    α(t) = α_A sin(2πkt)

    where:
        the peak amplitude:
            α_A = 1°
        the reduced frequency:
            k = 0.08

    Parameters
    ----------
    t : np.array of floats
        time steps

    Returns
    -------
    np.array of floats

    """
    alpha_A = np.pi/180
    k = 0.08
    return alpha_A * np.sin(2*np.pi*k*t)


def validate(time, step, M, closeup=False):
    """validate(time, step, M, closeup=False)

    Runs the indicial ROM with:
     * the generalised indicial response model
     * the CFD indicial response for aspect ratio of 8
     * Wagner function
    and compares it to the CFD simulation results by graphing

    Requires the CFD indicial response data to be available, such as by running
        from modelling import *
        fit_multiple([
            'b4s0M0_3',
            'b4s0M0_5',
            'b4s0M0_7',
        ], d)

    Parameters
    ----------
    time : int or float
        ROM simulation time duration
    step : int
        Number of time steps per unit nondimensional time
    M : 3, 5 or 7
        Corresponds to Mach number of 0.3, 0.5, 0.7
    closeup : boolean
        if True, change graph plots to close up view

    """
    t = time_step_generator(time, step)
    f = perturbation(t)
    t2 = time_step_generator(time, 1/d['b4s0M0_{}'.format(M)]['t'][0])
    w = {
        3: wagner(t),
        5: wagner5(t),
        7: wagner7(t),
    }
    y_indicial[M] = indicial1(t, f, onerad(t, M/10., 4, 0),
                              filename='ar4_M{}'.format(M))
    yw_indicial[M] = onerad(9999, M/10., 4, 0) * \
        indicial1(t, f, w[M], filename='ar4_M{}_wagner'.format(M))
    y2_indicial[M] = indicial1(t2, perturbation(t2),
                               d['b4s0M0_{}'.format(M)]['y'])
    plt.figure('{}_compare'.format(M))
    plot1, = plt.plot(tau[M], y_data[M], '-', label='CFD')
    plot2, = plt.plot(t, y_indicial[M], '--',
                      label='Indicial ROM, modelled response')
    plot3, = plt.plot(t2, y2_indicial[M], '--',
                      label='Indicial ROM, CFD indicial response for AR = 8')
    plot4, = plt.plot(t, yw_indicial[M], '--',
                      label='Indicial ROM, Wagner function')
    plt.xlabel(r'$t^*$', fontsize='xx-large')
    plt.ylabel(r'$C_L$', fontsize='xx-large')
    plt.legend(
        handles=[
            plot1,
            plot2,
            plot3,
            plot4,
        ],
        fontsize='large',
    )
    xmin, xmax, ymin, ymax = plt.axis()
    filename = '-output-compare'
    if closeup is True:
        xmax = 10
        filename += '-closeup'
    if M == 7:
        plt.axis([xmin, xmax, ymin, 0.17])
    else:
        plt.axis([xmin, xmax, ymin, 0.14])
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(
        'output/ar4_M{}{}.png'.format(M, filename),
        format='png',
        dpi=150,
        bbox_inches='tight',
    )
    fig.savefig(
        'output/ar4_M{}{}.svg'.format(M, filename),
        format='svg',
        dpi=300,
        bbox_inches='tight',
    )
    csv = open('output/ar4_M{}{}.csv'.format(M, filename), 'w')
    for i in range(len(t)):
        csv.write('{t},{y}\n'.format(
            t=t[i], y=y_indicial[M][i],
        ))
    csv.close()


def validate_all():
    """validate_all()

    Process CFD simulation results for plotting purposes,
    run validation code for Mach number 0.3, 0.5, 0.7,
    with full scale and close up graph views.

    """
    tau[3], y_data[3] = extract_2d_data('edge_03_ar4_s0.csv')
    tau[5], y_data[5] = extract_2d_data('edge_05_ar4_s0.csv')
    tau[7], y_data[7] = extract_2d_data('edge_07_ar4_s0.csv')
    validate(62, 160, M=3)
    validate(62, 160, M=5)
    validate(49, 160, M=7)
    validate(15, 160, M=3, closeup=True)
    validate(15, 160, M=5, closeup=True)
    validate(15, 160, M=7, closeup=True)


def comparison():
    """comparison()

    Print values of interest after running validation cases.

    """
    t = time_step_generator(62, 80)
    for M in [3, 5, 7]:
        max_values = [
            max(y_indicial[M]),
            max(y2_indicial[M]),
            max(ye_data[M]),
            max(yw_indicial[M]),
        ]
        print('M={}: {}'.format(M, max_values))
        print('CFD max time: {}'.format(tau[M][y_data[M].argmax()]))
        print('Wagner max time: {}'.format(t[yw_indicial[M].argmax()]))


def compare_step_sizes():
    """compare_step_sizes()

    For Mach number of 0.7, run the indicial ROM with a nondimensional time of
    15 to observe effect of increasing step size.

    """
    M = 7
    time = 15
    step_sizes = [40, 80, 160, 320, 640]
    for step_size in step_sizes:
        t = time_step_generator(time, step_size)
        f = perturbation(t)
        y_steps[step_size] = indicial1(
            t,
            f,
            onerad(t, M/10., 4, 0),
            filename='time_step_compare_M{}'.format(M)
        )


def plot_step_sizes(closeup=False):
    """plot_step_sizes(closeup=False)

    After running compare_step_sizes(), plot the results.
    For Mach number of 0.7, run the indicial ROM with a nondimensional time of
    15 to observe effect of increasing step size.

    Parameters
    ----------
    closeup : boolean
        if True, change graph plots to close up view

    """
    plot1, = plt.plot(tau[M], y_data[M], '-', label='CFD simulation')
    handles = [plot1]
    for step_size in step_sizes:
        t = time_step_generator(time, step_size)
        plots[step_size], = plt.plot(t, y_steps[step_size], '--',
                                     label='{} time steps'.format(step_size))
        handles.append(plots[step_size])
    plt.xlabel(r'$t^*$', fontsize='xx-large')
    plt.ylabel(r'$C_L$', fontsize='xx-large')
    xmin, xmax, ymin, ymax = plt.axis()
    filename = str(M)
    if closeup is True:
        loc = 4
        xmax = 1
        ymin = -0.005
        ymax = 0.04
        filename += '-closeup'
    else:
        loc = 1
        xmax = 15
        ymax 0.17
    plt.legend(
        handles=handles,
        loc=loc,
        fontsize='large',
    )
    plt.axis([xmin, xmax, ymin, ymax])
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.savefig(
        'output/time_step_compare_M{}.png'.format(filename),
        format='png',
        dpi=150,
        bbox_inches='tight',
    )
    fig.savefig(
        'output/time_step_compare_M{}.svg'.format(filename),
        format='svg',
        dpi=300,
        bbox_inches='tight',
    )
