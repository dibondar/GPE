"""
This is an example how to parallelize calculations in assymetric_tunneling.py
"""
from multiprocessing import Pool
from numba import njit # compile python
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation

########################################################################################################################
#
# Parameters
#
########################################################################################################################

propagation_dt = 1e-4
g = 2194.449140

hight_assymetric = 1e2

delta = 3.5

@njit
def v(x, t=0.):
    """
    Potential energy
    """
    return 0.5 * x ** 2 + x ** 2 * hight_assymetric * np.exp(-(x / delta) ** 2) * (x < 0)


@njit
def diff_v(x, t=0.):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    return x + (2. * x - 2. * (1. / delta) ** 2 * x ** 3) * hight_assymetric * np.exp(-(x / delta) ** 2) * (x < 0)


@njit
def diff_k(p, t=0.):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p


@njit
def k(p, t=0.):
    """
    Non-relativistic kinetic energy
    """
    return 0.5 * p ** 2

@njit
def initial_trap(x, t=0):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    # omega = 2 * Pi * 100Hz
    return 12.5 * (x + 20.) ** 2

########################################################################################################################
#
# This is where the functions to be run in parallel are declared
#
########################################################################################################################


def run_single_case(params):
    """
    This function that will be run on different processors.
    First it will find the initial state and then it will propagate
    :param params: dict with parameters for propagation
    :return: dict contaning results
    """
    # get the initial state
    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        dt=1e-3,
        epsilon=1e-8,
        **params
    )

    init_state, mu = imag_time_gpe1D(
        v=params['initial_trap'],
        g=g,
        init_wavefunction=init_state,
        dt=1e-4,
        epsilon=1e-9,
        **params
    )

    ########################################################################################################################
    #
    # Propagate GPE equation
    #
    ########################################################################################################################

    print("\nPropagate GPE equation")

    gpe_propagator = SplitOpGPE1D(
        v=v,
        g=g,
        dt=propagation_dt,
        **params
    ).set_wavefunction(init_state)

    # propagate till time T and for each time step save a probability density
    gpe_wavefunctions = [
        gpe_propagator.propagate(t).copy() for t in params['times']
    ]

    ####################################################################################################################
    #
    # Propagate Schrodinger equation
    #
    ####################################################################################################################

    print("\nPropagate Schrodinger equation")

    schrodinger_propagator = SplitOpGPE1D(
        v=v,
        g=0.,
        dt=propagation_dt,
        **params
    ).set_wavefunction(init_state)

    # propagate till time T and for each time step save a probability density
    schrodinger_wavefunctions = [
        schrodinger_propagator.propagate(t).copy() for t in params['times']
    ]

    # bundle results into a dictionary
    return {
        'init_state': init_state,
        'mu': mu,

        # bundle separately GPE data
        'gpe': {
            'wavefunctions': gpe_wavefunctions,
            'extent': [gpe_propagator.x.min(), gpe_propagator.x.max(), 0., max(gpe_propagator.times)],
            'times': gpe_propagator.times,

            'x_average': gpe_propagator.x_average,
            'x_average_rhs': gpe_propagator.x_average_rhs,

            'p_average': gpe_propagator.p_average,
            'p_average_rhs': gpe_propagator.p_average_rhs,
            'hamiltonian_average': gpe_propagator.hamiltonian_average,

            'time_increments': gpe_propagator.time_increments,

            'dx': gpe_propagator.dx,
            'x': gpe_propagator.x,
        },

        # bundle separately Schrodinger data
        'schrodinger': {
            'wavefunctions': schrodinger_wavefunctions,
            'extent': [schrodinger_propagator.x.min(), schrodinger_propagator.x.max(), 0., max(schrodinger_propagator.times)],
            'times': schrodinger_propagator.times,

            'x_average': schrodinger_propagator.x_average,
            'x_average_rhs': schrodinger_propagator.x_average_rhs,

            'p_average': schrodinger_propagator.p_average,
            'p_average_rhs': schrodinger_propagator.p_average_rhs,
            'hamiltonian_average': schrodinger_propagator.hamiltonian_average,

            'time_increments': schrodinger_propagator.time_increments,
        },
    }

########################################################################################################################
#
# the code below will be run serially (and only once). This is where we launch parallel computations and do analysis
#
########################################################################################################################


if __name__ == '__main__':

    # get time duration of 2 periods
    T = 0.1 * 2. * np.pi
    times = np.linspace(0, T, 500)

    # save parameters as a separate bundle
    sys_params = dict(
        x_grid_dim=1 * 1024,
        x_amplitude=80.,

        k=k,

        initial_trap=initial_trap,

        diff_v=diff_v,
        diff_k=diff_k,

        times=times,
    )

    # create parameters for the flipped case
    sys_params_flipped = sys_params.copy()

    # we just need to flip the initial trap for cooling
    sys_params_flipped['initial_trap'] = njit(lambda x, t: initial_trap(-x, t))

    ####################################################################################################################
    #
    # Run calculations in parallel
    #
    ####################################################################################################################

    # get the unflip and flip simulations run in parallel;
    # results will be saved in qsys and qsys_flipped, respectively
    with Pool() as pool:
        qsys, qsys_flipped = pool.map(run_single_case, [sys_params, sys_params_flipped])

    ####################################################################################################################
    #
    # Generate plots to test the propagation
    #
    ####################################################################################################################

    def analyze_propagation(qsys, title):
        """
        Make plots to check the quality of propagation
        :param qsys: dict with parameters
        :param title: str
        :return: None
        """
        plt.title(title)

        # plot the time dependent density
        extent = qsys['extent']

        plt.imshow(
            np.abs(qsys['wavefunctions']) ** 2,
            # some plotting parameters
            origin='lower',
            extent=extent,
            aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
            norm=SymLogNorm(vmin=1e-13, vmax=1., linthresh=1e-15)
        )
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')
        plt.colorbar()
        plt.savefig(title + '.pdf')

        plt.show()

        times = qsys['times']

        plt.subplot(131)
        plt.title("Verify the first Ehrenfest theorem")

        plt.plot(
            times,
            # calculate the derivative using the spline interpolation
            # because times is not a linearly spaced array
            UnivariateSpline(times, qsys['x_average'], s=0).derivative()(times),
            '-r',
            label='$d\\langle\\hat{x}\\rangle / dt$'
        )
        plt.plot(
            times,
            qsys['x_average_rhs'],
            '--b',
            label='$\\langle\\hat{p}\\rangle$'
        )
        plt.legend()
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(132)
        plt.title("Verify the second Ehrenfest theorem")

        plt.plot(
            times,
            # calculate the derivative using the spline interpolation
            # because times is not a linearly spaced array
            UnivariateSpline(times, qsys['p_average'], s=0).derivative()(times),
            '-r',
            label='$d\\langle\\hat{p}\\rangle / dt$'
        )
        plt.plot(times, qsys['p_average_rhs'], '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend()
        plt.ylabel('force')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(133)
        plt.title("The expectation value of the hamiltonian")

        # Analyze how well the energy was preserved
        h = np.array(qsys['hamiltonian_average'])
        print(
            "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
                100. * (1. - h.min() / h.max())
            )
        )
        print("Initial energy {:.4e}".format(h[0]))

        plt.plot(times, h)
        plt.ylabel('energy')
        plt.xlabel('time $t$ (a.u.)')

        plt.show()

        plt.title('time increments $dt$')
        plt.plot(qsys['time_increments'])
        plt.ylabel('$dt$')
        plt.xlabel('time step')
        plt.show()

    ####################################################################################################################
    #
    # Analyze the simulations
    #
    ####################################################################################################################

    # Analyze the schrodinger propagation
    analyze_propagation(qsys['schrodinger'], "Schrodinger evolution")

    # Analyze the Flipped schrodinger propagation
    analyze_propagation(qsys_flipped['schrodinger'], "Flipped Schrodinger evolution")

    # Analyze the GPE propagation
    analyze_propagation(qsys['gpe'], "GPE evolution")

    # Analyze the Flipped GPE propagation
    analyze_propagation(qsys_flipped['gpe'], "Flipped GPE evolution")

    ####################################################################################################################
    #
    # Calculate the transmission probability
    #
    ####################################################################################################################

    dx = qsys['gpe']['dx']
    size = qsys['gpe']['x'].size
    x_cut = int(0.6 * size)
    x_cut_flipped = int(0.4 * size)

    plt.subplot(121)
    plt.plot(
        times,
        np.sum(np.abs(qsys['schrodinger']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dx,
        label='Schrodinger'
    )
    plt.plot(
        times,
        np.sum(np.abs(qsys_flipped['schrodinger']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
        label='Flipped Schrodinger'
    )
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("transmission probability")

    plt.subplot(122)
    plt.plot(
        times,
        np.sum(np.abs(qsys['gpe']['wavefunctions'])[:, x_cut:] ** 2, axis=1) * dx,
        label='GPE'
    )
    plt.plot(
        times,
        np.sum(np.abs(qsys_flipped['gpe']['wavefunctions'])[:, :x_cut_flipped] ** 2, axis=1) * dx,
        label='Flipped GPE'
    )
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("transmission probability")

    plt.show()

    ####################################################################################################################
    #
    # Plot the potential
    #
    ####################################################################################################################

    plt.title('Potential')
    x = qsys['gpe']['x']
    plt.plot(x, v(x))
    plt.xlabel('$x / 2.4\mu m$ ')
    plt.ylabel('$v(x)$')
    plt.show()
