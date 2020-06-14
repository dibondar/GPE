from numba import njit # compile python
import matplotlib.pyplot as plt # plotting facility
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.interpolate import UnivariateSpline

from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation

for omega in [4.,]:
    @njit
    def v(x, t=0.):
        """
        Potential energy
        """
        return 0.5 * (omega * x) ** 2


    @njit
    def diff_v(x, t=0.):
        """
        the derivative of the potential energy for Ehrenfest theorem evaluation
        """
        return (omega) ** 2 * x


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


    # save parameters as a separate bundle
    harmonic_osc_params = dict(
        x_grid_dim=512,
        x_amplitude=12.,
        v=v,
        k=k,

        diff_v=diff_v,
        diff_k=diff_k,

        g=1000,

        # epsilon=1e-2,
    )

    ##################################################################################################

    # create the harmonic oscillator with time-independent hamiltonian
    harmonic_osc = SplitOpGPE1D(dt=0.001, **harmonic_osc_params)


    # set the initial condition
    harmonic_osc.set_wavefunction(
        #lambda x: np.exp(-1 * (x - 0.01) ** 2)
        imag_time_gpe1D(dt=1e-5, **harmonic_osc_params)[0]
    )

    # get time duration of 6 periods
    T = 3 * 2. * np.pi / omega

    # propagate till time T and for each time step save a probability density
    wavefunctions = [harmonic_osc.propagate(t).copy() for t in np.arange(0, T, harmonic_osc.dt)]

    plt.title(
        "Test 1: Time evolution of harmonic oscillator with $\\omega$ = {:.1f} (a.u.)".format(omega)
    )

    # plot the time dependent density
    plt.imshow(
        np.abs(wavefunctions) ** 2,
        # some plotting parameters
        origin='lower',
        extent=[harmonic_osc.x.min(), harmonic_osc.x.max(), 0., T],
        norm=SymLogNorm(vmin=1e-13, vmax=1., linthresh=1e-15)
    )
    plt.xlabel('coordinate $x$ (a.u.)')
    plt.ylabel('time $t$ (a.u.)')
    plt.colorbar()
    plt.show()

    ##################################################################################################

    times = harmonic_osc.times

    plt.subplot(131)
    plt.title("Verify the first Ehrenfest theorem")

    plt.plot(
        times,
        # calculate the derivative using the spline interpolation
        # because times is not a linearly spaced array
        UnivariateSpline(times, harmonic_osc.x_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{x}\\rangle / dt$'
    )
    plt.plot(
        times,
        harmonic_osc.x_average_rhs,
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
        UnivariateSpline(times, harmonic_osc.p_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{p}\\rangle / dt$'
    )
    plt.plot(harmonic_osc.times, harmonic_osc.p_average_rhs, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title("The expectation value of the hamiltonian")

    # Analyze how well the energy was preserved
    h = np.array(harmonic_osc.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
            100. * (1. - h.min() / h.max())
        )
    )

    plt.plot(times, h)
    plt.ylabel('energy')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    plt.title('time increments $dt$')
    plt.plot(harmonic_osc.time_incremenets)
    plt.ylabel('$dt$')
    plt.xlabel('time step')
    plt.show()