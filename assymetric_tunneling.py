from numba import njit # compile python
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy.interpolate import UnivariateSpline
from split_op_gpe1D import SplitOpGPE1D, imag_time_gpe1D # class for the split operator propagation

g = 2194.449140
propagation_dt = 1e-4

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


# save parameters as a separate bundle
params = dict(
    x_grid_dim=1 * 1024,
    x_amplitude=80.,

    k=k,

    diff_v=diff_v,
    diff_k=diff_k,

    # epsilon=1e-2,
)


########################################################################################################################
#
# Get the initial state
#
########################################################################################################################

@njit
def initial_trap(x, t=0):
    """
    Trapping potential to get the initial state
    :param x:
    :return:
    """
    # omega = 2 * Pi * 100Hz
    return 12.5 * (x + 20.) ** 2


init_state, mu = imag_time_gpe1D(
    v=initial_trap,
    g=g,
    dt=1e-3,
    epsilon=1e-8,
    **params
)

init_state, mu = imag_time_gpe1D(
    init_wavefunction=init_state,
    g=g,
    v=initial_trap,
    dt=1e-4,
    epsilon=1e-9,
    **params
)

flipped_initial_trap = njit(lambda x, t: initial_trap(-x, t))

flipped_init_state, mu = imag_time_gpe1D(
    v=flipped_initial_trap,
    g=g,
    dt=1e-3,
    epsilon=1e-8,
    **params
)

flipped_init_state, mu = imag_time_gpe1D(
    init_wavefunction=flipped_init_state,
    g=g,
    v=flipped_initial_trap,
    dt=1e-4,
    epsilon=1e-9,
    **params
)

# qsys = SplitOpGPE1D(
#     v=initial_trap,
#     g=g,
#     dt=1e-3,
#     **params
# ).set_wavefunction(init_state)
# test_init_state = qsys.propagate(0.05)
#
# x = qsys.x
# plt.semilogy(x, np.abs(init_state), label='initial condition')
# plt.semilogy(x, np.abs(test_init_state), label='propagated init condition')
#
# plt.semilogy(x, v(x))
# plt.xlabel('$x$')
# plt.ylabel('$v(x)$')
# plt.legend()
# plt.show()

########################################################################################################################
#
# Generate plots to test the propagation
#
########################################################################################################################


def analyze_propagation(qsys, wavefunctions, title):
    """
    Make plots to check the quality of propagation
    :param qsys: an instance of SplitOpGPE1D
    :param wavefunctions: list of numpy.arrays
    :param title: str
    :return: None
    """
    plt.title(title)

    # plot the time dependent density
    extent = [qsys.x.min(), qsys.x.max(), 0., T]

    plt.imshow(
        np.abs(wavefunctions) ** 2,
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

    times = qsys.times

    plt.subplot(131)
    plt.title("Verify the first Ehrenfest theorem")

    plt.plot(
        times,
        # calculate the derivative using the spline interpolation
        # because times is not a linearly spaced array
        UnivariateSpline(times, qsys.x_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{x}\\rangle / dt$'
    )
    plt.plot(
        times,
        qsys.x_average_rhs,
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
        UnivariateSpline(times, qsys.p_average, s=0).derivative()(times),
        '-r',
        label='$d\\langle\\hat{p}\\rangle / dt$'
    )
    plt.plot(qsys.times, qsys.p_average_rhs, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title("The expectation value of the hamiltonian")

    # Analyze how well the energy was preserved
    h = np.array(qsys.hamiltonian_average)
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
    plt.plot(qsys.time_increments)
    plt.ylabel('$dt$')
    plt.xlabel('time step')
    plt.show()


########################################################################################################################
#
# Propagate GPE equation
#
########################################################################################################################

print("\nPropagate GPE equation")

gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
).set_wavefunction(init_state)

# get time duration of 2 periods
T = 0.1 * 2. * np.pi
times = np.linspace(0, T, 500)

# propagate till time T and for each time step save a probability density
gpe_wavefunctions = [
     gpe_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Propagate GPE equation with flipped initial condition
#
########################################################################################################################

print("\nPropagate GPE equation with flipped initial condition")

flipped_gpe_qsys = SplitOpGPE1D(
    v=v,
    g=g,
    dt=propagation_dt,
    **params
).set_wavefunction(flipped_init_state)

# propagate till time T and for each time step save a probability density
flipped_gpe_wavefunctions = [
     flipped_gpe_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Propagate Schrodinger equation
#
########################################################################################################################

print("\nPropagate Schrodinger equation")

schrodinger_qsys = SplitOpGPE1D(
    v=v,
    g=0.,
    dt=propagation_dt,
    **params
).set_wavefunction(init_state)

# propagate till time T and for each time step save a probability density
schrodinger_wavefunctions = [
     schrodinger_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Propagate Schrodinger equation with flipped initial condition
#
########################################################################################################################

print("\nSchrodinger equation with flipped initial condition")

flipped_schrodinger_qsys = SplitOpGPE1D(
    v=v,
    g=0.,
    dt=propagation_dt,
    **params
).set_wavefunction(flipped_init_state)

# propagate till time T and for each time step save a probability density
flipped_schrodinger_wavefunctions = [
     flipped_schrodinger_qsys.propagate(t).copy() for t in times
]

########################################################################################################################
#
# Analyze the simulations
#
########################################################################################################################

# Analyze the schrodinger propagation
analyze_propagation(schrodinger_qsys, schrodinger_wavefunctions, "Schrodinger evolution")

# Analyze the Flipped schrodinger propagation
analyze_propagation(
    flipped_schrodinger_qsys,
    flipped_schrodinger_wavefunctions, # [psi[::-1] for psi in flipped_schrodinger_wavefunctions],
    "Flipped Schrodinger evolution"
)

# Analyze the GPE propagation
analyze_propagation(gpe_qsys, gpe_wavefunctions, "GPE evolution")

# Analyze the Flipped GPE propagation
analyze_propagation(
    flipped_gpe_qsys,
    flipped_gpe_wavefunctions, # [psi[::-1] for psi in flipped_gpe_wavefunctions],
    "Flipped GPE evolution"
)

########################################################################################################################
#
# Calculate the transmission probability
#
########################################################################################################################

dx = gpe_qsys.dx
x_cut = int(0.6 * gpe_qsys.wavefunction.size)
x_cut_flipped = int(0.4 * gpe_qsys.wavefunction.size)

plt.subplot(121)
plt.plot(
    times,
    np.sum(np.abs(schrodinger_wavefunctions)[:, x_cut:] ** 2, axis=1) * dx,
    label='Schrodinger'
)
plt.plot(
    times,
    np.sum(np.abs(flipped_schrodinger_wavefunctions)[:, :x_cut_flipped] ** 2, axis=1) * dx,
    label='Flipped Schrodinger'
)
plt.legend()
plt.xlabel("time")
plt.ylabel("transmission probability")

plt.subplot(122)
plt.plot(
    times,
    np.sum(np.abs(gpe_wavefunctions)[:, x_cut:] ** 2, axis=1) * dx,
    label='GPE'
)
plt.plot(
    times,
    np.sum(np.abs(flipped_gpe_wavefunctions)[:, :x_cut_flipped] ** 2, axis=1) * dx,
    label='Flipped GPE'
)
plt.legend()
plt.xlabel("time")
plt.ylabel("transmission probability")

plt.show()

########################################################################################################################
#
# Plot the potential
#
########################################################################################################################

plt.title('Potential')
x = gpe_qsys.x
plt.plot(x, v(x))
plt.xlabel('$x / 2.4\mu m$ ')
plt.ylabel('$v(x)$')
plt.show()