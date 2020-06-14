import numpy as np
import pyfftw
import pickle
from numpy import linalg  # Linear algebra for dense matrix
from numba import njit, jit
from numba.targets.registry import CPUDispatcher
from types import FunctionType
from multiprocessing import cpu_count

def imag_time_gpe1D(*, x_grid_dim, x_amplitude, v, k, dt, g, init_wavefunction=None, epsilon=1e-7,
                    abs_boundary=1., fftw_wisdom_fname='fftw.wisdom', **kwargs):
    """
    Imaginary time propagator to get the ground state and chemical potential

    :param x_grid_dim: the grid size
    :param x_amplitude: the maximum value of the coordinates
    :param v: the potential energy (as a function)
    :param k: the kinetic energy (as a function)
    :param dt: initial time increment
    :param init_wavefunction: initial guess for wavefunction
    :param g: the coupling constant
    :param epsilon: relative error tolerance
    :param abs_boundary: absorbing boundary
    :param fftw_wisdom_fname: File name from where the FFT wisdom will be loaded from and saved to
    :param kwargs: ignored

    :return: wavefunction, chemical potential
    """
    print("\nStarting imaginary time propagation")

    # Check that all attributes were specified
    # make sure self.x_amplitude has a value of power of 2
    assert 2 ** int(np.log2(x_grid_dim)) == x_grid_dim, \
        "A value of the grid size (x_grid_dim) must be a power of 2"

    ####################################################################################################
    #
    #   Initialize Fourier transform for efficient calculations
    #
    ####################################################################################################

    # Load the FFTW wisdom
    try:
        with open(fftw_wisdom_fname, 'rb') as fftw_wisdow:
            pyfftw.import_wisdom(pickle.load(fftw_wisdow))
    except FileNotFoundError:
        pass

    # allocate the array for wave function
    wavefunction = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

    # allocate the array for wave function in momentum representation
    wavefunction_p = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

    # allocate the array for calculating the momentum representation for the energy evaluation
    wavefunction_p_ = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

    # parameters for FFT
    fft_params = {
        "flags": ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'),
        "threads": cpu_count(),
        "planning_timelimit": 60,
    }

    # FFT
    fft = pyfftw.FFTW(wavefunction, wavefunction_p, **fft_params)

    # iFFT
    ifft = pyfftw.FFTW(wavefunction_p, wavefunction, direction='FFTW_BACKWARD', **fft_params)

    # fft for momentum representation
    fft_p = pyfftw.FFTW(wavefunction_p, wavefunction_p_, **fft_params)

    # Save the FFTW wisdom
    with open(fftw_wisdom_fname, 'wb') as fftw_wisdow:
        pickle.dump(pyfftw.export_wisdom(), fftw_wisdow)

    ####################################################################################################
    #
    #   Initialize grids
    #
    ####################################################################################################

    # get coordinate step size
    dx = 2. * x_amplitude / x_grid_dim

    # generate coordinate range
    x = (np.arange(x_grid_dim) - x_grid_dim / 2) * dx

    # generate momentum range as it corresponds to FFT frequencies
    p = (np.arange(x_grid_dim) - x_grid_dim / 2) * (np.pi / x_amplitude)

    # tha array of alternating signs for going to the momentum representation
    minues = (-1) ** np.arange(x_grid_dim)

    # evaluate the potential energy
    v = v(x, 0.)
    v -= v.min()

    # evaluate the kinetic energy
    k = k(p, 0.)
    k -= k.min()

    # pre-calculate the absorbing potential and the sequence of alternating signs
    abs_boundary = (abs_boundary if isinstance(abs_boundary, (float, int)) else abs_boundary(x))

    # precalucate the exponent of the potential and kinetic energy
    img_exp_v = (-1) ** np.arange(x.size) * abs_boundary * np.exp(-0.5 * dt * v)
    img_exp_k = np.exp(-dt * k)

    # initial guess for the wave function
    wavefunction[:] = (np.exp(-v) + 0j if init_wavefunction is None else init_wavefunction)

    @njit
    def exp_potential(psi):
        """
        Modulate the wavefunction with the nonlinear interaction potential in GPE
        :param psi: wavefunction
        :return: None
        """
        psi *= img_exp_v
        psi /= linalg.norm(psi) * np.sqrt(dx)
        psi *= np.exp(-0.5 * dt * g * np.abs(psi) ** 2)

    @jit
    def get_energy(psi, pis_p):
        """
        Calculate the energy for a given wave function and its momentum representaion
        :return: float
        """
        density = np.abs(psi) ** 2
        density /= density.sum()

        energy = np.sum((v + 0.5 * g * density / dx) * density)

        # get momentum density
        density = np.abs(pis_p) ** 2
        density /= density.sum()

        energy += np.sum(k * density)

        return energy

    counter = 0
    energy = 0.
    energy_previous = np.infty

    # reaped until energy increases or convergence
    while (energy_previous > energy) and (1 - energy / energy_previous > epsilon):

        exp_potential(wavefunction)

        # going to the momentum representation
        wavefunction_p = fft(wavefunction)

        wavefunction_p *= img_exp_k

        # going back to the coordinate representation
        wavefunction = ifft(wavefunction_p)

        exp_potential(wavefunction)

        wavefunction /= linalg.norm(wavefunction) * np.sqrt(dx)

        # save previous energy
        energy_previous = (energy if energy else np.infty)

        # get the wave function in the momentum representation for getting the energy
        #wavefunction_p[:] = wavefunction
        np.copyto(wavefunction_p, wavefunction)
        wavefunction_p *= minues
        wavefunction_p_ = fft_p(wavefunction_p)

        # calculate the energy
        energy = get_energy(wavefunction, wavefunction_p_)

        # print progress report
        if counter % 2000 == 0:

            print("current ground state energy = {:.4e}".format(energy))

        counter += 1

    print("\n\nFinal current ground state energy = {:.4e}".format(energy))

    return wavefunction, energy

########################################################################################################################
#
#   Class to perform the real time propagation of GPE
#
########################################################################################################################


@njit
def relative_diff(psi_next, psi):
    """
    Efficiently calculate the relative difference of two wavefunctions. (Used in thea adaptive scheme)
    :param psi_next: numpy.array
    :param psi: numpy.array
    :return: float
    """
    return linalg.norm(psi_next - psi) / linalg.norm(psi_next)


class SplitOpGPE1D(object):
    """
    The second-order split-operator propagator of the 1D Gross–Pitaevskii equation in the coordinate representation
    with the time-dependent Hamiltonian

        H = K(p, t) + V(x, t) + g * abs(wavefunction) ** 2

    """
    def __init__(self, *, x_grid_dim, x_amplitude, v, k, dt, g,
                 epsilon=1e-2, diff_k=None, diff_v=None, t=0, abs_boundary=1.,
                 fftw_wisdom_fname='fftw.wisdom', **kwargs):
        """
        :param x_grid_dim: the grid size
        :param x_amplitude: the maximum value of the coordinates
        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)
        :param diff_k: the derivative of the potential energy for the Ehrenfest theorem calculations
        :param diff_v: the derivative of the kinetic energy for the Ehrenfest theorem calculations
        :param t: initial value of time
        :param dt: initial time increment
        :param g: the coupling constant
        :param epsilon: relative error tolerance
        :param abs_boundary: absorbing boundary
        :param fftw_wisdom_fname: File name from where the FFT wisdom will be loaded from and saved to
        :param kwargs: ignored
        """

        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v
        self.k = k
        self.diff_v = diff_v
        self.t = t
        self.dt = dt
        self.g = g
        self.epsilon = epsilon
        self.abs_boundary = abs_boundary

        ####################################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ####################################################################################################

        # Load the FFTW wisdom
        try:
            with open(fftw_wisdom_fname, 'rb') as fftw_wisdow:
                pyfftw.import_wisdom(pickle.load(fftw_wisdow))
        except FileNotFoundError:
            pass

        # allocate the array for wave function
        self.wavefunction = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

        # allocate an extra copy for the wavefunction necessary for adaptive time step propagation
        self.wavefunction_next = pyfftw.empty_aligned(self.x_grid_dim, dtype=np.complex)

        # allocate the array for wave function in momentum representation
        self.wavefunction_next_p = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

        # allocate the array for calculating the momentum representation for the energy evaluation
        self.wavefunction_next_p_ = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

        # parameters for FFT
        self.fft_params = {
            "flags": ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'),
            "threads": cpu_count(),
            "planning_timelimit": 60,
        }

        # FFT
        self.fft = pyfftw.FFTW(self.wavefunction_next, self.wavefunction_next_p, **self.fft_params)

        # iFFT
        self.ifft = pyfftw.FFTW(self.wavefunction_next_p, self.wavefunction_next, direction='FFTW_BACKWARD', **self.fft_params)

        # fft for momentum representation
        self.fft_p = pyfftw.FFTW(self.wavefunction_next_p, self.wavefunction_next_p_, **self.fft_params)

        # Save the FFTW wisdom
        with open(fftw_wisdom_fname, 'wb') as fftw_wisdow:
            pickle.dump(pyfftw.export_wisdom(), fftw_wisdow)

        ####################################################################################################
        #
        #   Initialize grids
        #
        ####################################################################################################

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        dx = self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        x = self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx
        # The same as
        # self.x = np.linspace(-self.x_amplitude, self.x_amplitude - self.dx , self.x_grid_dim)

        # generate momentum range as it corresponds to FFT frequencies
        p = self.p = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)

        # the relative change estimators for the time adaptive scheme
        self.e_n = self.e_n_1 = self.e_n_2 = 0

        self.previous_dt = 0

        # list of self.dt to monitor how the adaptive step method is working
        self.time_incremenets = []

        ####################################################################################################
        #
        # Codes for efficient evaluation
        #
        ####################################################################################################

        # Decide whether the potential depends on time
        try:
            v(x, 0)
            time_independent_v = False
        except TypeError:
            time_independent_v = True

        # Decide whether the kinetic energy depends on time
        try:
            k(p, 0)
            time_independent_k = False
        except TypeError:
            time_independent_k = True

        # pre-calculate the absorbing potential and the sequence of alternating signs

        abs_boundary = (abs_boundary if isinstance(abs_boundary, (float, int)) else abs_boundary(x))
        abs_boundary = (-1) ** np.arange(self.wavefunction.size) * abs_boundary

        # Cache the potential if it does not depend on time
        if time_independent_v:
            pre_calculated_v = v(x, 0.)
            v = njit(lambda _, __: pre_calculated_v)

        # Cache the kinetic energy if it does not depend on time
        if time_independent_k:
            pre_calculated_k = k(p, 0.)
            k = njit(lambda _, __: pre_calculated_k)

        @njit
        def expV(wavefunction, t, dt):
            """
            function to efficiently evaluate
                wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
            """
            wavefunction *= abs_boundary * np.exp(-0.5j * dt * (v(x, t + 0.5 * dt) + g * np.abs(wavefunction) ** 2))

        self.expV = expV

        @njit
        def expK(wavefunction, t, dt):
            """
            function to efficiently evaluate
                wavefunction *= exp(-1j * dt * k)
            """
            wavefunction *= np.exp(-1j * dt * k(p, t + 0.5 * dt))

        self.expK = expK

        ####################################################################################################

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        if diff_k and diff_v:

            # Cache the potential if it does not depend on time
            if time_independent_v:
                pre_calculated_diff_v = diff_v(x, 0.)
                diff_v = njit(lambda _, __: pre_calculated_diff_v)

            # Cache the kinetic energy if it does not depend on time
            if time_independent_k:
                pre_calculated_diff_k = diff_k(p, 0.)
                diff_k = njit(lambda _, __: pre_calculated_diff_k)

            # Get codes for efficiently calculating the Ehrenfest relations

            @njit
            def get_p_average_rhs(density, t):
                return np.sum(density * diff_v(x, t))

            self.get_p_average_rhs = get_p_average_rhs

            # The code above is equivalent to
            # self.get_p_average_rhs = njit(lambda density, t: np.sum(density * diff_v(x, t)))

            @njit
            def get_v_average(density, t):
                return np.sum((v(x, t) + 0.5 * g * density / dx) * density)

            self.get_v_average = get_v_average

            @njit
            def get_x_average(density):
                return np.sum(x * density)

            self.get_x_average = get_x_average

            @njit
            def get_x_average_rhs(density, t):
                return np.sum(diff_k(p, t) * density)

            self.get_x_average_rhs = get_x_average_rhs

            @njit
            def get_k_average(density, t):
                return np.sum(k(p, t) * density)

            self.get_k_average = get_k_average

            @njit
            def get_p_average(density):
                return np.sum(p * density)

            self.get_p_average = get_p_average

            # since the variable time propagator is used, we record the time when expectation values are calculated
            self.times = []

            # Lists where the expectation values of x and p
            self.x_average = []
            self.p_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for x and p
            self.x_average_rhs = []
            self.p_average_rhs = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # sequence of alternating signs for getting the wavefunction in the momentum representation
            self.minus = (-1) ** np.arange(self.x_grid_dim)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.is_ehrenfest = True
        else:
            # Since diff_v and diff_k are not specified, we are not going to evaluate the Ehrenfest relations
            self.is_ehrenfest = False

    def propagate(self, time_final):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_final: until what time to propagate the wavefunction
        :return: self.wavefunction
        """
        e_n = self.e_n
        e_n_1 = self.e_n_1
        e_n_2 = self.e_n_2
        previous_dt = self.previous_dt

        # copy the initial condition into the propagation array self.wavefunction_next
        np.copyto(self.wavefunction_next, self.wavefunction)

        while self.t < time_final:

            ############################################################################################################
            #
            #   Adaptive scheme propagator
            #
            ############################################################################################################

            # propagate the wavefunction by a single dt
            self.single_step_propagation(self.dt)

            e_n = relative_diff(self.wavefunction_next, self.wavefunction)

            while e_n > self.epsilon:
                # the error is to high, decrease the time step and propagate with the new time step

                self.dt *= self.epsilon / e_n

                np.copyto(self.wavefunction_next, self.wavefunction)
                self.single_step_propagation(self.dt)

                e_n = relative_diff(self.wavefunction_next, self.wavefunction)

            # accept the current wave function
            np.copyto(self.wavefunction, self.wavefunction_next)

            # save self.dt for monitoring purpose
            self.time_incremenets.append(self.dt)

            # increment time
            self.t += self.dt

            # calculate the Ehrenfest theorems
            self.get_ehrenfest()

            ############################################################################################################
            #
            #   Update time step via the Evolutionary PID controller
            #
            ############################################################################################################

            # overwrite the zero values of e_n_1 and e_n_2
            previous_dt = (previous_dt if previous_dt else self.dt)
            e_n_1 = (e_n_1 if e_n_1 else e_n)
            e_n_2 = (e_n_2 if e_n_2 else e_n)

            # the adaptive time stepping method from
            #   http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture8.pdf
            # self.dt *= (e_n_1 / e_n) ** 0.075 * (self.epsilon / e_n) ** 0.175 * (e_n_1 ** 2 / e_n / e_n_2) ** 0.01

            # the adaptive time stepping method from
            #   https://linkinghub.elsevier.com/retrieve/pii/S0377042705001123
            self.dt *= (self.epsilon ** 2 / e_n / e_n_1 * previous_dt / self.dt) ** (1 / 12.)

            # update the error estimates in order to go next to the next step
            e_n_2, e_n_1 = e_n_1, e_n

        # save the error estimates
        self.previous_dt = previous_dt
        self.e_n = e_n
        self.e_n_1 = e_n_1
        self.e_n_2 = e_n_2

        return self.wavefunction

    def single_step_propagation(self, dt):
        """
        Propagate the wavefunction, saved in self.wavefunction_next, by a single time-step
        :param dt: time-step
        :return: None
        """
        wavefunction = self.wavefunction_next

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(wavefunction, self.t, dt)

        # going to the momentum representation
        wavefunction = self.fft(wavefunction)

        # efficiently evaluate
        #   wavefunction *= exp(-1j * dt * k)
        self.expK(wavefunction, self.t, dt)

        # going back to the coordinate representation
        wavefunction = self.ifft(wavefunction)

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(wavefunction, self.t, dt)

        # normalize
        # this line is equivalent to
        # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction) ** 2 ) * self.dx)
        wavefunction /= linalg.norm(wavefunction) * np.sqrt(self.dx)

    def get_ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.is_ehrenfest:
            # alias
            density = self.wavefunction_next_p

            # evaluate the coordinate density
            np.abs(self.wavefunction, out=density)
            density *= density
            # normalize
            density /= density.sum()

            # save the current value of <x>
            self.x_average.append(
                self.get_x_average(density.real)
            )

            self.p_average_rhs.append(
                -self.get_p_average_rhs(density.real, self.t)
            )

            # save the potential energy
            self.hamiltonian_average.append(
                self.get_v_average(density.real, self.t)
            )

            # calculate density in the momentum representation
            np.copyto(density, self.wavefunction)
            density *= self.minus
            density = self.fft_p(density)

            # get the density in the momentum space
            np.abs(density, out=density)
            density *= density
            # normalize
            density /= density.sum()

            # save the current value of <p>
            self.p_average.append(self.get_p_average(density.real))

            self.x_average_rhs.append(self.get_x_average_rhs(density.real, self.t))

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += self.get_k_average(density.real, self.t)

            # save the current time
            self.times.append(self.t)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or function specifying the wave function
        :return: self
        """
        if isinstance(wavefunc, (CPUDispatcher, FunctionType)):
            self.wavefunction[:] = wavefunc(self.x)

        elif isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape, \
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            raise ValueError("wavefunc must be either function or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dx)

        return self