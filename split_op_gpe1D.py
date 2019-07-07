import numpy as np
from scipy import fftpack  # Tools for fourier transform
from scipy import linalg  # Linear algebra for dense matrix
from numba import njit
from numba.targets.registry import CPUDispatcher
from types import FunctionType


class SplitOpGPE1D(object):
    """
    The second-order split-operator propagator of the 1D Schrodinger equation
    in the coordinate representation
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    """

    def __init__(self, *, x_grid_dim, x_amplitude, v, k, dt,
                 epsilon=1e-1, diff_k=None, diff_v=None, t=0, abs_boundary=1., **kwargs):
        """
        :param x_grid_dim: the grid size
        :param x_amplitude: the maximum value of the coordinates
        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)
        :param diff_k: the derivative of the potential energy for the Ehrenfest theorem calculations
        :param diff_v: the derivative of the kinetic energy for the Ehrenfest theorem calculations
        :param t: initial value of time
        :param dt: initial time increment
        :param epsilon: relative error tolerance
        :param abs_boundary: absorbing boundary
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
        self.epsilon = epsilon
        self.abs_boundary = abs_boundary

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        x = self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx
        # The same as
        # self.x = np.linspace(-self.x_amplitude, self.x_amplitude - self.dx , self.x_grid_dim)

        # generate momentum range as it corresponds to FFT frequencies
        p = self.p = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros(self.x.size, dtype=np.complex)

        # allocate an extra copy for the wavefunction necessary for adaptive time step propagation
        self.wavefunction_next = np.zeros_like(self.wavefunction)

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

        # pre-calculate the absorbing potential and the sequence of alternating signs

        abs_boundary = (abs_boundary if isinstance(abs_boundary, (float, int)) else abs_boundary(x))
        abs_boundary = (-1) ** np.arange(self.wavefunction.size) * abs_boundary

        @njit
        def expV(wavefunction, t, dt):
            """
            function to efficiently evaluate
                wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
            """
            wavefunction *= abs_boundary * np.exp(-0.5j * dt * v(x, t + 0.5 * dt))

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

            # Get codes for efficiently calculating the Ehrenfest relations

            @njit
            def get_p_average_rhs(density, t):
                return np.sum(density * diff_v(x, t))

            self.get_p_average_rhs = get_p_average_rhs

            # The code above is equivalent to
            # self.get_p_average_rhs = njit(lambda density, t: np.sum(density * diff_v(x, t)))

            @njit
            def get_v_average(density, t):
                return np.sum(v(x, t) * density)

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

            # Allocate array for storing coordinate or momentum density of the wavefunction
            self.density = np.zeros(self.wavefunction.shape, dtype=np.float)

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

        while self.t < time_final:

            ############################################################################################################
            #
            #   Adaptive scheme propagator
            #
            ############################################################################################################

            # propagate the wavefunction by a single dt
            np.copyto(self.wavefunction_next, self.wavefunction)
            self.wavefunction_next = self.single_step_propagation(self.dt, self.wavefunction_next)

            e_n = linalg.norm(self.wavefunction_next - self.wavefunction) / linalg.norm(self.wavefunction_next)

            while e_n > self.epsilon:
                # the error is to high, decrease the time step and propagate with the new time step

                self.dt *= self.epsilon / e_n

                np.copyto(self.wavefunction_next, self.wavefunction)
                self.wavefunction_next = self.single_step_propagation(self.dt, self.wavefunction_next)

                e_n = linalg.norm(self.wavefunction_next - self.wavefunction) / linalg.norm(self.wavefunction_next)

            # accept the current wave function
            self.wavefunction, self.wavefunction_next = self.wavefunction_next, self.wavefunction

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

            # self.dt *= (e_n_1 / e_n) ** 0.075 * (self.epsilon / e_n) ** 0.175 * (e_n_1 ** 2 / e_n / e_n_2) ** 0.01
            self.dt *= (self.epsilon ** 2 / e_n / e_n_1 * previous_dt / self.dt) ** (1 / 12.)

            # update the error estimates in order to go next to the next step
            e_n_2, e_n_1 = e_n_1, e_n

        # save the error estimates
        self.previous_dt = previous_dt
        self.e_n = e_n
        self.e_n_1 = e_n_1
        self.e_n_2 = e_n_2

        return self.wavefunction

    def single_step_propagation(self, dt, wavefunction):
        """
        Propagate the wavefunction by a single time-step
        :param dt: time-step
        :param wavefunction: 1D numpy array
        :return: wavefunction
        """
        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(wavefunction, self.t, dt)

        # going to the momentum representation
        wavefunction = fftpack.fft(wavefunction, overwrite_x=True)

        # efficiently evaluate
        #   wavefunction *= exp(-1j * dt * k)
        self.expK(wavefunction, self.t, dt)

        # going back to the coordinate representation
        wavefunction = fftpack.ifft(wavefunction, overwrite_x=True)

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(wavefunction, self.t, dt)

        # normalize
        # this line is equivalent to
        # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction) ** 2 ) * self.dx)
        wavefunction /= linalg.norm(wavefunction) * np.sqrt(self.dx)

        return wavefunction

    def get_ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.is_ehrenfest:
            # evaluate the coordinate density
            np.abs(self.wavefunction, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of <x>
            self.x_average.append(self.get_x_average(self.density))

            self.p_average_rhs.append(-self.get_p_average_rhs(self.density, self.t))

            # save the potential energy
            self.hamiltonian_average.append(self.get_v_average(self.density, self.t))

            # calculate density in the momentum representation
            wavefunction_p = fftpack.fft(self.minus * self.wavefunction, overwrite_x=True)

            # get the density in the momentum space
            np.abs(wavefunction_p, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of <p>
            self.p_average.append(self.get_p_average(self.density))

            self.x_average_rhs.append(self.get_x_average_rhs(self.density, self.t))

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += self.get_k_average(self.density, self.t)

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