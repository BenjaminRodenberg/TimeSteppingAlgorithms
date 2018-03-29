from __future__ import division
import scipy.sparse as sp
import scipy.sparse.linalg as lin

import abc

import numpy as np
from numpy.polynomial.polynomial import Polynomial


def scale_to_new_window(p, s):
    """
    @:type p Polynomial
    returns a new Polynomial with a window scaled to s*p.windows.
    EXAMPLE:
        If p has the window [0,1] and we use s=2, the window of the returned polynomial is [0,2]
    :param p:
    :param s:
    :return:
    """
    c_scaled = np.array(p.coef)

    # first compute coefficients of scaled polynomial. The scaled polynomial has len(domain) == len(window)
    for i in range(c_scaled.shape[0]):
        c_scaled[i] *= (1.0/s)**i

    return Polynomial(c_scaled, window=s*np.array(p.window), domain=p.domain)


class TimeIntegrationScheme(object):
    __metaclass__ = abc.ABCMeta

    name = "Time Integration Scheme"
    evaluation_times = []  # t values where numerical scheme evaluates the rhs f(t,x)

    def __init__(self, n):
        """
        initialize buffer for rhs values.
        :param n: number of different rhs functions (usually evaluations at different points in time, different BC...)
        """
        # type: () -> object
        self._rhs = n * [None]

    @abc.abstractmethod
    def do_step(self, u0, tau):
        """
        Performs one step with steplength tau of the time integration scheme
        :param u0: initial condition u0 = u(t0)
        :param tau:
        :return:
        """
        return

    def set_rhs(self, A, b, tau=0):
        """
        set rhs of timestepping scheme.
        :param A: Linear part of f(u)
        :param b: Constant part of f(u)
        :param t: optional parameter, if time stepping scheme requires more than one evaluation point
        :return:
        """
        self._rhs[tau] = (A, b)

    def set_all_rhs(self, A, b):
        """
        sets all rhs
        :param A: Linear part of f(u)
        :param b: Constant part of f(u)
        :return:
        """
        for i in range(self._rhs.__len__()):
            self.set_rhs(A, b, i)

    def get_evaluation_point(self, i):
        """
        returns the evaluation point consisting of time and u value, corresponding to the i-th right hand side.
        :param i:
        :return:
        """
        ## todo duplicate of get_sampling_times
        return (self.evaluation_times[i], None)

    def number_of_function_evaluations(self):
        return self._rhs.__len__()


class ContinuousRepresentationScheme(TimeIntegrationScheme):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n, sampling_times):
        self._sampling_times = sampling_times
        super(ContinuousRepresentationScheme, self).__init__(self.stages)  # initialize four empty spots for substepts k_1,2,3,4: https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren

    @abc.abstractmethod
    def get_continuous_representation_for_component(self, component_index, t0, u0, u1, tau):
        return

    def get_sampling_times(self, t0, tau):
        ## todo duplicate of get_evaluation_point
        return t0 + self._sampling_times * tau


class ExplicitEuler(TimeIntegrationScheme):

    name = "Explicit Euler"
    evaluation_times = np.array([0])
    stages = 1

    def __init__(self):
        super(ExplicitEuler, self).__init__(self.stages)  # only initialize one empty spot for f(t0,u0)

    def do_step(self, u0, tau):
        """
        perform one explicit euler step with stepwidth tau and initial condition u0. Solving the ODE
        du/dx = f(u), where f(u)=Au+b is a linear function of u.
        :param u0: initial condition u0 = u(t0)
        :param tau: stepwidth
        :param A: Linear part of f(u)
        :param b: COnstant part of f(u)
        :return: u1 = u(t0+tau)
        """
        A,b = self._rhs[0]  # rhs at t0
        f = lambda u: A*u+b
        u1 = u0 + tau * f(u0)
        return u1


class ImplicitEuler(TimeIntegrationScheme):

    name = "Implicit Euler"
    evaluation_times = np.array([1.0])
    stages = 1

    def __init__(self):
        super(ImplicitEuler, self).__init__(self.stages)  # only initialize one empty spot for f(t1,u1)

    def do_step(self, u0, tau):
        """
        perform one implicit euler step with stepwidth tau and initial condition u0. Solving the ODE
        du/dx = f(u), where f(u)=Au+b is a linear function of u.
        :param u0: initial condition u0 = u(t0)
        :param tau: stepwidth
        :param A: Linear part of f(u)
        :param b: COnstant part of f(u)
        :return: u1 = u(t0+tau)
        """
        A,b = self._rhs[0]  # rhs at t0
        u1 = lin.spsolve(sp.eye(A.shape[0]) - tau*A, tau * b + u0)
        return u1


class RungeKutta4(ContinuousRepresentationScheme):

    name = "Runge Kutta 4"
    evaluation_times = np.array([0, .5, .5, 1])
    stages = 4

    def __init__(self):
        # initialize four empty spots for substeps k_1,2,3,4: https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren
        super(RungeKutta4, self).__init__(self.stages, self.evaluation_times)
        self.k = self.stages*[None]

    def do_step(self, u0, tau):
        """
        perform one explicit rk4 step with stepwidth tau and initial condition u0. Solving the ODE
        du/dx = f(u), where f(u)=Au+b is a linear function of u.

        adapted implementation from https://rosettacode.org/wiki/Runge-Kutta_method#Python
        :param u0: initial condition u0 = u(t0)
        :param tau: stepwidth
        :param A: Linear part of f(u)
        :param b: Constant part of f(u)
        :return: u1 = u(t0+tau)
        """

        def f1(u):
            A1, b1 = self._rhs[0]
            return A1*u+b1

        def f2(u):
            A2, b2 = self._rhs[1]
            return A2*u+b2

        def f3(u):
            A3, b3 = self._rhs[2]
            return A3*u+b3

        def f4(u):
            A4, b4 = self._rhs[3]
            return A4*u+b4

        self.k[0] = f1(u0)  # fully explicity c0 -> c1 = u0 + k1
        self.k[1] = f2(u0 + 0.5 * tau * self.k[0])  # interpolated (c0 + c1)*.5 -> c2 = u0 + k2
        self.k[2] = f3(u0 + 0.5 * tau * self.k[1])  # interpolated (c0 + c2)*.5 -> c3 = u0 + k3
        self.k[3] = f4(u0 + tau * self.k[2])  # extrapolated c3
        u1 = u0 + tau * (self.k[0] + 2*self.k[1] + 2*self.k[2] + self.k[3]) / 6.0
        return u1

    def get_continuous_representation_for_component(self, component_index, t0, u0, u1, tau):
        b = [Polynomial([0, 1.0, -3.0/2.0, +2.0/3.0]),  # theta - 3.0 * theta**2 / 2.0 + 2 * theta**3 / 3.0,
             Polynomial([0, 0, 1.0, -2.0/3.0]),         # theta**2 - 2 * theta**3 / 3.0,
             Polynomial([0, 0, 1.0, -2.0/3.0]),         # theta**2 - 2 * theta**3 / 3.0,
             Polynomial([0, 0, -1.0/2.0, 2.0/3.0])]     # - theta**2 / 2.0 + 2 * theta**3 / 3.0]

        u1_cont = u0[component_index] + tau * np.sum([b[s] * self.k[s][component_index] for s in range(self.stages)])

        # map from [0,1] to [t0, t0+tau]
        u1_cont.window = [0, 1]  # polynomial coefficients are defined on [0, 1]
        u1_cont.domain = [t0, t0+tau]  # polynomial is evaluated at time [t0, t0+tau]

        scaling_factor = (u1_cont.domain[1]-u1_cont.domain[0])/(u1_cont.window[1]-u1_cont.window[0])  # scaling factor
        return scale_to_new_window(u1_cont, scaling_factor)


class RungeKutta2(ContinuousRepresentationScheme):

    name = "Runge Kutta 2"
    evaluation_times = np.array([0, .5])
    stages = 2

    def __init__(self):
        super(RungeKutta2, self).__init__(self.stages, self.evaluation_times)  # initialize two empty spots for substepts k_1,2
        self.up = None

    def do_step(self, u0, tau):

        def f1(u):
            A1, b1 = self._rhs[0]
            return A1*u+b1

        def f2(u):
            A2, b2 = self._rhs[1]
            return A2*u+b2

        k1 = tau * f1(u0)
        self.up = u0 + .5 * k1  # predicted u
        k2 = tau * f2(self.up)
        u1 = u0 + k2
        return u1

    def get_continuous_representation_for_component(self, component_index, t0, u0, u1, tau):

        u1_cont = Polynomial([u0[component_index], u1[component_index]-u0[component_index]])  # linear interpolation Polynomial

        # map from [0,1] to [t0, t0+tau]
        u1_cont.window = [0, 1]  # polynomial coefficients are defined on [0, 1]
        u1_cont.domain = [t0, t0+tau]  # polynomial is evaluated at time [t0, t0+tau]

        scaling_factor = (u1_cont.domain[1]-u1_cont.domain[0])/(u1_cont.window[1]-u1_cont.window[0])  # scaling factor
        return scale_to_new_window(u1_cont, scaling_factor)


class ExplicitHeun(ContinuousRepresentationScheme):

    name = "Explicit Heun"
    evaluation_times = np.array([0, 1])
    stages = 2

    def __init__(self):
        super(ExplicitHeun, self).__init__(self.stages, self.evaluation_times)  # initialize two empty spots for values at t0 and t1
        self._euler_method = ExplicitEuler()
        self.up = None

    def do_step(self, u0, tau):
        """
        perform one explicit heun step with stepwidth tau and initial condition u0. Solving the ODE
        du/dx = f(u), where f(u)=Au+b is a linear function of u.
        :param u0: initial condition u0 = u(t0)
        :param tau: stepwidth
        :param A: Linear part of f(u)
        :param b: COnstant part of f(u)
        :return: u1 = u(t0+tau)
        """

        A0, b0 = self._rhs[0]
        A1, b1 = self._rhs[1]

        # perform predictor step with explicit euler
        self._euler_method.set_rhs(A0, b0)
        self.up = self._euler_method.do_step(u0, tau)  # predicted u

        def f0(u):
            return A0*u+b0

        def f1(u):
            return A1*u+b1

        # perform corrector step with Heun
        u1 = u0 + .5 * tau * (f0(u0) + f1(self.up))
        return u1

    def get_continuous_representation_for_component(self, component_index, t0, u0, u1, tau):

        u1_cont = Polynomial([u0[component_index], u1[component_index]-u0[component_index]])  # linear interpolation Polynomial

        # map from [0,1] to [t0, t0+tau]
        u1_cont.window = [0, 1]  # polynomial coefficients are defined on [0, 1]
        u1_cont.domain = [t0, t0+tau]  # polynomial is evaluated at time [t0, t0+tau]

        scaling_factor = (u1_cont.domain[1]-u1_cont.domain[0])/(u1_cont.window[1]-u1_cont.window[0])  # scaling factor
        return scale_to_new_window(u1_cont, scaling_factor)


class ImplicitTrapezoidalRule(ContinuousRepresentationScheme):

    name = "Implicit Trapezoidal Rule"
    evaluation_times = np.array([0, 1])
    stages = 2

    def __init__(self):
        super(ImplicitTrapezoidalRule, self).__init__(self.stages, self.evaluation_times)  # initialize two empty spots for values at t0 and t1

    def do_step(self, u0, tau):
        """
        perform one implicit trapezoidal rule step with stepwidth tau and initial condition u0. Solving the ODE
        du/dx = f(u), where f(u)=Au+b is a linear function of u.
        :param u0: initial condition u0 = u(t0)
        :param tau: stepwidth
        :return: u1 = u(t0+tau)
        """

        A0, b0 = self._rhs[0]
        A1, b1 = self._rhs[1]

        # u1 = u0 + tau/2 * (f(u0,t0) + f(u1,t1))
        u1 = lin.spsolve(sp.eye(A1.shape[0]) - .5 * tau * A1, u0 + .5 * tau * (b1 + b0 + A0 * u0))
        return u1

    def get_continuous_representation_for_component(self, component_index, t0, u0, u1, tau):

        u1_cont = Polynomial([u0[component_index], u1[component_index]-u0[component_index]])  # linear interpolation Polynomial

        # map from [0,1] to [t0, t0+tau]
        u1_cont.window = [0, 1]  # polynomial coefficients are defined on [0, 1]
        u1_cont.domain = [t0, t0+tau]  # polynomial is evaluated at time [t0, t0+tau]

        scaling_factor = (u1_cont.domain[1]-u1_cont.domain[0])/(u1_cont.window[1]-u1_cont.window[0])  # scaling factor
        return scale_to_new_window(u1_cont, scaling_factor)
