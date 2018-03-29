import numpy as np
from scipy.interpolate import PPoly


from space_discretization import second_derivative_matrix, compute_finite_difference_scheme_coeffs
from time_integration import ImplicitTrapezoidalRule, ExplicitHeun, TimeIntegrationScheme, RungeKutta4, ContinuousRepresentationScheme
import finite_difference_schemes

from domain import Domain

import numeric_parameters

import abc


def estimate_coupling_neumann_BC(left_domain, u_left, right_domain, u_right):
    """
    @type left_domain Domain
    @type right_domain Domain
    """
    # set neumann BC at coupling interface
    if numeric_parameters.neumann_coupling_scheme == 'forward':
        u_neumann_coupled__ = finite_difference_schemes.one_sided_forward_FD_at(0, right_domain.h, u_right, order=numeric_parameters.neumann_coupling_order)
    elif numeric_parameters.neumann_coupling_scheme == 'central':
        if abs(left_domain.grid.h - right_domain.grid.h) < 10**-10: # use central finite differences
            u_glued = np.array(u_left.tolist()[0:-1] + u_right.tolist())
            u_neumann_coupled__ = finite_difference_schemes.central_FD_at(left_domain.u.shape[0]-1, right_domain.grid.h, u_glued, order=numeric_parameters.neumann_coupling_order)
        else: # use modified operator for non-identical mesh size
            if numeric_parameters.neumann_coupling_order != 2:
                print "Operator of order %d is not implemented!!!" % numeric_parameters.neumann_coupling_order
                quit()
            fraction = left_domain.grid.h / right_domain.grid.h  # normalize to right domain's meshwidth
            p = np.array([-fraction, 0, 1.0])
            c = compute_finite_difference_scheme_coeffs(evaluation_points=p, derivative_order=1)
            #assert abs(u_right[0] - u_left[-1]) < 10**-5
            u = np.array([u_left[-2], u_right[0], u_right[1]])
            u_neumann_coupled__ = 1.0/right_domain.grid.h * (u.dot(c))
    else:
        print "not implemented schemes for coupling Neumann BC demanded!"
        quit()

    return u_neumann_coupled__


class CouplingScheme(object):
    __metaclass__ = abc.ABCMeta

    name = "Coupling Scheme"

    def __init__(self):
        # type: () -> object
        return

    @abc.abstractmethod
    def perform(self, t0, tau, left_domain, right_domain):
        return


class FullyImplicitCoupling(CouplingScheme):

    name = "Fully Implicit Coupling"

    def __init__(self):
        super(FullyImplicitCoupling, self).__init__()

    def perform(self, t0, tau, left_domain, right_domain):
        """
        @type left_domain Domain
        @type right_domain Domain
        :param t0:
        """

        residual = np.inf
        tol = numeric_parameters.fixed_point_tol
        n_steps_max = numeric_parameters.n_max_fixed_point_iterations

        n_steps = 0

        # use boundary conditions of t_n-1 as initial guess for t_n
        u_neumann_coupled = left_domain.right_BC["neumann"]
        u_dirichlet_coupled = right_domain.left_BC["dirichlet"]

        # enforce boundary conditions
        left_domain.u[0] = left_domain.left_BC["dirichlet"]
        right_domain.u[-1] = right_domain.right_BC["dirichlet"]

        # start fixed point iteration for determining boundary conditions for t_n
        while abs(residual) > tol and n_steps < n_steps_max:
            # operator for left participant
            # f(u,t_n) with boundary conditions from this timestep
            A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_coupled)

            # use most recent coupling variables for all
            left_domain.time_integration_scheme.set_all_rhs(A_left, R_left)

            # time stepping
            u_left_new = left_domain.time_integration_scheme.do_step(left_domain.u, tau)

            # update dirichlet BC at coupling interface
            u_dirichlet_coupled = u_left_new[-1]

            # operator for right participant
            A_right, R_right = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_coupled, dirichlet_r=right_domain.right_BC["dirichlet"])

             # use most recent coupling variables for all
            right_domain.time_integration_scheme.set_all_rhs(A_right, R_right)

            # time stepping
            u_right_new = right_domain.time_integration_scheme.do_step(right_domain.u, tau)  # only use most recent coupling variables for implicit part of time stepping -> semi implicit
            # set dirichlet BC at coupling interface
            u_right_new[0] = u_dirichlet_coupled

            u_neumann_coupled__ = estimate_coupling_neumann_BC(left_domain, u_left_new, right_domain, u_right_new)

            residual = u_neumann_coupled__ - u_neumann_coupled

            # Aitken's Underrelaxation
            omega = .5  # todo just a random number currently
            u_neumann_coupled += omega * residual
            n_steps += 1

        if n_steps == n_steps_max:
            print "maximum number of steps exceeded!"
            return False

        # update solution
        left_domain.update_u(u_left_new)
        right_domain.update_u(u_right_new)
        # update coupling variables
        left_domain.right_BC["neumann"] = u_neumann_coupled
        right_domain.left_BC["dirichlet"] = u_dirichlet_coupled
        return True


class FullyExplicitCoupling(CouplingScheme):

    name = "Fully Explicit Coupling"

    def __init__(self):
        super(FullyExplicitCoupling, self).__init__()

    def perform(self, t0, tau, left_domain, right_domain):
        """
        @type left_domain Domain
        @type right_domain Domain
        :param t0:
        """

        # enforce boundary conditions
        left_domain.u[0] = left_domain.left_BC["dirichlet"]
        right_domain.u[-1] = right_domain.right_BC["dirichlet"]

        # get coupling boundary conditions for left participant
        u_neumann_coupled = left_domain.right_BC["neumann"]

        # operator for left participant
        # f(u,t_n) with boundary conditions from this timestep
        A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_coupled)

        # always use most recent coupling variables for all substeps -> fully explicit
        left_domain.time_integration_scheme.set_all_rhs(A_left, R_left)

        # time stepping
        u_left = left_domain.time_integration_scheme.do_step(left_domain.u, tau)

        # get coupling boundary conditions for right participant
        u_dirichlet_coupled = u_left[-1]

        # operator for right participant
        A_right, R_right = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_coupled, dirichlet_r=right_domain.right_BC["dirichlet"])

        # always use most recent coupling variables for all substeps -> fully explicit
        right_domain.time_integration_scheme.set_all_rhs(A_right, R_right)

        # time stepping
        u_right = right_domain.time_integration_scheme.do_step(right_domain.u, tau)

        # set dirichlet BC at coupling interface
        u_right[0] = u_dirichlet_coupled

        u_neumann_coupled__ = estimate_coupling_neumann_BC(left_domain, u_left, right_domain, u_right)

        residual = u_neumann_coupled__ - left_domain.right_BC["neumann"]

        # Aitken's Underrelaxation
        omega = .5 # todo just a random number currently
        u_neumann_coupled = left_domain.right_BC["neumann"] + omega * residual

        left_domain.update_u(u_left)
        right_domain.update_u(u_right)
        # update coupling variables
        left_domain.right_BC["neumann"] = u_neumann_coupled
        right_domain.left_BC["dirichlet"] = u_dirichlet_coupled
        return True


class WaveformCoupling(CouplingScheme):

    name = "Waveform Coupling"

    def __init__(self, n_left=1, n_right=1):
        if n_left != n_right:
            self.name_suffix = "inhom ("+str(n_left)+"-"+str(n_right)+")"
        self.n_substeps_left = n_left
        self.n_substeps_right = n_right
        super(WaveformCoupling, self).__init__()

    def perform(self, t0, tau, left_domain, right_domain):
        """
        @type left_domain Domain
        @type right_domain Domain
        :param t0:
        """

        assert issubclass(type(left_domain.time_integration_scheme), ContinuousRepresentationScheme)
        assert issubclass(type(right_domain.time_integration_scheme), ContinuousRepresentationScheme)

        # use boundary conditions of t_n-1 as initial guess for t_n
        u_neumann_continuous = lambda tt: left_domain.right_BC["neumann"] * np.ones_like(tt)

        # enforce boundary conditions
        left_domain.u[0] = left_domain.left_BC["dirichlet"]
        right_domain.u[-1] = right_domain.right_BC["dirichlet"]

        t1 = t0+tau
        # do fixed number of sweeps
        for window_sweep in range(5):
            # subcycling parameters
            max_approximation_order = 5

            # operator for left participant
            t_sub, tau_sub = np.linspace(t0, t1, self.n_substeps_left + 1, retstep=True)
            u0_sub = left_domain.u

            coeffs_m1 = np.zeros([max_approximation_order + 1, self.n_substeps_left])
            coeffs_m2 = np.zeros([max_approximation_order + 1, self.n_substeps_left])
            for ii in range(self.n_substeps_left):
                t0_sub = t_sub[ii]
                sampling_times_substep = left_domain.time_integration_scheme.get_sampling_times(t0_sub, tau_sub)
                for i in range(sampling_times_substep.shape[0]):
                    # f(u,t_n) with boundary conditions from this timestep
                    A, R = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_continuous(sampling_times_substep[i]))
                    # use most recent coupling variables for all
                    left_domain.time_integration_scheme.set_rhs(A, R, i)

                # time stepping
                u1_sub = left_domain.time_integration_scheme.do_step(u0_sub, tau_sub)

                # do time continuous reconstruction of Nodal values
                u_dirichlet_continuous_sub_m1 = left_domain.time_integration_scheme.get_continuous_representation_for_component(
                    -1, t0_sub, u0_sub, u1_sub, tau_sub)
                u_dirichlet_continuous_sub_m2 = left_domain.time_integration_scheme.get_continuous_representation_for_component(
                    -2, t0_sub, u0_sub, u1_sub, tau_sub)
                coeffs_m1[:u_dirichlet_continuous_sub_m1.coef.shape[0], ii] = u_dirichlet_continuous_sub_m1.coef
                coeffs_m2[:u_dirichlet_continuous_sub_m2.coef.shape[0], ii] = u_dirichlet_continuous_sub_m2.coef
                u0_sub = u1_sub

            if self.n_substeps_left == 1:
                u_dirichlet_continuous_m1 = u_dirichlet_continuous_sub_m1
                u_dirichlet_continuous_m2 = u_dirichlet_continuous_sub_m2
            else:
                u_dirichlet_continuous_m1 = PPoly(coeffs_m1[::-1,:], t_sub)  # we have to reverse the order of the coefficients for PPoly
                u_dirichlet_continuous_m2 = PPoly(coeffs_m2[::-1,:], t_sub)  # we have to reverse the order of the coefficients for PPoly

            u_left_new = u1_sub  # use result of last subcycle for result of window

            # operator for right participant
            t_sub, tau_sub = np.linspace(t0, t1, self.n_substeps_right + 1, retstep=True)
            u0_sub = right_domain.u

            coeffs_p1 = np.zeros([max_approximation_order + 1, self.n_substeps_right])
            for ii in range(self.n_substeps_right):
                t0_sub = t_sub[ii]
                sampling_times_substep = right_domain.time_integration_scheme.get_sampling_times(t0_sub, tau_sub)
                for i in range(sampling_times_substep.shape[0]):
                    # f(u,t_n) with boundary conditions from this timestep
                    A, R = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_continuous_m1(sampling_times_substep[i]), dirichlet_r=right_domain.right_BC["dirichlet"])
                    # use most recent coupling variables for all
                    right_domain.time_integration_scheme.set_rhs(A, R, i)

                # time stepping
                u1_sub = right_domain.time_integration_scheme.do_step(u0_sub, tau_sub)
                u_dirichlet_continuous_sub_p1 = right_domain.time_integration_scheme.get_continuous_representation_for_component(
                    1, t0_sub, u0_sub, u1_sub, tau_sub)
                u1_sub[0] = u_dirichlet_continuous_m1(t0_sub+tau_sub)  # we have to set the (known and changing) dirichlet value manually, since this value is not changed by the timestepping
                coeffs_p1[:u_dirichlet_continuous_sub_p1.coef.shape[0], ii] = u_dirichlet_continuous_sub_p1.coef
                u0_sub = u1_sub

            if self.n_substeps_right == 1:
                u_dirichlet_continuous_p1 = u_dirichlet_continuous_sub_p1
            else:
                u_dirichlet_continuous_p1 = PPoly(coeffs_p1[::-1,:], t_sub)  # we have to reverse the order of the coefficients for PPoly

            u_right_new = u1_sub  # use result of last subcycle for result of window
            u_right_new[0] = u_dirichlet_continuous_m1(t0+tau)  # we have to set the (known and changing) dirichlet value manually, since this value is not changed by the timestepping

            if numeric_parameters.neumann_coupling_order != 2:
                print "Operator of order %d is not implemented!!!" % numeric_parameters.neumann_coupling_order
                quit()

            fraction = left_domain.grid.h / right_domain.grid.h  # normalize to right domain's meshwidth
            p = np.array([-fraction, 0, 1.0])
            c = compute_finite_difference_scheme_coeffs(evaluation_points=p, derivative_order=1)
            # for u_stencil[1] we have to use the left_domain's continuous representation, because the right_domain's
            # representation is constant in time. This degrades the order to 1 for a irregular mesh.
            u_stencil = [
                u_dirichlet_continuous_m2,
                u_dirichlet_continuous_m1,
                u_dirichlet_continuous_p1
            ]
            # compute continuous representation for Neumann BC
            u_neumann_continuous = lambda x: 1.0/right_domain.grid.h * (u_stencil[0](x) * c[0] + u_stencil[1](x) * c[1] + u_stencil[2](x) * c[2])
        # update solution
        left_domain.update_u(u_left_new)
        right_domain.update_u(u_right_new)
        # update coupling variables
        left_domain.right_BC["neumann"] = u_neumann_continuous(t1)
        right_domain.left_BC["dirichlet"] = u_dirichlet_continuous_m1(t1)
        return True


class ExplicitPredictorCoupling(CouplingScheme):

    name = "Explicit Predictor Coupling"

    def __init__(self):
        super(ExplicitPredictorCoupling, self).__init__()

    def perform(self, t0, tau, left_domain, right_domain):
        """
        @type left_domain Domain
        @type right_domain Domain
        :param t0:
        """

        # enforce boundary conditions
        left_domain.u[0] = left_domain.left_BC["dirichlet"]
        right_domain.u[-1] = right_domain.right_BC["dirichlet"]

        # get coupling boundary conditions for left participant
        u_neumann_coupled = left_domain.right_BC["neumann"]
        u_neumann_coupled_predicted = u_neumann_coupled  # just initialize

        u_dirichlet_coupled = left_domain.u[-1]

        for i in range(2):

            # operator for left participant
            # f(u,t_n) with boundary conditions from this timestep
            A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_coupled)
            A_left_predicted, R_left_predicted = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_coupled_predicted)

            # use explicit coupling variables and predicted ones
            left_domain.time_integration_scheme.set_rhs(A_left, R_left, 0)
            left_domain.time_integration_scheme.set_rhs(A_left_predicted, R_left_predicted, 1)

            # time stepping
            u_left = left_domain.time_integration_scheme.do_step(left_domain.u, tau)
            u_left_predicted = left_domain.time_integration_scheme.up

            # get coupling boundary conditions for right participant
            # u_dirichlet_coupled = u_left[-1]
            u_dirichlet_coupled_predicted = u_left_predicted[-1]

            # operator for right participant
            A_right, R_right = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_coupled, dirichlet_r=right_domain.right_BC["dirichlet"])
            A_right_predicted, R_right_predicted = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_coupled_predicted, dirichlet_r=right_domain.right_BC["dirichlet"])

            # use explicit coupling variables and predicted ones
            right_domain.time_integration_scheme.set_rhs(A_right, R_right, 0)
            right_domain.time_integration_scheme.set_rhs(A_right_predicted, R_right_predicted, 1)

            # time stepping
            u_right = right_domain.time_integration_scheme.do_step(right_domain.u, tau)
            u_right_predicted = right_domain.time_integration_scheme.up

            # set dirichlet BC at coupling interface
            u_right[0] = u_dirichlet_coupled
            u_right_predicted[0] = u_dirichlet_coupled_predicted

            # set neumann BC at coupling interface
            u_neumann_coupled_predicted = estimate_coupling_neumann_BC(left_domain, u_left_predicted, right_domain, u_right_predicted)  # computed with finite differences

        u_dirichlet_coupled = u_left[-1]
        u_right[0] = u_dirichlet_coupled
        left_domain.update_u(u_left)
        right_domain.update_u(u_right)

        left_domain.right_BC["neumann"] = u_neumann_coupled
        right_domain.left_BC["dirichlet"] = u_dirichlet_coupled

        return True


class SemiImplicitExplicitCoupling(CouplingScheme):

    name = "Semi Implicit Explicit Coupling"

    def __init__(self):
        super(SemiImplicitExplicitCoupling, self).__init__()

    def perform(self, t0, tau, left_domain, right_domain):
        """
        @type left_domain Domain
        @type right_domain Domain
        :param t0:
        """

        residual = np.inf
        tol = numeric_parameters.fixed_point_tol
        n_steps_max = numeric_parameters.n_max_fixed_point_iterations

        n_steps = 0

        # f(u,t_n-1) with boundary conditions from last timestep
        A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=left_domain.right_BC["neumann"])
        # f(v,t_n-1)
        A_right, R_right = second_derivative_matrix(right_domain.grid, dirichlet_l=right_domain.left_BC["dirichlet"], dirichlet_r=right_domain.right_BC["dirichlet"])

        # use boundary conditions of t_n-1 as initial guess for t_n
        u_neumann_coupled = left_domain.right_BC["neumann"]
        u_dirichlet_coupled = right_domain.left_BC["dirichlet"]

        # enforce boundary conditions
        left_domain.u[0] = left_domain.left_BC["dirichlet"]
        right_domain.u[-1] = right_domain.right_BC["dirichlet"]

        # set rhs at t0 constant for all fixed point iterations
        left_domain.time_integration_scheme.set_rhs(A_left, R_left, 0)
        right_domain.time_integration_scheme.set_rhs(A_right, R_right, 0)

        # start fixed point iteration for determining boundary conditions for t_n
        while abs(residual) > tol and n_steps < n_steps_max:
            # LEFT
            for i in range(left_domain.time_integration_scheme.evaluation_times.shape[0]):
                # operator for left participant
                evaluation_time = left_domain.time_integration_scheme.evaluation_times[i]
                u_neumann_interpolated = (1-evaluation_time) * left_domain.right_BC["neumann"] + evaluation_time * u_neumann_coupled
                A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_interpolated)
                # use most recent coupling variables for all
                left_domain.time_integration_scheme.set_rhs(A_left, R_left, i)

            # time stepping
            u_left_new = left_domain.time_integration_scheme.do_step(left_domain.u, tau)

            # update dirichlet BC at coupling interface
            u_dirichlet_coupled = u_left_new[-1]

            # RIGHT
            for i in range(right_domain.time_integration_scheme.evaluation_times.shape[0]):
                # operator for right participant
                evaluation_time = right_domain.time_integration_scheme.evaluation_times[i]
                u_dirichlet_interpolated = (1-evaluation_time) * right_domain.left_BC["dirichlet"] + evaluation_time * u_dirichlet_coupled
                A_right, R_right = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_interpolated, dirichlet_r=right_domain.right_BC["dirichlet"])
                # use most recent coupling variables for all
                right_domain.time_integration_scheme.set_rhs(A_right, R_right, i)

            # time stepping
            u_right_new = right_domain.time_integration_scheme.do_step(right_domain.u, tau)
            # set dirichlet BC at coupling interface
            u_right_new[0] = u_dirichlet_coupled

            u_neumann_coupled__ = estimate_coupling_neumann_BC(left_domain, u_left_new, right_domain, u_right_new)

            residual = u_neumann_coupled__ - u_neumann_coupled

            # Aitken's Underrelaxation
            omega = .5  # todo just a random number currently
            u_neumann_coupled += omega * residual
            n_steps += 1

        if n_steps == n_steps_max:
            print "maximum number of steps exceeded!"
            return False

        # update solution
        left_domain.update_u(u_left_new)
        right_domain.update_u(u_right_new)
        # update coupling variables
        left_domain.right_BC["neumann"] = u_neumann_coupled
        right_domain.left_BC["dirichlet"] = u_dirichlet_coupled
        return True


class StrangSplittingCoupling(CouplingScheme):

    name = "Strang Splitting Coupling"

    def __init__(self):
        super(StrangSplittingCoupling, self).__init__()

    def perform(self, t0, tau, left_domain, right_domain):
        """
        uses Strang splitting for explicit coupling
        @type left_domain Domain
        @type right_domain Domain
        :param t0:
        :param tau:
        :param left_domain:
        :param right_domain:
        :return:
        """
        """
        @type left_domain Domain
        @type right_domain Domain
        """

        # enforce boundary conditions
        left_domain.u[0] = left_domain.left_BC["dirichlet"]
        right_domain.u[-1] = right_domain.right_BC["dirichlet"]

        # get coupling boundary conditions for left participant
        u_neumann_coupled = left_domain.right_BC["neumann"]

        # operator for left participant
        # f(u,t_n) with boundary conditions from this timestep
        A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_coupled)

        # always use most recent coupling variables for all substeps -> fully explicit
        left_domain.time_integration_scheme.set_all_rhs(A_left, R_left)

        # time stepping -> only perform a half step f_1, STRANG SPLITTING APPROACH
        u_left_mid = left_domain.time_integration_scheme.do_step(left_domain.u, .5 * tau)

        # get coupling boundary conditions for right participant
        u_dirichlet_coupled = u_left_mid[-1]

        # operator for right participant
        A_right, R_right = second_derivative_matrix(right_domain.grid, dirichlet_l=u_dirichlet_coupled, dirichlet_r=right_domain.right_BC["dirichlet"])

        # always use most recent coupling variables for all substeps -> fully explicit
        right_domain.time_integration_scheme.set_all_rhs(A_right, R_right)

        # time stepping -> full step f_2, STRANG SPLITTING APPROACH
        u_right = right_domain.time_integration_scheme.do_step(right_domain.u, tau)

        # set dirichlet BC at coupling interface
        u_right[0] = u_dirichlet_coupled
        right_domain.u[0] = u_dirichlet_coupled  # the new dirichlet boundary condition has to be enforced for all times!

        # get coupling boundary conditions for left participant
        u_neumann_coupled = estimate_coupling_neumann_BC(left_domain, u_left_mid, right_domain, .5*(right_domain.u+u_right))

        # updated operator for left participant
        # f(u,t_n) with boundary conditions from STRANG SPLITTING STEP
        A_left, R_left = second_derivative_matrix(left_domain.grid, dirichlet_l=left_domain.left_BC["dirichlet"], neumann_r=u_neumann_coupled)

        # always use most recent coupling variables for all substeps -> fully explicit
        left_domain.time_integration_scheme.set_all_rhs(A_left, R_left)

        # time stepping -> do second half step f_1, STRANG SPLITTING APPROACH
        u_left = left_domain.time_integration_scheme.do_step(u_left_mid, .5 * tau)

        u_dirichlet_coupled = u_left[-1]
        u_right[0] = u_dirichlet_coupled

        # update u
        left_domain.update_u(u_left)
        right_domain.update_u(u_right)

        # update coupling variables
        left_domain.right_BC["neumann"] = u_neumann_coupled
        right_domain.left_BC["dirichlet"] = u_dirichlet_coupled
        return True


class MonolithicScheme(CouplingScheme):
    # todo this class hierarchy stinks!

    name = "Monolithic Approach"

    def __init__(self):
        super(MonolithicScheme, self).__init__()

    def perform(self, t0, tau, domain, dummy):
        """
        @type domain Domain
        :param t0:
        """
        A, R = second_derivative_matrix(domain.grid, dirichlet_l=domain.left_BC['dirichlet'], dirichlet_r=domain.right_BC['dirichlet'])
        domain.time_integration_scheme.set_all_rhs(A, R)
        u = domain.time_integration_scheme.do_step(domain.u, tau)
        domain.update_u(u)
        return True


