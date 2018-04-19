# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from domain import RegularGrid, IrregularGrid, Domain
from experiment_documentation import Experiment
import coupling_schemes
import time_integration
from coupling_schemes import CouplingScheme
from time_integration import TimeIntegrationScheme
import numeric_parameters
import datetime


def initialize_monolithic(monolithic_domain):
    # initialize
    monolithic_domain.update_u(f(monolithic_domain.grid.x))  # initial condition in domain

    # Dirichlet boundary conditions
    monolithic_domain.left_BC["dirichlet"] = f(monolithic_domain.grid.x_left)
    monolithic_domain.right_BC["dirichlet"] = f(monolithic_domain.grid.x_right)


def initialize_domains_and_boundary_conditions(left_domain, right_domain):
    # initialize
    left_domain.update_u(f(left_domain.grid.x))  # initial condition in left domain
    right_domain.update_u(f(right_domain.grid.x))  # initial condition in right domain

    # real and coupling boundary conditions (dirichlet/neumann coupling)
    left_domain.left_BC["dirichlet"] = f(left_domain.grid.x_left)  # Dirichlet boundary conditions
    left_domain.right_BC["neumann"] = coupling_schemes.estimate_coupling_neumann_BC(left_domain, left_domain.u, right_domain, right_domain.u)
    right_domain.left_BC["dirichlet"] = left_domain.u[-1]  # dirichlet coupling in right domain
    right_domain.right_BC["dirichlet"] = f(right_domain.grid.x_right)  # Dirichlet boundary conditions


def experimental_series(taus, u_ref_left, u_ref_right, experiment):
    """
    @type taus list
    """
    print("EXPERIMENTAL SERIES")

    errors_left = []
    errors_right = []
    experiment_counter = 1
    n_experiments = taus.__len__()

    for tau in taus:
        print("----")
        print("%i of %i" % (experiment_counter, n_experiments))
        print("tau = %f" % tau)

        u_left, u_right = experiment(tau)

        # compute error
        e_tot_left = np.sum(np.abs(u_left-u_ref_left)) / u_left.shape[0]
        e_tot_right = np.sum(np.abs(u_right-u_ref_right)) / (u_right.shape[0])

        print("Error total = %.2e | %.2e" % (e_tot_left, e_tot_right))

        experiment_counter += 1
        errors_left += [e_tot_left]
        errors_right += [e_tot_right]

    print("----")
    np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')}) # print in scientific notation
    print(repr(np.array(errors_left)))
    print(repr(np.array(errors_right)))
    print(taus)
    print()

    return errors_left, errors_right


def solve_coupled_problem(tau, time_stepping_scheme, coupling_scheme, left_domain, right_domain):
    """
    :param tau:
    :param time_stepping_scheme: @type TimeIntegrationScheme
    :param coupling_scheme: @type coupling_scheme CouplingScheme
    :param left_domain: @type Domain
    :param right_domain: @type Domain
    :return:
    """

    initialize_domains_and_boundary_conditions(left_domain, right_domain)

    # initialize time integration schemes
    left_domain.time_integration_scheme = time_stepping_scheme()
    right_domain.time_integration_scheme = time_stepping_scheme()

    for t in np.arange(0, T, tau):
        success = coupling_scheme.perform(t, tau, left_domain, right_domain)
        if not success:
            return np.ones_like(left_domain.u)*np.inf, np.ones_like(right_domain.u)*np.inf

    return left_domain.u, right_domain.u


def solve_inhomogeneously_coupled_problem(tau, time_stepping_schemes, coupling_scheme, domains):
    """
    :param tau:
    :param coupling_scheme: @type coupling_scheme CouplingScheme
    :param time_stepping_schemes: @type list(TimeIntegrationScheme) time_stepping_schemes corresponding to domains
    :param domains: @type list(Domain) currently only two coupled domains are supported!
    :return:
    """

    initialize_domains_and_boundary_conditions(domains[0], domains[1])

    # initialize time integration schemes
    for domain, time_stepping_scheme in zip(domains, time_stepping_schemes):
        domain.time_integration_scheme = time_stepping_scheme()

    for t in np.arange(0, T, tau):
        coupling_scheme.perform(t, tau, domains[0], domains[1])

    return domains[0].u, domains[1].u


def solve_monolithic_problem(tau, time_stepping_scheme, domain):
    """
    :param tau:
    :param time_stepping_scheme: @type time_integration.TimeIntegrationScheme
    :param domain: @type Domain
    :return:
    """

    initialize_monolithic(domain)

    # initialize time integration schemes
    domain.time_integration_scheme = time_stepping_scheme()

    coupling_scheme = coupling_schemes.MonolithicScheme()
    for t in np.arange(0, T, tau):
        coupling_scheme.perform(t, tau, domain, None)

    # left part of monolithic solution
    w_left = domain.u[:left_domain.u.shape[0]]
    # right part of monolithic solution
    w_right = domain.u[left_domain.u.shape[0]-1:]

    return w_left, w_right


def compute_reference_solution(domain, tau, time_integration_scheme = time_integration.ImplicitTrapezoidalRule()):
    # compute monolithic at constant spatial and very fine temporal resolution

    print("REFERENCE SOLUTION for grid "+str(domain.grid.x))

    return solve_monolithic_problem(tau, time_integration_scheme, domain)


def coupling_experiment(coupling_scheme, time_stepping_schemes, left_domain, right_domain, u_ref_left, u_ref_right, experiment_timesteps):
    """
    performs a convergence study experiment with coupled domains using different time stepping schemes
    :param coupling_scheme: coupling scheme to be used to couple left_domain and right_domain
    :type coupling_scheme: CouplingScheme
    :param time_stepping_schemes: time stepping scheme(s) used. If two elements are provided, different timestepping schemes are used left and right
    :type time_stepping_schemes: list(time_integration.TimeIntegrationScheme)
    :param left_domain: left domain, where the problem is solved
    :param right_domain: right domain, where the problem is solved
    :param u_ref_left: reference solution on left domain
    :param u_ref_right: reference solution on right domain
    :param experiment_timesteps: range of timestep sizes considered for the convergence study
    :return:
    """
    if type(time_stepping_schemes) is list:
        experiment_name = time_stepping_schemes[0].name + " - " + time_stepping_schemes[1].name + " - " + coupling_scheme.name
        print(experiment_name)
        experiment_errors_left, experimental_errors_right = experimental_series(experiment_timesteps, u_ref_left, u_ref_right, lambda tau: solve_inhomogeneously_coupled_problem(
            tau, time_stepping_schemes, coupling_scheme, [left_domain, right_domain]))
    else:
        experiment_name = time_stepping_schemes.name + " - " + coupling_scheme.name
        try:
            experiment_name += " - "+coupling_scheme.name_suffix
        except AttributeError:
            pass
        print(experiment_name)
        experiment_errors_left, experimental_errors_right = experimental_series(experiment_timesteps, u_ref_left, u_ref_right, lambda tau: solve_coupled_problem(tau, time_stepping_schemes, coupling_scheme, left_domain, right_domain))

    return Experiment(experiment_name, [left_domain.grid.h, right_domain.grid.h], experiment_timesteps, experiment_errors_left, experimental_errors_right, additional_numerical_parameters=numeric_parameters.numeric_parameters_dict)


def monolithic_experiment(time_stepping_scheme, monolithic_domain, u_ref_left, u_ref_right, experiment_timesteps):
    """
    :param coupling_scheme: @type coupling_schemes.CouplingScheme
    :param time_stepping_scheme: time stepping scheme used
    :type time_stepping_scheme: time_integration.TimeIntegrationScheme
    :param monolithic_domain: domain, where the problem is solved
    :param u_ref_left: reference solution on left domain
    :param u_ref_right: reference solution on right domain
    :param experiment_timesteps: range of timestep sizes considered for the convergence study
    :return:
    """
    experiment_name = time_stepping_scheme.name + " - " + coupling_schemes.MonolithicScheme().name
    print(experiment_name)
    experiment_errors_left, experimental_errors_right = experimental_series(experiment_timesteps, u_ref_left, u_ref_right, lambda tau: solve_monolithic_problem(tau, time_stepping_scheme, monolithic_domain))

    if type(monolithic_domain.grid) is IrregularGrid:
        distances = monolithic_domain.grid.x[1:] - monolithic_domain.grid.x[:-1]
        h = str(np.unique(np.round(distances,3)))
    elif type(monolithic_domain.grid) is RegularGrid:
        h = monolithic_domain.grid.h
    return Experiment(experiment_name, h, experiment_timesteps, experiment_errors_left, experimental_errors_right, additional_numerical_parameters=numeric_parameters.numeric_parameters_dict)


def save_experiments(experimental_series, prefix):
    exp_path = './'+prefix+'_'+datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S_%f")

    for experiment in experimental_series:
        experiment.save(exp_path)

# definition of initial condition
f = lambda x: -1.0 * (1-x) * (0-x)  # f= x-x^2 -> F = 1/2*x^2 - 1/3*x^3, f'=1-2*x, f'(0)=1, f'(1)=-1

x0 = 0.0  # left boundary
x1 = 1.0  # coupling interface
x2 = 2.0  # right boundary

# convergence in time: setup
T = 1  # maximum time
tau0 = T * 0.25  # timestep init
N_tau = 10 # we run experiments for all tau = tau0*(.5)**(0...N_tau)
tau_ref = tau0 * 0.5 ** (N_tau+1)

experiment_timesteps = [tau0 * 0.5 ** n for n in range(N_tau)]

# spatial discretization: identical grid left and right
N_gridpoints_left = 6
N_gridpoints_right_coarse = (N_gridpoints_left - 1) + 1  # identical grid resolution like on the left
N_gridpoints_right_fine = (N_gridpoints_left - 1) * 4 + 1  # finer grid resolution like on the left

left_grid = RegularGrid(N_gridpoints=N_gridpoints_left, x_left=x0, x_right=x1)
left_domain = Domain(left_grid)
right_grid_coarse = RegularGrid(N_gridpoints=N_gridpoints_right_coarse, x_left=x1, x_right=x2)
right_domain_coarse = Domain(right_grid_coarse)
right_grid_fine = RegularGrid(N_gridpoints=N_gridpoints_right_fine, x_left=x1, x_right=x2)
right_domain_fine = Domain(right_grid_fine)

assert left_domain.grid.h == right_domain_coarse.grid.h
N_gridpoints_monolithic_regular = N_gridpoints_left * 2 - 1 # remove overlapping point
combined_grid_regular = RegularGrid(N_gridpoints=N_gridpoints_monolithic_regular, x_left=x0, x_right=x2)
monolithic_domain_regular = Domain(combined_grid_regular)

assert left_domain.grid.h == 4*right_domain_fine.grid.h
combined_grid_nonregular = IrregularGrid(x=np.concatenate([left_domain.grid.x[:-1], right_domain_fine.grid.x]))
monolithic_domain_nonregular = Domain(combined_grid_nonregular)

# compute monolithic reference solutions
u_ref_left_regular, u_ref_right_regular = compute_reference_solution(monolithic_domain_regular, tau_ref, time_integration_scheme=time_integration.RungeKutta4)
u_ref_left_nonregular, u_ref_right_nonregular = compute_reference_solution(monolithic_domain_nonregular, tau_ref, time_integration_scheme=time_integration.RungeKutta4)

print("### experimental series 1: order degradation with classical schemes on regular domain")

experiments = list()
experiments.append(monolithic_experiment(time_integration.ExplicitHeun, monolithic_domain_regular, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(monolithic_experiment(time_integration.ImplicitTrapezoidalRule, monolithic_domain_regular, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(monolithic_experiment(time_integration.RungeKutta4, monolithic_domain_regular, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.FullyImplicitCoupling(), time_integration.ExplicitHeun, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.FullyImplicitCoupling(), time_integration.ImplicitTrapezoidalRule, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.FullyImplicitCoupling(), time_integration.RungeKutta4, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
save_experiments(experiments, 'series1_Order')

print("### experimental series 2: Customized schemes Semi Implicit-Explicit coupling and Predictor coupling on regular domain")

experiments = list()
experiments.append(monolithic_experiment(time_integration.ExplicitHeun, monolithic_domain_regular, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(monolithic_experiment(time_integration.ImplicitTrapezoidalRule, monolithic_domain_regular, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.FullyExplicitCoupling(), time_integration.ExplicitHeun, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.FullyImplicitCoupling(), time_integration.ImplicitTrapezoidalRule, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.ExplicitPredictorCoupling(), time_integration.ExplicitHeun, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.SemiImplicitExplicitCoupling(), time_integration.ImplicitTrapezoidalRule, left_domain, right_domain_coarse, u_ref_left_regular, u_ref_right_regular, experiment_timesteps))
save_experiments(experiments, 'series2_Custom')

print("### experimental series 3: Strang splitting coupling on non-regular domain")

experiments = list()
experiments.append(monolithic_experiment(time_integration.ExplicitHeun, monolithic_domain_nonregular, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(monolithic_experiment(time_integration.ImplicitTrapezoidalRule, monolithic_domain_nonregular, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(monolithic_experiment(time_integration.RungeKutta4, monolithic_domain_nonregular, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.StrangSplittingCoupling(), time_integration.ExplicitHeun, left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.StrangSplittingCoupling(), time_integration.ImplicitTrapezoidalRule, left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.StrangSplittingCoupling(), time_integration.RungeKutta4, left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.StrangSplittingCoupling(), [time_integration.RungeKutta4, time_integration.ImplicitTrapezoidalRule], left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
save_experiments(experiments, 'series3_Strang')

print("### experimental series 4: Waveform relaxation coupling on non-regular domain")

experiments = list()
experiments.append(monolithic_experiment(time_integration.ImplicitTrapezoidalRule, monolithic_domain_nonregular, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(monolithic_experiment(time_integration.RungeKutta4, monolithic_domain_nonregular, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.WaveformCoupling(), time_integration.ImplicitTrapezoidalRule, left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.WaveformCoupling(), time_integration.RungeKutta4, left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.WaveformCoupling(1,10), time_integration.RungeKutta4, left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
experiments.append(coupling_experiment(coupling_schemes.WaveformCoupling(1,1), [time_integration.RungeKutta4, time_integration.ImplicitTrapezoidalRule], left_domain, right_domain_fine, u_ref_left_nonregular, u_ref_right_nonregular, experiment_timesteps))
save_experiments(experiments, 'series4_WR')
