fixed_point_tol = 10**-6  # tolerance for fixed point iteration
n_max_fixed_point_iterations = 20  # maximum number of fixed point iterations being performed
neumann_coupling_order = 2
neumann_coupling_scheme = 'central'  # choose central or forward FD scheme for approximation of coupling Neumann BC # todo replace with enum?

numeric_parameters_dict = {
    'fixed point tol':fixed_point_tol,
    'n max fixed point iterations':n_max_fixed_point_iterations,
    'neumann coupling order':neumann_coupling_order,
    'neumann coupling scheme':neumann_coupling_scheme
}
