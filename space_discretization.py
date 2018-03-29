from __future__ import division
import scipy.sparse as sp
import numpy as np

from domain import IrregularGrid, RegularGrid, OneDimensionalGrid


def compute_finite_difference_scheme_coeffs(evaluation_points, derivative_order):
    """
    computes the coefficients for a Finite Difference schemes to compute the d-th derivative based on evaluation points
    at u(x+p[i]*h). The resulting scheme has the form $\sum\limits_i coeff_i * u(x+p_i*h) / h^d$ with h defining the
    stepwidth, p_i defining the evaluation points and coeff_i being the returned FD coefficients.
    See http://web.media.mit.edu/~crtaylor/calculator.html
    :param evaluation_points: vector providing evaluation points
    :param derivative_order: derivative
    :return: coefficients corresponding to evaluation points
    """

    n = evaluation_points.shape[0]
    A = np.zeros([n,n])
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            A[i,j] = evaluation_points[j] ** i
        b[i] = np.math.factorial(i)*int(i == derivative_order)

    return np.linalg.solve(A, b)


def second_derivative_matrix(grid, **kwargs):
    """
    creates the matrix operator u_xx = Au + R with Dirichlet or Neumann boundary conditions.
    Used e.g. for the 1D heat transport equation: u_t = c * u_xx
    u = [u_0, ... ,u_N]
    :param grid: grid where we create the finite difference operator for u_xx
    :type grid: OneDimensionalGrid
    :param kwargs: boundary conditions neumann/dirichlet_l/r correspond to neumann/dirichlet boundary conditions at left/right boundary
    :return: A,R
    """

    if type(grid) is RegularGrid:
        return regular_matrix_operator(grid.h, grid.N_gridpoints, **kwargs)
    elif type(grid) is IrregularGrid:
        return irregular_matrix_operator(grid.x, **kwargs)
    else:
        print("in second_derivative_matrix(...) unsupported grid type!")
        quit()


def irregular_matrix_operator(x, **kwargs):
    """
    creates the matrix operator u_xx = Au + R with Dirichlet or Neumann boundary conditions.
    Used e.g. for the 1D heat transport equation: u_t = c * u_xx
    u = [u_0, ... ,u_N]
    :param x: arbitrary 1D mesh
    :param kwargs: boundary conditions neumann/dirichlet_l/r correspond to neumann/dirichlet boundary conditions at left/right boundary
    :return: A,R
    """

    N = x.shape[0]
    l = np.zeros(N-1) # lower diagonal
    d = np.zeros(N) # diagonal
    u = np.zeros(N-1) # upper diagonal
    for i in range(1,N-1):  # iterate over inner nodes
        stencil = np.array([x[i-1]-x[i],x[i]-x[i],x[i+1]-x[i]])  # create three point stencil
        l[i-1], d[i], u[i] = compute_finite_difference_scheme_coeffs(stencil, 2)  # compute coefficients for three point stencil

    A = (sp.diags(l,-1) + sp.diags(d) + sp.diags(u,+1)).tolil()
    R = np.zeros(N)

    # add boundary conditions
    if 'neumann_l' in kwargs:
        # modify row and column corresponding to neumann BC at u_0
        h = x[1]-x[0]  # use h of leftmost cell
        A[0,0] = -2.0/h**2
        A[0,1] = +2.0/h**2
        R[0] = -2.0/h * kwargs.get('neumann_l')
    elif 'dirichlet_l' in kwargs:
        # remove row and column corresponding to dirichlet BC at u_0
        h = x[1]-x[0]  # use h of leftmost cell
        A[0,:] = 0
        A[:,0] = 0
        # manually add it to R
        R[1] = 1.0/h**2 * kwargs.get('dirichlet_l')
    else:
        print("in irregular_matrix_operator(...): insufficient boundary conditions at left boundary!")
        quit()

    if 'neumann_r' in kwargs:
        # modify row and column corresponding to neumann BC at u_N
        h = x[-2]-x[-1]  # use h of rightmost cell
        A[-1,-1] = -2.0/h**2
        A[-1,-2] = +2.0/h**2
        R[-1] = 2.0/h * kwargs.get('neumann_r')
    elif 'dirichlet_r' in kwargs:
        # remove row and column corresponding to dirichlet BC at u_N
        h = x[-2]-x[-1]  # use h of rightmost cell
        A[-1,:] = 0
        A[:,-1] = 0
        # manually add it to R
        R[-2] = 1.0/h**2 * kwargs.get('dirichlet_r')
    else:
        print("in irregular_matrix_operator(...): insufficient boundary conditions at right boundary!")
        quit()

    return A, R


def regular_matrix_operator(h, N, **kwargs):
    """
    creates the matrix operator u_xx = Au + R with Dirichlet or Neumann boundary conditions.
    Used e.g. for the 1D heat transport equation: u_t = c * u_xx
    u = [u_0, ... ,u_N]
    :param h: meshwidth
    :param N: number of gridpoints - 1 (including boundary condition gridpoints)
    :param kwargs: boundary conditions neumann/dirichlet_l/r correspond to neumann/dirichlet boundary conditions at left/right boundary
    :return: A,R
    """

    # second derivative u_xx
    A = 1.0/h**2 * (sp.eye(N, N, -1) - 2 * sp.eye(N, N) + sp.eye(N, N, 1)).tolil()
    R = 2.0/h*np.zeros(N)

    # add boundary conditions
    if 'neumann_l' in kwargs:
        # modify row and column corresponding to neumann BC at u_0
        A[0,0] = -2.0/h**2
        A[0,1] = +2.0/h**2
        R[0] = -2.0/h * kwargs.get('neumann_l')
    elif 'dirichlet_l' in kwargs:
        # remove row and column corresponding to dirichlet BC at u_0
        A[0,:] = 0
        A[:,0] = 0
        # manually add it to R
        R[1] = 1.0/h**2 * kwargs.get('dirichlet_l')
    else:
        print("in regular_matrix_operator(...): insufficient boundary conditions at left boundary!")
        quit()

    if 'neumann_r' in kwargs:
        # modify row and column corresponding to neumann BC at u_N
        A[-1,-1] = -2.0/h**2
        A[-1,-2] = +2.0/h**2
        R[-1] = 2.0/h * kwargs.get('neumann_r')
    elif 'dirichlet_r' in kwargs:
        # remove row and column corresponding to dirichlet BC at u_N
        A[-1,:] = 0
        A[:,-1] = 0
        # manually add it to R
        R[-2] = 1.0/h**2 * kwargs.get('dirichlet_r')
    else:
        print("in regular_matrix_operator(...): insufficient boundary conditions at right boundary!")
        quit()

    return A, R
