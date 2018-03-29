import numpy as np
import abc
from time_integration import TimeIntegrationScheme


class Domain():
    __metaclass__ = abc.ABCMeta

    def __init__(self, grid):
        """
        Simulation domain with a grid, corresponding function values u, boundary conditions and a time integration scheme
        :param grid: grid discretizing domain
        :type grid: OneDimensionalGrid
        """
        self.grid = grid
        self.u = np.zeros(self.grid.N_gridpoints)  # :type : np.array
        self.left_BC = {}
        self.right_BC = {}
        self.time_integration_scheme = None  # :type : TimeIntegrationScheme

    def update_u(self, new_u):
        """
        updates u on the domain
        :param new_u:
        :type new_u: np.array
        :return:
        """
        assert self.u.shape == new_u.shape
        self.u = new_u


class OneDimensionalGrid():
    __metaclass__ = abc.ABCMeta

    def __init__(self, x):
        """
        :param x: array of points on 1D grid
        """
        self.x = np.array(x)
        self.N_gridpoints = self.x.__len__()  # :type : int
        self.x_left = self.x[0]  # :type : float
        self.x_right = self.x[-1]  # :type : float


class RegularGrid(OneDimensionalGrid):

    def __init__(self, N_gridpoints, x_left, x_right):
        """
        Regular grid with N_gridpoints between x_left and x_right
        :param N_gridpoints: number of gridpoints
        :type N_gridpoints: int
        :param x_left: leftmost coordinate
        :type x_left: float
        :param x_right: rightmost coordinate
        :type x_right: float
        """
        # meshwidth of grid
        self.h = (x_right - x_left) / (N_gridpoints-1)  # :type : float
        # vector of mesh points
        x = np.linspace(x_left, x_right, N_gridpoints)  # :type : np.array
        super(RegularGrid, self).__init__(x)


class IrregularGrid(OneDimensionalGrid):

    def __init__(self, x):
        """
        Irregular grid defined by ascending vector of gridpoints
        :param x: vector of gridpoints
        """
        super(IrregularGrid, self).__init__(x)
