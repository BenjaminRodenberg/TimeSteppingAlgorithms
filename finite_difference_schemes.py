def central_FD_at(i, h, u, derivative=1, order=2):
    """
    computes the n_th central finite difference at x_i on the basis of u_i.
    :param i:
    :param h:
    :param u:
    :param derivative:
    :param order:
    :return:
    """

    du = 0

    if derivative == 1:
        if order == 2:
            du = 1.0/(2.0 * h) * (-1.0 * u[i-1] + 1.0 * u[i+1])
        else:
            print "not implemented!"
    else:
        print "not implemented!"

    return du


def one_sided_forward_FD_at(i, h, u, derivative=1, order=1):
    """
    computes the n_th one-sided forward finite difference derivative at x_i on the basis of u_i.

    Coefficients from https://en.wikipedia.org/wiki/Finite_difference_coefficient

    d^n u/dn (x_i)= a*u_i + b*u_i+1 + c*u_i+2 + ...

    :param i:
    :param u:
    :param order:
    :param derivative:
    :return:
    """

    du = 0

    if derivative == 1:
        if order == 1:
            du = 1.0/h * (- 1.0 * u[i] + 1.0 * u[i+1] )
        elif order == 2:
            du = 1.0/h * (-1.5 * u[i] + 2.0 * u[i+1] - 0.5 * u[i+2])
        elif order == 3:
            du = 1.0/h * (-11.0/6.0 * u[i] + 3.0 * u[i+1] - 3.0/2.0 * u[i+2] + 1.0/3.0 * u[i+3])
        elif order == 4:
            du = 1.0/h * (-25.0/12.0 * u[i] + 4.0 * u[i+1] - 3.0 * u[i+2] + 4.0/3.0 * u[i+3] - 1.0/4.0 * u[i+4])
        elif order == 5:
            du = 1.0/h * (-137.0/60.0 * u[i] + 5.0 * u[i+1] - 5.0 * u[i+2] + 10.0/3.0 * u[i+3] - 5.0/4.0 * u[i+4] + 1.0/5.0 * u[i+5])
        else:
            print "not implemented!"
    else:
        print "not implemented!"

    return du
