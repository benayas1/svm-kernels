import numpy as np

def hermite_poly(x, degree):
    if degree == 0:
        return 1
    if degree == 1:
        return x
    if degree > 1:
        return np.dot(x.T, hermite_poly(x, degree - 1)) - degree * hermite_poly(x, degree - 2)


def k_hermite(x, z, degree):
    k = 0
    for i in range(0, degree):
        k = k + np.dot(hermite_poly(x, degree), hermite_poly(z, degree))
    return k / (np.sqrt(np.pi) * np.math.factorial(degree) / (2 ** degree))


def hermite(degree=2):
    k = k_hermite

    def kernel(X, Z):
        return np.array([np.array([k(x, z, degree) for z in Z]) for x in X])

    return kernel


def chebyshev_poly(x, degree):
    if degree == 0:
        return 1
    if degree == 1:
        return x
    if degree > 1:
        return np.dot(2 * x.T, chebyshev_poly(x, degree - 1)) - chebyshev_poly(x, degree - 2)


def k_chebyshev(x, z, degree):
    k = 0
    for i in range(0, degree):
        k = k + np.dot(chebyshev_poly(x, degree), chebyshev_poly(z, degree))
    return k / (np.sqrt(len(x) - np.dot(x, z)))


def chebyshev(degree=2):
    k = k_chebyshev

    def kernel(X, Z):
        return np.array([np.array([k(x, z, degree) for z in Z]) for x in X])

    return kernel