

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 22:41:38 2025

@author: Hassan
"""
import numpy as np

def sphere_function(x):
    return np.sum(x**2)

def rastrigin(x):
    """
    Rastrigin function
    xx: numpy array or list of shape (d,)
    returns: scalar value
    """
    x = np.asarray(x)
    d = len(x)

    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(xx, a=20.0, b=0.2, c=2*np.pi):
    """
    Ackley function

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s)
    a, b, c : float
        Ackley constants

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    if xx.ndim == 1:
        xx = xx[None, :]  # make it (1, d)

    d = xx.shape[1]

    sum1 = np.sum(xx**2, axis=1)
    sum2 = np.sum(np.cos(c * xx), axis=1)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    y = term1 + term2 + a + np.exp(1)

    return y if len(y) > 1 else y[0]

def styblinski_tang(xx):
    """
    Styblinski–Tang function

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s)

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    if xx.ndim == 1:
        xx = xx[None, :]  # convert to (1, d)

    val = np.sum(xx**4 - 16*xx**2 + 5*xx, axis=1)
    y = val / 2.0

    return y if len(y) > 1 else y[0]
def rosenbrock(xx):
    """
    Rosenbrock function

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s)

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    if xx.ndim == 1:
        xx = xx[None, :]  # make batch

    xi = xx[:, :-1]
    xnext = xx[:, 1:]

    y = np.sum(100.0 * (xnext - xi**2)**2 + (xi - 1)**2, axis=1)

    return y if len(y) > 1 else y[0]
def levy(xx):
    """
    Levy function (benchmark optimization test function)

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s). If 1D, returns a float. If 2D, returns an array of values.

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    # Ensure batch shape
    if xx.ndim == 1:
        xx = xx[None, :]

    # w_i = 1 + (x_i - 1)/4
    w = 1 + (xx - 1) / 4

    # First term: sin^2(pi * w[0])
    term1 = np.sin(np.pi * w[:, 0])**2

    # Sum over i=1 to d-1
    wi = w[:, :-1]
    sum_terms = np.sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2), axis=1)

    # Last term
    wd = w[:, -1]
    term3 = (wd - 1)**2 * (1 + np.sin(2 * np.pi * wd)**2)

    y = term1 + sum_terms + term3
    return y if len(y) > 1 else y[0]


def schwefel(xx):
    """
    Schwefel function

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s)

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    if xx.ndim == 1:
        xx = xx[None, :]  # make batch

    d = xx.shape[1]

    sum_term = np.sum(xx * np.sin(np.sqrt(np.abs(xx))), axis=1)
    y = 418.9829 * d - sum_term

    return y if len(y) > 1 else y[0]


def zakharov(xx):
    """
    Zakharov function

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s)

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    if xx.ndim == 1:
        xx = xx[None, :]  # make batch

    d = xx.shape[1]
    ii = np.arange(1, d + 1)  # 1, 2, ..., d

    sum1 = np.sum(xx**2, axis=1)
    sum2 = np.sum(0.5 * ii * xx, axis=1)

    y = sum1 + sum2**2 + sum2**4

    return y if len(y) > 1 else y[0]


def lennard_jones_cluster(x, N, epsilon=1.0, sigma=1.0, penalty=1e8):
    """
    x: flat vector of length 3N -> [x0,y0,z0, x1,y1,z1, ...]
    returns: total LJ energy
    """
    coords = np.asarray(x, dtype=float).reshape(N, 3)

    E = 0.0
    sig6 = sigma**6
    sig12 = sig6**2

    for i in range(N - 1):
        ri = coords[i]
        for j in range(i + 1, N):
            rj = coords[j]
            d = ri - rj
            r2 = float(d @ d)

            # Overlap / extremely close -> huge penalty
            if r2 < 1e-12:
                return penalty

            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2

            E += 4.0 * epsilon * (sig12 * inv_r12 - sig6 * inv_r6)

    return E

def griewank(xx):
    """
    Griewank function

    Global minimum:
        f(x*) = 0 at x* = (0, ..., 0)

    Typical domain:
        [-600, 600]^d

    Parameters
    ----------
    xx : array-like, shape (d,) or (N, d)
        Input vector(s)

    Returns
    -------
    y : float or np.ndarray
        Function value(s)
    """
    xx = np.asarray(xx)

    if xx.ndim == 1:
        xx = xx[None, :]  # make batch

    d = xx.shape[1]
    ii = np.arange(1, d + 1)  # 1, 2, ..., d

    sum_term = np.sum(xx**2, axis=1) / 4000.0
    prod_term = np.prod(np.cos(xx / np.sqrt(ii)), axis=1)

    y = sum_term - prod_term + 1.0

    return y if len(y) > 1 else y[0]



def shifted_rosenbrock(x, fbias=390.0):
    o_full = np.loadtxt("rosenbrock_func_data.txt")
    D = x.size
    o = o_full[:D]

    """
    CEC2005 F6: Shifted Rosenbrock's Function
    z = x - o + 1
    F(x) = sum_{i=1}^{D-1} [100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2] + fbias
    Global optimum at x = o (then z = 1-vector).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    o = np.asarray(o, dtype=np.float64).reshape(-1)
    assert x.size == o.size, "x and o must have same dimension D"

    z = x - o + 1.0
    return float(np.sum(100.0 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1.0)**2) + fbias)

def shifted_ackley(x, fbias=-140):
    """
    Shifted Ackley Function (CEC style)

    Standard Ackley:
    f(z) = -20 * exp(-0.2 * sqrt(1/D * sum(z_i^2)))
           - exp(1/D * sum(cos(2*pi*z_i)))
           + 20 + e

    Shifted version:
    z = x - o

    Global optimum:
    at x = o
    f(x) = fbias
    """

    # load full shift vector
    o_full = np.loadtxt("ackley_func_data.txt")

    # ensure x is numpy vector
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    D = x.size

    # use first D elements of shift vector
    o = o_full[:D]

    assert x.size == o.size, "x and o must have same dimension"

    # shifted variable
    z = x - o

    # Ackley constants
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi

    sum_sq = np.sum(z**2)
    sum_cos = np.sum(np.cos(c * z))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / D))
    term2 = -np.exp(sum_cos / D)

    f = term1 + term2 + a + np.e

    return float(f + fbias)



def shifted_rastrigin(x, fbias=-330.0):
    """
    CEC2005 F9: Shifted Rastrigin Function

    Standard Rastrigin:
        f(z) = sum(z_i^2 - 10*cos(2*pi*z_i) + 10)

    Shifted version:
        z = x - o

    Global optimum:
        x = o
        f(x) = -330
    """

    # load shift vector
    o_full = np.loadtxt("rastrigin_func_data.txt")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    D = x.size

    o = o_full[:D]

    assert x.size == o.size, "dimension mismatch"

    # shift
    z = x - o

    # rastrigin computation
    f = np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0)

    return float(f + fbias)


def shifted_griewank(x, fbias=-180.0):
    """
    CEC2005 F7: Shifted Griewank Function

    Standard Griewank:
        f(z) = 1 + sum(z_i^2)/4000 - prod(cos(z_i / sqrt(i)))

    Shifted version:
        z = x - o

    Global optimum:
        x = o
        f(x) = -180
    """

    # load shift vector
    o_full = np.loadtxt("griewank_func_data.txt")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    D = x.size

    o = o_full[:D]

    assert x.size == o.size, "dimension mismatch"

    # shift
    z = x - o

    # compute griewank
    sum_term = np.sum(z**2) / 4000.0

    indices = np.arange(1, D + 1)
    prod_term = np.prod(np.cos(z / np.sqrt(indices)))

    f = 1.0 + sum_term - prod_term

    return float(f + fbias)


def shifted_sphere(x, fbias=-450.0):
    """
    CEC2005 F1: Shifted Sphere Function

    Standard Sphere:
        f(z) = sum(z_i^2)

    Shifted version:
        z = x - o

    Global optimum:
        x = o
        f(x) = -450
    """

    # load shift vector
    o_full = np.loadtxt("sphere_func_data.txt")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    D = x.size

    o = o_full[:D]

    assert x.size == o.size, "dimension mismatch"

    # shift
    z = x - o

    return float(np.sum(z**2) + fbias)


_ROT_MATS = {}

def random_orthogonal_matrix(D: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, D))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q

def get_rotation(D: int, seed: int = 42) -> np.ndarray:
    key = (D, seed)
    if key not in _ROT_MATS:
        _ROT_MATS[key] = random_orthogonal_matrix(D, seed)
    return _ROT_MATS[key]

def rotated_rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    M = get_rotation(x.size, seed=42)
    return rosenbrock(M @ x)

def rotated_ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    M = get_rotation(x.size, seed=42)
    return ackley(M @ x)

def rotated_rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    M = get_rotation(x.size, seed=42)
    return rastrigin(M @ x)

def rotated_griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    M = get_rotation(x.size, seed=42)
    return griewank(M @ x)