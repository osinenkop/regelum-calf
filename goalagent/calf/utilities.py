"""
Contains auxiliary functions.

"""

import numpy as np
from numpy.random import rand
import scipy.stats as st


def rej_sampling_rvs(dim, pdf, m):
    """
    Random variable (pseudo)-realizations via rejection sampling.

    Parameters
    ----------
    dim : : integer
        dimension of the random variable
    pdf : : function
        desired probability density function
    m : : number greater than 1
        it must hold that :math:`\\text{pdf}_{\\text{desired}} \le m \\text{pdf}_{\\text{proposal}}`.
        This function uses a normal pdf with zero mean and identity covariance matrix as a proposal distribution.
        The smaller `M` is, the fewer iterations to produce a sample are expected.

    Returns
    -------
    A single realization (in general, as a vector) of the random variable with the desired probability density.

    """

    # Use normal pdf with zero mean and identity covariance matrix as a proposal distribution
    normal_RV = st.multivariate_normal(cov=np.eye(dim))

    # Bound the number of iterations to avoid too long loops
    max_iters = 1e3

    curr_iter = 0

    while curr_iter <= max_iters:
        proposal_sample = normal_RV.rvs()

        unif_sample = rand()

        if unif_sample < pdf(proposal_sample) / m / normal_RV.pdf(proposal_sample):
            return proposal_sample


def to_col_vec(argin):
    """
    Convert number or array to a column vector (as 2D array!).

    """
    if np.isscalar(argin):
        return np.array([[argin]])

    if argin.ndim < 2:
        return np.reshape(argin, (argin.size, 1))
    elif argin.ndim == 2:
        if argin.shape[0] < argin.shape[1]:
            return argin.T
        else:
            return argin


def to_scalar(argin):
    """
    Convert an array to a scalar. If already a scalar, return just it.

    """
    if np.isscalar(argin):
        return argin
    else:
        return argin.item()


def to_row_vec(argin):
    """
    Convert number or array to a row vector (as 2D array!).

    """
    if np.isscalar(argin):
        return np.array([[argin]])

    if argin.ndim < 2:
        return np.reshape(argin, (1, argin.size))
    elif argin.ndim == 2:
        if argin.shape[0] > argin.shape[1]:
            return argin.T
        else:
            return argin


def is_row_vec(argin):
    """
    Checks whether the input is a row vector.

    """

    if len(argin.shape) != 2:
        return False
    else:
        if argin.shape[0] != 1:
            return False
        else:
            return True


def push_vec(matrix, vec):
    """
    Pushes a vector into a matrix at its bottom.

    """
    return np.vstack([matrix[1:, :], vec])


def uptria2vec(mat, force_row_vec=False):
    """
    Convert upper triangular square sub-matrix to column vector.

    """
    n = mat.shape[0]

    vec = np.zeros((int(n * (n + 1) / 2)))

    k = 0
    for i in range(n):
        for j in range(i, n):
            vec[k] = mat[i, j]
            k += 1

    if force_row_vec:
        return to_row_vec(vec)
    else:
        return vec


class ZOH:
    """
    Zero-order hold.

    """

    def __init__(self, init_time=0, init_value=0, sampling_time=1):
        self.clock = init_time
        self.sampling_time = sampling_time
        self.current_signal_value = init_value

    def hold(self, signal_value, t):
        time_in_sample = t - self.clock
        if time_in_sample >= self.sampling_time:  # New sample
            self.clock = t
            self.current_signal_value = signal_value

        return self.current_signal_value
