import numpy as np
from numba import njit
from scipy.stats import gamma
from scipy import signal
from scipy.fftpack import dct
from nilearn.glm.first_level import spm_hrf, compute_regressor
from sklearn.linear_model import Ridge


def ridge_deconvolution(BOLD: np.ndarray,
                        TR: float,
                        alpha: float = 0.005,
                        NT: int = 16) -> np.ndarray:
    """
    Deconvolves a preprocessed BOLD signal into neuronal time series
    based on discrete cosine set and ridge regression.

   :param BOLD: Preprocessed BOLD signal (numpy array)
   :param TR: Time repetition, in seconds
   :param alpha:  Regularization parameter, defaults to 0.005
   :param NT: Microtime resolution (number of time bins per scan), defaults to 16
   :return: Deconvolved neuronal time series
   """

    N = len(BOLD)
    # Create cosine basis set
    xb = dctmtx_numpy_vect(NT * N + 128, N)

    # Create Nilearn HRF with specified microtime resolution
    time_length = 32
    dt = TR / NT
    frame_times = np.linspace(0, time_length, int(time_length / dt + 1))
    onset, amplitude, duration = 0.0, 1.0, dt
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)

    nilearn_hrf = compute_regressor(exp_condition, "spm", frame_times, con_id="main")[0]

    Hxb = np.zeros((N, N))

    for i in range(N):
        Hxb[:, i] = np.squeeze(signal.convolve(xb[:, i].reshape(-1, 1), nilearn_hrf)[1 + 128:N * NT + 128:NT])

    reg_alpha = Ridge(alpha=alpha, solver='lsqr', fit_intercept=False, max_iter=1000)
    reg_alpha.fit(Hxb, BOLD)

    neuronal = np.matmul(xb[128:N * NT + 128, :], reg_alpha.coef_[0, :N])
    return neuronal.flatten()


def ridge_regress_deconvolution(BOLD: np.ndarray,
                                TR: float,
                                alpha: float = 0.005,
                                NT: int = 16) -> np.ndarray:
    """
    Deconvolves a preprocessed BOLD signal into neuronal time series
    based on discrete cosine set and ridge regression.

    This function does not use confound regressors (e.g., motion).
    This function does not perform whitening and temporal filtering.
    The BOLD input signal must already be pre-processed.

    :param BOLD: Preprocessed BOLD signal (numpy array)
    :type BOLD: numpy.ndarray
    :param TR: Time repetition, in seconds
    :type TR: float
    :param alpha: Regularization parameter, defaults to 0.005
    :type alpha: float, optional
    :param NT: Microtime resolution (number of time bins per scan), defaults to 16
    :type NT: int, optional
    :return: Deconvolved neuronal time series
    :rtype: numpy.ndarray

    :raises ValueError: If the BOLD signal is empty
    :raises ZeroDivisionError: If the time repetition (TR) is zero
    """

    if len(BOLD) == 0:
        raise ValueError('BOLD signal is empty.')
    if TR == 0:
        raise ZeroDivisionError('TR should be more than 0')
    dt = TR / NT  # Length of time bin, [s]
    N = len(BOLD)  # Scan duration, [dynamics]
    k = np.arange(0, N * NT, NT)  # Microtime to scan time indices

    # Create canonical HRF in microtime resolution (identical to SPM cHRF)
    t = np.arange(0, 32 + dt, dt)
    hrf = gamma.pdf(t, 6) - gamma.pdf(t, NT) / 6
    hrf = hrf / np.sum(hrf)

    # Create convolved discrete cosine set
    M = N * NT + 128
    xb = dctmtx_numpy_vect(M, N)

    Hxb = np.zeros((N, N))
    for i in range(N):
        Hx = np.convolve(xb[:, i], hrf, mode='full')
        Hxb[:, i] = Hx[k + 128]
    xb = xb[128:, :]

    # Perform ridge regression
    C = np.linalg.solve(Hxb.T @ Hxb + alpha * np.eye(N), Hxb.T @ BOLD)

    # Recover neuronal signal
    neuro = xb @ C

    return neuro.flatten()


def dctmtx(N: int, K: int, is_numba=False) -> np.ndarray:
    if is_numba:
        C = dctmtx_numba(N, K)
    else:
        C = dctmtx_numba(N, K)
    return C


# Regular numpy version of dctmtx
def dctmtx_numpy(N: int, K: int) -> np.ndarray:
    n = np.arange(N)
    C = np.zeros((N, K))
    C[:, 0] = np.ones(N) / np.sqrt(N)
    for k in range(1, K):
        C[:, k] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n) * k / (2 * N))
    return C


def dct_scipy():
    # not implemented, but definitely need to check with dct
    pass


def dctmtx_numpy_vect(N: int, K: int) -> np.ndarray:
    n = np.arange(N)
    C = np.zeros((N, K))
    C[:, 0] = 1 / np.sqrt(N)
    k = np.arange(1, K)
    C[:, 1:K] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n[:, np.newaxis]) * k / (2 * N))
    return C


# Numba optimized version of dctmtx
@njit(fastmath=True)
def dctmtx_numba(N: int, K: int) -> np.ndarray:
    n = np.arange(N)
    C = np.zeros((N, K))
    C[:, 0] = np.ones(N) / np.sqrt(N)
    for k in range(1, K):
        C[:, k] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n) * k / (2 * N))
    return C
