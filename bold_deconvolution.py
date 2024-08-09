import numpy as np
from scipy.stats import gamma

def ridge_regress_deconvolution(BOLD: np.ndarray,
                                TR: float,
                                alpha: float = 0.005,
                                NT: int = 16,
                                xb: np.ndarray = None,
                                Hxb: np.ndarray = None) -> np.ndarray:
    """
    Deconvolves a preprocessed BOLD signal into neuronal time series
    based on selected temporal basis set (by default: discrete cosine
    set convolved with canonical HRF) and ridge regression.

    This function does not use confound regressors (e.g., motion).
    This function does not perform whitening and temporal filtering.
    The BOLD input signal must already be pre-processed.

    :param BOLD: Preprocessed BOLD signal (numpy array)
    :type BOLD: numpy.ndarray (NxROI)
    :param TR: Time repetition, in seconds
    :type TR: float
    :param alpha: Regularization parameter, defaults to 0.005
    :type alpha: float, optional
    :param NT: Microtime resolution (number of time bins per scan), defaults to 16
    :type NT: int, optional
    :param xb: Temporal basis set in microtime resolution
    :type xb: numpy.ndarray (N*NTxN*NT)
    :param Hxb: Convolved temporal basis set in scan resolution
    :type Hxb: numpy.ndarray (NxN)
    :return: Deconvolved neuronal time series
    :rtype: numpy.ndarray

    :raises ValueError: If the BOLD signal is empty
    :raises ZeroDivisionError: If the time repetition (TR) is zero
    """

    if len(BOLD) == 0:
        raise ValueError('BOLD signal is empty.')
    if TR == 0:
        raise ZeroDivisionError('TR should be more than 0')
    dt = TR / NT   # Length of time bin, [s]
    N = len(BOLD)  # Scan duration, [dynamics]
    k = np.arange(0, N * NT, NT)  # Microtime to scan time indices

    if xb is None and Hxb is not None:
        raise ValueError("Both xb and Hxb should be specified, or neither of them")
    if Hxb is None and xb is not None:
        raise ValueError("Both xb and Hxb should be specified, or neither of them")

    if xb is None:
        xb, Hxb = compute_xb_Hxb(N, NT, TR) # Use discrete cosine set convolved with canonical HRF

    # Perform ridge regression
    C = np.linalg.solve(Hxb.T @ Hxb + alpha * np.eye(N), Hxb.T @ BOLD)

    # Recover neuronal signal
    neuro = xb @ C

    return neuro.flatten()


def compute_xb_Hxb(N: int, NT: int, TR: float) -> np.ndarray:
    dt = TR / NT
    k = np.arange(0, N * NT, NT)  # Microtime to scan time indices

    # Create canonical HRF in microtime resolution (identical to SPM12 cHRF)
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
    return xb, Hxb


def dctmtx_numpy_vect(N: int, K: int) -> np.ndarray:
    n = np.arange(N)
    C = np.zeros((N, K))
    C[:, 0] = 1 / np.sqrt(N)
    k = np.arange(1, K)
    C[:, 1:K] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n[:, np.newaxis]) * k / (2 * N))
    return C


