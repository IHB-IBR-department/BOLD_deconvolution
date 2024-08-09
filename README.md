# Hemodynamic fMRI BOLD deconvolution


This repository provides a Python function to perform hemodynamic deconvolution of fMRI BOLD signals. The deconvolution process is based on a discrete cosine set and ridge regression, converting preprocessed BOLD signals into neuronal time series.

Function: **ridge_regress_deconvolution**
The **ridge_regress_deconvolution** function deconvolves a preprocessed BOLD signal into neuronal time series without using confound regressors (e.g., motion) and without performing whitening and temporal filtering. The input BOLD signal must already be preprocessed.

Function: **compute_xb_Hxb**
The **compute_xb_Hxb** function calculates discrete cosine set (xb) in microtime resulution and convolved discrete cosine set (Hxb) in scan resolution. Convolution is performed using canonical HRF (identical to SPM12 HRF). 


## Parameters
**BOLD**(np.ndarray):
Preprocessed BOLD signal (numpy array).

**TR** (float): Time repetition in seconds.

**alpha** (float, optional, default=0.005): Regularization parameter for ridge regression.

**NT** (int, optional, default=16): Microtime resolution (number of time bins per scan).

**xb** (np.ndarray, optional, default: discrete cosine set): Temporal basis set in microtime resolution.

**Hxb**  (np.ndarray, optional, default: discrete cosine set convolved with canonical HRF): Convolved temporal basis set in scan resolution.

**Returns** Neuronal time series (np.ndarray): The deconvolved neuronal time series.


## Basic usage
```python
import numpy as np
from bold_deconvolution import ridge_regress_deconvolution,  compute_xb_Hxb


# Example usage
preprocessed_BOLD = np.load('path_to_preprocessed_BOLD.npy')
TR = 2.0
NT = 16
alpha = 0.005

neuronal_time_series = ridge_regress_deconvolution(BOLD=preprocessed_BOLD, TR=TR, alpha=alpha, NT=NT)

# If we deconvolve multiple BOLD-time series from the same session,
# we can precompute cosine basis set to speed up computations (since we are using the same basis set for all time series)
xb, Hxb = compute_xb_Hxb(len(preprocessed_BOLD), NT, TR)

neuro_ridge_precomp = ridge_regress_deconvolution(preprocessed_BOLD, TR, alpha, NT, xb=xb, Hxb=Hxb)


```

## Notebook Example

See usage example in [usage_example.ipynb](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/usage_example.ipynb)
