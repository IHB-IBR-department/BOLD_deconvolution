# fMRI BOLD signal deconvolution
 

This repository provides [Python](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/python_code/bold_deconvolution.py) and [MATLAB](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/matlab_code/bold_deconvolution.m) functions to perform hemodynamic deconvolution of preprocessed BOLD signals into estimated neuronal time series. The deconvolution process is based on temporal basis set (dy default: discrete cosine set) and ridge regression.

## Python function: **ridge_regress_deconvolution**

The **ridge_regress_deconvolution** function deconvolves a preprocessed BOLD signal into neuronal time series without using confound regressors (e.g., motion) and without performing whitening and temporal filtering. The input BOLD signal must already be preprocessed.

### Parameters
**BOLD**(np.ndarray):
Preprocessed BOLD signal (numpy array).

**TR** (float): Time repetition in seconds.

**alpha** (float, optional, default=0.005): Regularization parameter for ridge regression.

**NT** (int, optional, default=16): Microtime resolution (number of time bins per scan).

**xb** (np.ndarray, optional, default: discrete cosine set): Temporal basis set in microtime resolution.

**Hxb**  (np.ndarray, optional, default: discrete cosine set convolved with canonical HRF): Convolved temporal basis set in scan resolution.

**Returns:** Deconvolved neuronal time series (np.ndarray)


### Basic usage
```python
import numpy as np
from bold_deconvolution import ridge_regress_deconvolution,  compute_xb_Hxb


# Example usage
preprocessed_BOLD = np.load('path_to_preprocessed_BOLD.npy')
TR = 2.0
NT = 16
alpha = 0.005

neuronal_activity = ridge_regress_deconvolution(BOLD=preprocessed_BOLD, TR=TR, alpha=alpha, NT=NT)

# If we deconvolve multiple BOLD-time series from the same session,
# we can precompute cosine basis set to speed up computations 
# (since we are using the same basis set for all time series)
xb, Hxb = compute_xb_Hxb(len(preprocessed_BOLD), NT, TR)

neural_time_series = ridge_regress_deconvolution(preprocessed_BOLD, TR, alpha, NT, xb=xb, Hxb=Hxb)


```

## Jupyter notebook examples

See usage example in [usage_example.ipynb](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/python_code/usage_example.ipynb)

## MATLAB examples

See usage example in [usage_example.m](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/matlab_code/usage_example.m)
