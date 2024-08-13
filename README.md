# fMRI BOLD signal deconvolution
 

This repository provides [Python](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/python_code/bold_deconvolution.py) and [MATLAB](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/matlab_code/bold_deconvolution.m) functions to perform hemodynamic deconvolution of preprocessed BOLD signals into estimated neuronal time series. The deconvolution process is based on temporal basis set (dy default: discrete cosine set) and ridge regression. More details can be found in the [Masharipov et al. "Comparison of whole-brain task-modulated functional connectivity methods for fMRI task connectomics." bioRxiv (2024): 2024-01](https://doi.org/10.1101/2024.01.22.576622). If you employ this code, please cite the referenced study.

## Python function: *ridge_regress_deconvolution*

The *ridge_regress_deconvolution* function deconvolves a preprocessed BOLD signal into neuronal time series without using confound regressors (e.g., motion) and without performing whitening and temporal filtering. The input BOLD signal must already be preprocessed.

## Parameters
**BOLD**(np.ndarray):
Preprocessed BOLD signal (numpy array).<br />
**TR** (float): Time repetition in seconds.<br />
**alpha** (float, optional, default=0.005): Regularization parameter for ridge regression.<br />
**NT** (int, optional, default=16): Microtime resolution (number of time bins per scan).<br />
**xb** (np.ndarray, optional, default: discrete cosine set): Temporal basis set in microtime resolution.<br />
**Hxb**  (np.ndarray, optional, default: discrete cosine set convolved with canonical HRF): Convolved temporal basis set in scan resolution.<br />

**Returns:** Deconvolved neuronal time series (np.ndarray)


## Basic usage
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

## MATLAB function: *bold_deconvolution*

The *bold_deconvolution* function deconvolves a preprocessed BOLD signal into neuronal time series without using confound regressors (e.g., motion) and without performing whitening and temporal filtering. The input BOLD signal must already be preprocessed.

## Parameters

**BOLD**  - Preprocessed BOLD signal (time points X ROIs) <br />
**TR**    - Time repetition, [s]

*Optional:*

**alpha** - Regularization parameter (default: 0.005)<br />
**NT**    - Microtime resolution (number of time bins per scan)(default: 16)<br />
**par**   - Parallel or sequential computations (default: 0)<br />
**xb**    - Temporal basis set in microtime resolution (default: discrete cosine set)<br />
**Hxb**   - Convolved temporal basis set in scan resolution (default: discrete cosine set convolved with canonical HRF)

## Basic usage
```matlab
% Load preprocessed BOLD signal
BOLD = load(preprocessed_BOLD.mat).

% Setup variables
TR = 2;                          % Time repetition, [s]
NT = 16;                         % Microtime resolution (number of time bins per scan)
alpha = 0.005;                   % Regularization parameter

%% Run BOLD deconvolution function
neuro = bold_deconvolution(BOLD,TR,alpha,NT);

```

## MATLAB examples

See usage example in [usage_example.m](https://github.com/IHB-IBR-department/BOLD_deconvolution/blob/main/matlab_code/usage_example.m)
