function neuro = bold_deconvolution(BOLD,TR,alpha,NT,par,xb,Hxb)

% Deconvolves a preprocessed BOLD signal into neuronal time series 
% based on discrete cosine set and ridge regression.
% This function does not use confound regressors (e.g. motion).
% This function does not perform whitening and temporal filtering.
% The BOLD input signal must already be pre-processed.
%
% FORMAT neuro = ridge_regress_deconv(BOLD,TR)
%
% BOLD  - Preprocessed BOLD signal (time points X ROIs)
% TR    - Time repetition, [s]
%
% FORMAT neuro = bold_deconvolution(BOLD,TR,alpha,NT,par,xb,Hxb)
%
% Optional inputs:
%
% alpha - Regularization parameter 
%         (default: 0.005)
% NT    - Microtime resolution (number of time bins per scan)
%         (default: 16)
% par   - Parallel or sequential computations
%         (default: 0)
% xb    - Temporal basis set in microtime resolution 
%         (default: discrete cosine set)
% Hxb   - Convolved temporal basis set in scan resolution
%         (default: discrete cosine set convolved with canonical HRF)
% _________________________________________________________________________
% Copyright (C) 2024 Ruslan Masharipov
% Contact email: masharipov@ihb.spb.ru

% Setup variables
if nargin < 2
    error('Define time repetition (TR)')
elseif nargin < 3
    alpha = 0.005;
    NT = 16;
    par = 0;
elseif nargin < 4
    NT = 16;
    par = 0;
elseif nargin < 5
    par = 0;
elseif nargin == 6
    error('Define convolved temporal basis set (Hxb)')
end

dt = TR/NT;                      % Length of time bin, [s]
N = size(BOLD,1);                % Scan duration, [dynamics] 
k = 1:NT:N*NT;                   % Microtime to scan time indices

if exist('Hxb', 'var') ~= 1

    % Create canonical HRF in microtime resolution (identical to SPM cHRF)
    t = 0:dt:32;
    hrf = gampdf(t,6) - gampdf(t,NT)/6;
    hrf = hrf'/sum(hrf);
    
    % Create convolved discrete cosine set
    M = N*NT + 128;
    n = (0:(M -1))';
    xb = zeros(size(n,1),N);
    xb(:,1) = ones(size(n,1),1)/sqrt(M);
    for j=2:N
        xb(:,j) = sqrt(2/M)*cos(pi*(2*n+1)*(j-1)/(2*M));
    end
    
    Hxb = zeros(N,N);
    for i = 1:N
        Hx       = conv(xb(:,i),hrf);
        Hxb(:,i) = Hx(k + 128);
    end
    xb = xb(129:end,:);

end

switch par
    case 0
        for ROI = 1:size(BOLD,2)
            % Perform ridge regression
            C(:,ROI) = (Hxb'*Hxb + alpha*eye(length(Hxb)))\(Hxb'*BOLD(:,ROI));
            
            % Recover neuronal signal
            neuro(:,ROI) = xb*C(:,ROI);
        end
    case 1
        parfor ROI = 1:size(BOLD,2)
            % Perform ridge regression
            C(:,ROI) = (Hxb'*Hxb + alpha*eye(length(Hxb)))\(Hxb'*BOLD(:,ROI));
            
            % Recover neuronal signal
            neuro(:,ROI) = xb*C(:,ROI);
        end
end