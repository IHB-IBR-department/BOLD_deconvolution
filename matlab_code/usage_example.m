clear
close all

% Load neuronal PSY regressor and PPI regressor calculated by SPM PEB
load(fullfile(pwd,'examples','01_Simulated_Block_Design_[TR_2s]_[NT_16].mat'))

% Setup variables
TR = 2;                          % Time repetition, [s]
NT = 16;                         % Microtime resolution (number of time bins per scan)
alpha = 0.005;                   % Regularization parameter 


% % Uncomment to load HCP Working Memory task example
% % Load neuronal PSY regressor and PPI regressor calculated by SPM PEB
% load(fullfile(pwd,'examples','03_Empirical_Block_Design_[TR_720ms]_[NT_16].mat'))
% % Setup variables
% TR = 0.72;                       % Time repetition, [s]
% NT = 16;                         % Microtime resolution (number of time bins per scan)
% alpha = 0.005;                   % Regularization parameter 

%% Run BOLD deconvolution function
neuro = bold_deconvolution(preproc_BOLD_signal,TR,alpha,NT);


%% Plot neuronal signal recovered by SPM12 PEB and ridge regression
figure(1)
plot(neuro); hold on; plot(spm_phys_neuro);
legend(['Ridge regression, alpha = ' num2str(alpha)],'SPM12 PEB');
title('Recovered neuronal signal');
fprintf(['Correlation between neuronal signals recovered by SPM12 PEB and ridge regression, r = ' num2str(corr(neuro,spm_phys_neuro)) '\n']);


%% Calculate PPI term

% PPI term at neuronal level
neuro = detrend(neuro);
PSY = detrend(psy_neuro);
PSYxn = PSY.*neuro;

% Create SPM HRF in microtime resolution
dt = TR/NT;                      % Length of time bin, [s]
t = 0:dt:32;
hrf = gampdf(t,6) - gampdf(t,NT)/6;
hrf = hrf'/sum(hrf);

% Reconvolution
N = size(preproc_BOLD_signal,1); % Scan duration, [dynamics] 
k = 1:NT:N*NT;                   % microtime to scan time indices
T0 = 8;                          % Microtime onset (reference time bin, see slice timing)
ppi = conv(PSYxn,hrf);
ppi = ppi((k-1) + T0);
figure(2)
plot(ppi); hold on; plot(spm_ppi); 
legend(['Ridge regression, alpha = ' num2str(alpha)],'SPM12 PEB');
title('PPI regressors');
fprintf(['Correlation between PPI terms calculated by SPM12 PEB and ridge regression, r = ' num2str(corr(ppi,spm_ppi)) '\n']);


%% Run BOLD deconvolution for multiple ROIs

% Assume we have 400 ROIs 
BOLD = repmat(preproc_BOLD_signal,1,400);  

% Use parallel computations 
par = 1;                         
neuro_ts = bold_deconvolution(BOLD,TR,alpha,NT,par);