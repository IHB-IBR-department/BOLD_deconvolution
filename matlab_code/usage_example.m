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


%% Run BOLD deconvolution function with precomputed discrete cosine set 
% If we deconvolve multiple BOLD-time series from the same session,
% we can precompute cosine basis set to speed up computations 
% (since we are using the same basis set for all time series)

% Assume we have 400 ROIs 
BOLD = repmat(preproc_BOLD_signal,1,400);  

% Create SPM HRF in microtime resolution
dt = TR/NT;                      % Length of time bin, [s]
t = 0:dt:32;
hrf = gampdf(t,6) - gampdf(t,NT)/6;
hrf = hrf'/sum(hrf);

% Create convolved discrete cosine set
N = size(BOLD,1);                % Scan duration, [dynamics] 
k = 1:NT:N*NT;                   % microtime to scan time indices
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

% Run BOLD deconvolution 
par = 1;                         % Use parallel computations
neuro_ts = bold_deconvolution(BOLD,TR,alpha,NT,par,xb,Hxb);