%% Use full path for directories and files wherever needed. After entering parameters, use command icatb_dfnc_batch(input_file);

%% Output directory to place results
outputDir = 'F:\The_first_training\results_dfnc_script';

%% *_dfnc.mat
dfnc_param = 'F:\The_first_training\results_dfnc_script\lc_dfnc.mat';

%% ICA parameter file 
ica_param_file = 'F:\The_first_training\results\lc_ica_parameter_info.mat';


%% Cell array of dimensions number of network names by 2. Don't duplicate components in different
% network names
comp_network_names = {'DMN', [52,70];                 
                      'AUD', [25,32,46,71];
                      'VIS',[10,13,67];
                      'LFPN',84;
                      'RFPN',68;
                      'CEN',48;
                      'SN',[58,91];
                      'DorsAttn',83;
                      'SMN',42;
                      };

% 58：扣带回

%% TR of the experiment
TR = 2.5;


%% dFNC params

% 1. tc_detrend - Detrend number used to remove the trends in timecourses.
% Options are 0, 1, 2 and 3.
% 2. tc_despike - Remove spikes from the timecourses. Options are 'yes' and
% 'no'.
% 3. tc_filter - High frequency cutoff.

% 4. a. tc_covariates.filesList - Include character array or cell array of
% covariates to be regressed out from the timecourses. Cell array is of
% length number of subjects * sessions by 1. The order of file names will be
% first subject sessions followed by second subject sessions and so on.
%    b.  tc_covariates.file_numbers - Enter scan numbers to include. Leave
%    empty if you want to select all.
%


% 5. Regularisation method - Options are 'none' and 'L1'. 
% 6. wsize - Window size (scans) 
% 7. window_alpha - Gaussian Window alpha value.
% 8. num_repetitions - No. of repetitions (L1 regularisation).

dfnc_params.tc_detrend = 3;
dfnc_params.tc_despike = 'yes';
dfnc_params.tc_filter = 0.15;

dfnc_params.tc_covariates.filesList = [];
dfnc_params.tc_covariates.file_numbers = [];

dfnc_params.method = 'none';
dfnc_params.wsize = 30;  % **重要的参数之一，滑动窗的大小**
dfnc_params.window_alpha = 3;
dfnc_params.num_repetitions = 10;

%% Post-processing (K-means on dfnc corrleations, meta state analysis)
% Number of clusters extracted from windowed dfnc correlations using standard dfnc approach
postprocess.num_clusters = 4;


% Meta state analysis
% Number of clusters/components extracted from windowed dfnc correlations using meta state analysis (Temporal ICA, spatial ICA, K-means, PCA)
postprocess.ica.num_comps = 4;

% ICA algorithm used in meta state analysis. 
% Options are 'Infomax', 'Fast ICA', 'Erica', 'Simbec', 'Evd', 'Jade Opac', 'Amuse', 'SDD ICA', 'Radical ICA', 'Combi', 'ICA-EBM', 'ERBM'
postprocess.ica.algorithm = 'infomax';
% Number of times ICA is run to get stable estimates using Minimum spanning
% tree algorithm
postprocess.ica.num_ica_runs = 5;

% Specify covariates to be regressed from windowed dFNC correlations. You
% could specify FNC MAT file (*mancovan*results*fnc.mat) containing UNI variable from mancovan or continuous covariates in single ascii file.  
postprocess.regressCovFile = '';

% Max number of iterations used in kmeans
postprocess.kmeans_max_iter = 150 ;
% Distance method. Options are 'City', 'sqEuclidean', 'Hamming', 'Correlation', 'Cosine'
postprocess.dmethod = 'city';


%% Save HTML report in directory html with suffix *dfnc*results*html
postprocess.display_results = 0;

%% Save it into *_dfnc.mat
load(dfnc_param)
dfncInfo.postprocess = postprocess;
[path, file, suffix] = fileparts(dfnc_param);
save(dfnc_param, 'dfncInfo');


