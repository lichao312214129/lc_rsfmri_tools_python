function cluster_hydra_for_zhangyue(varargin)
% CLUSTER_HYDRA_FOR_ZHANGYUE
% Usage 1: cluster_hydra_for_zhangyue('--patient_dir', $your_patient_dir, '--hc_dir', $your_hc_dir, '--feature_type', $your_feature_type)
% Usage 2: cluster_hydra_for_zhangyue('-pd', $your_patient_dir, '-hd', $your_hc_dir, '-ft', $your_feature_type)
% ----------------------------------------------------------------------------
% INPUT:
%       REQUIRED:
%           [--patient_dir, -pd]: directory of patients' data
%           [--hc_dir,-hd]: directory of healthy controls' data
%           [--feature_type, -ft]:  % 'fc' OR 'nii'
%       OPTIONAL:
%           [--patient_cov_file, -pcf]: csv file of patients' covariates
%           [--hc_cov_file, -hcf]: csv file of healthy controls' covariates
%           [--mask_file, -mf]: mask file, default is empty
%           [--is_pca, -ip]: if perform principal components analysis to reduce feature dimensions, default is 1
%           [--explained_cov, -ec]: how many explained variance to retain, default is 0.95, range = (0, 1]
%           [--min_clustering_solutions, -mincs] : the minimum number of clusters, default is 1
%           [--max_clustering_solutions, -maxcs]: the maximum number of clusters, default is 5
%           [--cvfold, -cf]: inner number of cross validation to find best cluster, default is 5
%           [--output_dir, -od]: output directory, default is current working directory
%
% OUTPUT:
%       CIDX: sub-clustering assignments of the disease population (positive
%           class).
%       ARI: adjusted rand index measuring the overlap/reproducibility of
%           clustering solutions across folds
%       subtype_index: cell array, each cell contains a index of one cluster
% EXAMPLE:
% cluster_hydra_for_zhangyue('-pd', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\data_clustering\p',...
%                           '-hd', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\data_clustering\c', '-ft', 'fc',...
%                           '-od', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\data_clustering',...
%                           '-pcf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\data_clustering\patient_cov.csv',...
%                           '-hcf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\data_clustering\hc_cov.csv'))
% ----------------------------------------------------------------------------
%   Please cite the article: Varol, Erdem, Aristeidis Sotiras, Christos Davatzikos.
%   "HYDRA: Revealing heterogeneity of imaging and genetic patterns
%   through a multiple max-margin discriminative analysis framework."
%   NeuroImage 145 (2017): 346-364.
%   Because the dimensions of the image data is very high, so I use PCA to reduce its dimensions.
%   @author: Li Chao
%   Email: lichao19870617@gmail.com


% DEBUG
if nargin == 0
    help cluster_hydra_for_zhangyue
    return
end

% Init
patient_dir = '';
hc_dir = '';
feature_type = '';  % 'fc' OR 'nii'

patient_cov_file = '';
hc_cov_file = '';
mask_file = '';  
output_dir = pwd;
is_pca = 1;  
explained_cov = 0.95; 
min_clustering_solutions = 1; 
max_clustering_solutions = 3; 
cvfold = 3; 

% Varargin parser
if( sum(or(strcmpi(varargin,'--patient_dir'),strcmpi(varargin,'-pd')))==1)
    patient_dir = varargin{find(or(strcmpi(varargin,'--patient_dir'),strcmp(varargin,'-pd')))+1};
else
    error('Please specify patient_dir!');
end

if( sum(or(strcmpi(varargin,'--hc_dir'),strcmpi(varargin,'-hd')))==1)
    hc_dir = varargin{find(or(strcmpi(varargin,'--hc_dir'),strcmp(varargin,'-hd')))+1};
else
    error('Please specify hc_dir!');
end

if( sum(or(strcmpi(varargin,'--feature_type'),strcmpi(varargin,'-ft')))==1)
    feature_type = varargin{find(or(strcmpi(varargin,'--feature_type'),strcmp(varargin,'-ft')))+1};
else
    error('Please specify feature_type!');
end

if( sum(or(strcmpi(varargin,'--patient_cov_file'),strcmpi(varargin,'-pcf')))==1)
    patient_cov_file = varargin{find(or(strcmpi(varargin,'--patient_cov_file'),strcmp(varargin,'-pcf')))+1};
end

if( sum(or(strcmpi(varargin,'--hc_cov_file'),strcmpi(varargin,'-hcf')))==1)
    hc_cov_file = varargin{find(or(strcmpi(varargin,'--hc_cov_file'),strcmp(varargin,'-hcf')))+1};
end

if( sum(or(strcmpi(varargin,'--mask_file'),strcmpi(varargin,'-mf')))==1)
    mask_file = varargin{find(or(strcmpi(varargin,'--mask_file'),strcmp(varargin,'-mf')))+1};
end

if( sum(or(strcmpi(varargin,'--output_dir'),strcmpi(varargin,'-ud')))==1)
    output_dir = varargin{find(or(strcmpi(varargin,'--output_dir'),strcmp(varargin,'-ud')))+1};
end


if( sum(or(strcmpi(varargin,'--is_pca'),strcmpi(varargin,'-ip')))==1)
    is_pca = varargin{find(or(strcmpi(varargin,'--is_pca'),strcmp(varargin,'-ip')))+1};
end

if( sum(or(strcmpi(varargin,'--explained_cov'),strcmpi(varargin,'-ec')))==1)
    explained_cov = varargin{find(or(strcmpi(varargin,'--explained_cov'),strcmp(varargin,'-ec')))+1};
end

if( sum(or(strcmpi(varargin,'--min_clustering_solutions'),strcmpi(varargin,'-mincs')))==1)
   min_clustering_solutions = varargin{find(or(strcmpi(varargin,'--min_clustering_solutions'),strcmp(varargin,'-mincs')))+1};
end

if( sum(or(strcmpi(varargin,'--max_clustering_solutions'),strcmpi(varargin,'-maxcs')))==1)
   max_clustering_solutions = varargin{find(or(strcmpi(varargin,'--max_clustering_solutions'),strcmp(varargin,'-maxcs')))+1};
end

if( sum(or(strcmpi(varargin,'--cvfold'),strcmpi(varargin,'-cf')))==1)
   cvfold = varargin{find(or(strcmpi(varargin,'--cvfold'),strcmp(varargin,'-cf')))+1};
end

% Get image files path
patient_struct = dir(patient_dir);
patient_name = {patient_struct.name}';
patient_name = patient_name(3:end);
patient_path = fullfile(patient_dir, patient_name);

hc_struct = dir(hc_dir);
hc_name = {hc_struct.name}';
hc_name = hc_name(3:end);
hc_path = fullfile(hc_dir, hc_name);

% Load all data
num_patient = length(patient_path);
num_hc = length(hc_path);

if strcmp(feature_type, 'fc')
    d_tmp_patients = importdata(patient_path{1});
    d_tmp_hc = importdata(hc_path{1});
elseif strcmp(feature_type, 'nii')
    d_tmp_patients = y_Read(patient_path{1});
    d_tmp_hc = y_Read(hc_path{1});
else
    fprintf('Unsupport feature type %s\n',feature_type);
    return
end
 
% Load cov
if ~strcmp(patient_cov_file, '')
    patient_cov = importdata(patient_cov_file);
    patient_cov = patient_cov.data;
else
    patient_cov = [];
end
if ~strcmp(hc_cov_file, '')
    hc_cov = importdata(hc_cov_file);
    hc_cov = hc_cov.data;
else
    hc_cov = [];
end
all_cov = cat(1, patient_cov, hc_cov);

% Mask
if ~strcmp(mask_file, '')
    if strcmp(feature_type, 'fc')
        mask = importdata(mask_file) ~= 0;
        mask(tril(ones(size(mask))) == 1) = 0;
    else
        mask = y_Read(mask_file) ~= 0;
    end
else
    mask = ones(size(d_tmp_patients));
    mask(tril(ones(size(mask))) == 1) = 0;
end
mask = mask == 1;

% Check
if (~all(size(d_tmp_patients) == size(d_tmp_hc)))
    disp('Dimension of the patients and HCs are different');
    return;
end

if  (~all(size(mask) == size(d_tmp_hc)))
    disp('Dimension of the mask and data are different');
    return;
end

if length(patient_cov) ~= length(num_patient)
    error('Number of covariates is not match number of patients!');
    return;
end

if length(hc_cov) ~= length(num_hc)
    error('Number of covariates is not match number of hc!');
    return;
end

% Pre-allocate
% If fc, then only extract triu
data_patient = zeros(num_patient,sum(mask(:)));
data_hc = zeros(num_hc,sum(mask(:)));

% Load
for i = 1:num_patient
    if strcmp(feature_type, 'fc')
        data = importdata(patient_path{i});
    else
        data = y_Read(patient_path{i});
    end
    data = data(mask);
    data_patient(i, :) = data;
end

for i = 1:num_hc
    if strcmp(feature_type, 'fc')
        data = importdata(hc_path{i});
    else
        data = y_Read(hc_path{i});
    end
    data = data(mask);
    data_hc(i, :) = data;
end

% De-nan and de-inf
data_patient(isnan(data_patient)) = 0;
data_patient(isinf(data_patient)) = 1;
data_hc(isnan(data_hc)) = 0;
data_hc(isinf(data_hc)) = 1;

% Concat data
data_all = cat(1,  data_patient, data_hc);

% Generate unique ID and label
label = [ones(num_patient, 1); zeros(num_hc, 1) - 1];
subj = cat(1, patient_name, hc_name);
ms = regexp(subj, '(?<=\w+)[1-9][0-9]*', 'match' );
nms = length(ms);
subjid = zeros(nms,1);
for i = 1:nms
    if isempty(ms{i})
        tmp = ['99999',num2str(i)];
    else
        tmp = ms{i}{1};
    end
    subjid(i) = str2double(tmp);
end

% Give subject idx to cov and save to csv
if ~isempty(all_cov)
    if length(all_cov) ~= length(subjid)
        error('number of cov is not match the data!')
    else
        all_cov = cat(2, subjid, all_cov);
        csvwrite(fullfile(output_dir, 'cov_tmp.csv'), all_cov);
    end
end
% PCA
if is_pca
    [COEFF, data_all_reduced,~,~,explained] = pca(data_all);
    n_comp = numel(explained);
    cum_ex_list = zeros(n_comp, 1);
    cum_ex = 0;
    for i = 1:n_comp
        cum_ex = cum_ex + explained(i);
        cum_ex_list(i) = cum_ex;
    end
    loc_cutoff_cum_ex = find(cum_ex_list >= explained_cov*100);
    loc_cutoff_cum_ex = loc_cutoff_cum_ex(1);
    data_all_reduced = data_all_reduced(:,1:loc_cutoff_cum_ex);
else
    data_all_reduced = data_all;
end

% save to csv
data_to_csv = cat(2, subjid, data_all_reduced, label);
csvwrite(fullfile(output_dir, 'cluster_tmp.csv'), data_to_csv);

% Run HYDRA
if ~isempty(all_cov)
    hydra('-i', fullfile(output_dir, 'cluster_tmp.csv'), '-z', fullfile(output_dir, 'cov_tmp.csv'), '-o', output_dir, '-m', min_clustering_solutions, '-k', max_clustering_solutions, '-f', cvfold);
else
    hydra('-i', fullfile(output_dir, 'cluster_tmp.csv'), '-o', output_dir, '-m', min_clustering_solutions, '-k', max_clustering_solutions, '-f', cvfold);
end

% Get sub-type uid
load(fullfile(output_dir, 'HYDRA_results.mat'));
idx = CIDX(:, ARI == max(ARI));
uid_subtype = setdiff(unique(idx),  -1);
n_subtype = numel(uid_subtype);
subtype_index = cell(n_subtype, 1);
for i =  1: n_subtype
    subtype_index{i} = ID(idx == uid_subtype(i));
end
save( fullfile(output_dir, 'subtype_index.mat'), 'subtype_index');
end

function [CIDX,ARI] = hydra(varargin)
%  HYDRA
%  Version 1.0.0 --- January 2018
%  Section of Biomedical Image Analysis
%  Department of Radiology
%  University of Pennsylvania
%  Richard Building
%  3700 Hamilton Walk, 7th Floor
%  Philadelphia, PA 19104
%
%  Web:   https://www.med.upenn.edu/sbia/
%  Email: sbia-software at uphs.upenn.edu
%
%  Copyright (c) 2018 University of Pennsylvania. All rights reserved.
%  See https://www.med.upenn.edu/sbia/software-agreement.html or COPYING file.
%
%  Author:
%  Erdem Varol
%  software@cbica.upenn.edu

if nargin==0
    printhelp()
    return
end

if( strcmp(varargin{1},'--help') || isempty(varargin))
    printhelp()
    return;
end

if( strcmp(varargin{1},'-h') || isempty(varargin) )
    printhelp()
    return
end

if( strcmp(varargin{1},'--version') || isempty(varargin) )
    fprintf('Version 1.0.\n')
    return
end

if( strcmp(varargin{1},'-v') || isempty(varargin) )
    fprintf('Version 1.0.\n')
    return
end

if( strcmp(varargin{1},'-u') || isempty(varargin) )
    fprintf(' EXAMPLE USE (in matlab) \n');
    fprintf(' hydra(''-i'',''test.csv'',''-o'',''.'',''-k'',3,''-f'',3) \n');
    fprintf(' EXAMPLE USE (in command line) \n');
    fprintf(' hydra -i test.csv -o . -k 3 -f 3 \n');
    return
end

if( strcmp(varargin{1},'--usage') || isempty(varargin) )
    fprintf(' EXAMPLE USE (in matlab) \n');
    fprintf(' hydra(''-i'',''test.csv'',''-o'',''.'',''-k'',3,''-f'',3) \n');
    fprintf(' EXAMPLE USE (in command line) \n');
    fprintf(' hydra -i test.csv -o . -k 3 -f 3 \n');
    return
end

% function returns estimated subgroups by hydra for clustering
% configurations ranging from K=1 to K=10, or another specified range of
% values. The function returns also the Adjusted Rand Index that was
% calculated across the cross-validation experiments and comparing
% respective clustering solutions.
%
% INPUT
%
% REQUIRED
% [--input, -i] : .csv file containing the input features. (REQUIRED)
%              every column of the file contains values for a feature, with
%              the exception of the first and last columns. We assume that
%              the first column contains subject identifying information
%              while the last column contains label information. First line
%              of the file should contain header information. Label
%              convention: -1 -> control group - 1 -> pathological group
%              that will be partioned to subgroups
% [--outputDir, -o] : directory where the output from all folds will be saved (REQUIRED)
%
% OPTIONAL
%
% [--covCSV, -z] : .csv file containing values for different covariates, which
%           will be used to correct the data accordingly (OPTIONAL). Every
%           column of the file contains values for a covariate, with the
%           exception of the first column, which contains subject
%           identifying information. Correction is performed by solving a
%           solving a least square problem to estimate the respective
%           coefficients and then removing their effect from the data. The
%           effect of ALL provided covariates is removed. If no file is
%           specified, no correction is performed.
%
% NOTE: featureCSV and covCSV files are assumed to have the subjects given
%       in the same order in their rows
%
% [--c, -c] : regularization parameter (positive scalar). smaller values produce
%     sparser models (OPTIONAL - Default 0.25)
% [--reg_type, -r] : determines regularization type. 1 -> promotes sparsity in the
%            estimated hyperplanes - 2 -> L2 norm (OPTIONAL - Default 1)
% [--balance, -b] : takes into account differences in the number between the two
%           classes. 1-> in case there is mismatch between the number of
%           controls and patient - 0-> otherwise (OPTIONAL - Default 1)
% [--init, -g] : initialization strategy. 0 : assignment by random hyperplanes
%        (not supported for regression), 1 : pure random assignment, 2:
%        k-means assignment, 3: assignment by DPP random
%        hyperplanes (default)
% [--iter, -t] : number of iterations between estimating hyperplanes, and cluster
%        estimation. Default is 50. Increase if algorithms fails to
%        converge
% [--numconsensus, -n] : number of clustering consensus steps. Default is 20.
%                Increase if algorithm gives unstable clustering results.
% [--kmin, -m] : determines the range of clustering solutions to evaluate
%             (i.e., kmin to kmax). Default  value is 1.
% [--kmax, -k] : determines the range of clustering solutions to evaluate
%             (i.e., kmin to kmax). Default  value is 10.
% [--kstep, -s] : determines the range of clustering solutions to evaluate
%             (i.e., kmin to kmax, with step kstep). Default  value is 1.
% [--cvfold, -f]: number of folds for cross validation. Default value is 10.
% [--vo, -j] : verbose output (i.e., also saves input data to verify that all were
%      read correctly. Default value is 0
% [--usage, -u]  Prints basic usage message.          
% [--help, -h]  Prints help information.
% [--version, -v]  Prints information about software version.
%
% OUTPUT:
% CIDX: sub-clustering assignments of the disease population (positive
%       class).
% ARI: adjusted rand index measuring the overlap/reproducibility of
%      clustering solutions across folds
%
% NOTE: to compile this function do
% mcc -m  hydra.m
%
%
% EXAMPLE USE (in matlab)
% hydra('-i','test.csv','-o','.','-k',3,'-f',3);
% EXAMPLE USE (in command line)
% hydra -i test.csv -o . -k 3 -f 3


params.kernel=0;



if( sum(or(strcmpi(varargin,'--input'),strcmpi(varargin,'-i')))==1)
    featureCSV=varargin{find(or(strcmpi(varargin,'--input'),strcmp(varargin,'-i')))+1};
else
    error('hydra:argChk','Please specify input csv file!');
end


if( sum(or(strcmpi(varargin,'--outputDir'),strcmpi(varargin,'-o')))==1)
    outputDir=varargin{find(or(strcmp(varargin,'--outputDir'),strcmp(varargin,'-o')))+1};
else
    error('hydra:argChk','Please specify output directory!');
end


if( sum(or(strcmpi(varargin,'--cov'),strcmpi(varargin,'-z')))==1)
    covCSV=varargin{find(or(strcmpi(varargin,'--cov'),strcmp(varargin,'-z')))+1};
else
    covCSV=[];
end

if( sum(or(strcmpi(varargin,'--c'),strcmpi(varargin,'-c')))==1)
    params.C=varargin{find(or(strcmpi(varargin,'--c'),strcmp(varargin,'-c')))+1};
else
    params.C=0.25;
end

if( sum(or(strcmpi(varargin,'--reg_type'),strcmpi(varargin,'-r')))==1)
    params.reg_type=varargin{find(or(strcmpi(varargin,'--reg_type'),strcmp(varargin,'-r')))+1};
else
    params.reg_type=1;
end

if( sum(or(strcmpi(varargin,'--balance'),strcmpi(varargin,'-b')))==1)
    params.balanceclasses=varargin{find(or(strcmpi(varargin,'--balance'),strcmp(varargin,'-b')))+1};
else
    params.balanceclasses=1;
end

if( sum(or(strcmpi(varargin,'--init'),strcmpi(varargin,'-g')))==1)
    params.init_type=varargin{find(or(strcmpi(varargin,'--init'),strcmp(varargin,'-g')))+1};
else
    params.init_type=3;
end

if( sum(or(strcmpi(varargin,'--iter'),strcmpi(varargin,'-t')))==1)
    params.numiter=varargin{find(or(strcmpi(varargin,'--iter'),strcmp(varargin,'-t')))+1};
else
    params.numiter=50;
end

if( sum(or(strcmpi(varargin,'--numconsensus'),strcmpi(varargin,'-n')))==1)
    params.numconsensus=varargin{find(or(strcmpi(varargin,'--numconsensus'),strcmp(varargin,'-n')))+1};
else
    params.numconsensus=20;
end

if( sum(or(strcmpi(varargin,'--kmin'),strcmpi(varargin,'-m')))==1)
    params.kmin=varargin{find(or(strcmpi(varargin,'--kmin'),strcmp(varargin,'-m')))+1};
else
    params.kmin=1;
end

if( sum(or(strcmpi(varargin,'--kstep'),strcmpi(varargin,'-s')))==1)
    params.kstep=varargin{find(or(strcmpi(varargin,'--kstep'),strcmp(varargin,'-s')))+1};
else
    params.kstep=1;
end

if( sum(or(strcmpi(varargin,'--kmax'),strcmpi(varargin,'-k')))==1)
    params.kmax=varargin{find(or(strcmpi(varargin,'--kmax'),strcmp(varargin,'-k')))+1};
else
    params.kmax=10;
end

if( sum(or(strcmpi(varargin,'--cvfold'),strcmpi(varargin,'-f')))==1)
    params.cvfold=varargin{find(or(strcmpi(varargin,'--cvfold'),strcmp(varargin,'-f')))+1};
else
    params.cvfold=10;
end

if( sum(or(strcmpi(varargin,'--vo'),strcmpi(varargin,'-j')))==1)
    params.vo=varargin{find(or(strcmpi(varargin,'--vo'),strcmp(varargin,'-j')))+1};
else
    params.vo=0;
end

% create output directory
if (~exist(outputDir,'dir'))
    [status,~,~] = mkdir(outputDir);
    if (status == 0)
        error('hydra:argChk','Cannot create output directory!');
    end
end


params.C=input2num(params.C);
params.reg_type=input2num(params.reg_type);
params.balanceclasses=input2num(params.balanceclasses);
params.init_type=input2num(params.init_type);
params.numiter=input2num(params.numiter);
params.numconsensus=input2num(params.numconsensus);
params.kmin=input2num(params.kmin);
params.kstep=input2num(params.kstep);
params.kmax=input2num(params.kmax);
params.cvfold=input2num(params.cvfold);
params.vo=input2num(params.vo);


% confirm validity of optional input arguments
validateFcn_reg_type = @(x) (x==1) || (x == 2);
validateFcn_balance = @(x) (x==0) || (x == 1);
validateFcn_init = @(x) (x==0) || (x == 1) || (x==2) || (x == 3) || (x == 4);
validateFcn_iter = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_consensus = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_kmin = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_kmax = @(x,y) isscalar(x) && (x>0) && (mod(x,1)==0) && (x>y);
validateFcn_kstep = @(x,y,z) isscalar(x) && (x>0) && (mod(x,1)==0) && (x+y<z);
validateFcn_cvfold = @(x) isscalar(x) && (x>0) && (mod(x,1)==0);
validateFcn_vo = @(x) (x==0) || (x == 1);

if(~validateFcn_reg_type(params.reg_type))
    error('hydra:argChk','Input regularization type (reg_type) should be either 1 or 2!');
end
if(~validateFcn_balance(params.balanceclasses))
    error('hydra:argChk','Input balance classes (balance) should be either 1 or 2!');
end
if(~validateFcn_init(params.init_type))
    error('hydra:argChk','Initialization type can be either 0, 1, 2, 3, or 4!');
end
if(~validateFcn_iter(params.numiter))
    error('hydra:argChk','Number of iterations should be a positive integer!');
end
if(~validateFcn_consensus(params.numconsensus))
    error('hydra:argChk','Number of clustering consensus steps should be a positive integer!');
end
if(~validateFcn_kmin(params.kmin))
    error('hydra:argChk','Minimum number of clustering solutions to consider should be a positive integer!');
end
if(~validateFcn_kmax(params.kmax,params.kmin))
    error('hydra:argChk','Maximum number of clustering solutions to consider should be a positive integer that is greater than the minimum number of clustering solutions!');
end
if(~validateFcn_kstep(params.kstep,params.kmin,params.kmax))
    error('hydra:argChk','Step number of clustering solutions to consider should be a positive integer that is between the minimun and maximum number of clustering solutions!');
end
if(~validateFcn_cvfold(params.cvfold))
    error('hydra:argChk','Number of folds for cross-validation should be a positive integer!');
end
if(~validateFcn_vo(params.vo))
    error('hydra:argChk','VO parameter should be either 0 or 1!');
end

disp('Done');
disp('HYDRA runs with the following parameteres');
disp(['featureCSV: ' featureCSV]);
disp(['OutputDir: ' outputDir]);
disp(['covCSV: ' covCSV])
disp(['C: ' num2str(params.C)]);
disp(['reg_type: ' num2str(params.reg_type)]);
disp(['balanceclasses: ' num2str(params.balanceclasses)]);
disp(['init_type: ' num2str(params.init_type)]);
disp(['numiter: ' num2str(params.numiter)]);
disp(['numconsensus: ' num2str(params.numconsensus)]);
disp(['kmin: ' num2str(params.kmin)]);
disp(['kmax: ' num2str(params.kmax)]);
disp(['kstep: ' num2str(params.kstep)]);
disp(['cvfold: ' num2str(params.cvfold)]);
disp(['vo: ' num2str(params.vo)]);

% csv with features
fname=featureCSV;
if (~exist(fname,'file'))
    error('hydra:argChk','Input feature .csv file does not exist');
end

% csv with features
covfname=covCSV;
if(~isempty(covfname))
    if(~exist(covfname,'file'))
        error('hydra:argChk','Input covariate .csv file does not exist');
    end
end

% input data
% assumption is that the first column contains IDs, and the last contains
% labels
disp('Loading features...');
input=readtable(fname);
ID=input{:,1};
XK=input{:,2:end-1};
Y=input{:,end};

% z-score imaging features
XK=zscore(XK);
disp('Done');

% input covariate information if necesary
if(~isempty(covfname))
    disp('Loading covariates...');
    covardata = readtable(covfname) ;
    IDcovar = covardata{:,1};
    covar = covardata{:,2:end};
    covar = zscore(covar);
    disp('Done');
end

% NOTE: we assume that the imaging data and the covariate data are given in
% the same order. No test is performed to check that. By choosing to have a
% verbose output, you can have access to the ID values are read by the
% software for both the imaging data and the covariates

% verify that we have covariate data and imaging data for the same number
% of subjects
if(~isempty(covfname))
    if(size(covar,1)~=size(XK,1))
        error('hydra:argChk','The feature .csv and covariate .csv file contain data for different number of subjects');
    end
end

% residualize covariates if necessary
if(~isempty(covfname))
    disp('Residualize data...');
    [XK0,~]=GLMcorrection(XK,Y,covar,XK,covar);
    disp('Done');
else
    XK0=XK;
end

% for each realization of cross-validation
clustering=params.kmin:params.kstep:params.kmax;
part=make_xval_partition(size(XK0,1),params.cvfold); %Partition data to 10 groups for cross validation
% for each fold of the k-fold cross-validation
disp('Run HYDRA...');
for f=1:params.cvfold
    % for each clustering solution
    for kh=1:length(clustering)
        params.k=clustering(kh);
        disp(['Applying HYDRA for ' num2str(params.k) ' clusters. Fold: ' num2str(f) '/' num2str(params.cvfold)]);
        model=hydra_solver(XK0(part~=f,:),Y(part~=f,:),[],params);
        YK{kh}(part~=f,f)=model.Yhat;
    end
end
disp('Done');

disp('Estimating clustering stabilitiy...')
% estimate cluster stability for the cross-validation experiment
ARI = zeros(length(clustering),1);
for kh=1:length(clustering)
    tmp=cv_cluster_stability(YK{kh}(Y~=-1,:));
    ARI(kh)=tmp(1);
end
disp('Done')

disp('Estimating final consensus group memberships...')
% Computing final consensus group memberships
CIDX=-ones(size(Y,1),length(clustering)); %variable that stores subjects in rows, and cluster memberships for the different clustering solutions in columns
for kh=1:length(clustering)
    CIDX(Y==1,kh)=consensus_clustering_hydra_highlevel(YK{kh}(Y==1,:),clustering(kh));
end
disp('Done')

disp('Saving results...')
if(params.vo==0)
    save([outputDir '/HYDRA_results.mat'],'ARI','CIDX','clustering','ID');
else
    save([outputDir '/HYDRA_results.mat'],'ARI','CIDX','clustering','ID','XK','Y','covar','IDcovar');
end
disp('Done')
end

function [score,stdscore]=cv_cluster_stability(S)
k=0;
for i=1:size(S,2)-1
    for j=i+1:size(S,2)
        k=k+1;
        zero_idx=any([S(:,i) S(:,j)]==0,2);
        [a(k),b(k),c(k),d(k)]=RandIndex(S(~zero_idx,i),S(~zero_idx,j));
    end
end
score=[mean(a) mean(b) mean(c) mean(d)];
stdscore=[std(a) std(b) std(c) std(d)];
end

function [AR,RI,MI,HI]=RandIndex(c1,c2)
%RANDINDEX - calculates Rand Indices to compare two partitions
% ARI=RANDINDEX(c1,c2), where c1,c2 are vectors listing the
% class membership, returns the "Hubert & Arabie adjusted Rand index".
% [AR,RI,MI,HI]=RANDINDEX(c1,c2) returns the adjusted Rand index,
% the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
%
% See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of
% Classification 2:193-218

%(C) David Corney (2000)        D.Corney@cs.ucl.ac.uk

if nargin < 2 | min(size(c1)) > 1 | min(size(c2)) > 1
    error('RandIndex: Requires two vector arguments')
    return
end

C=Contingency(c1,c2);   %form contingency matrix

n=sum(sum(C));
nis=sum(sum(C,2).^2);       %sum of squares of sums of rows
njs=sum(sum(C,1).^2);       %sum of squares of sums of columns

t1=nchoosek(n,2);       %total number of pairs of entities
t2=sum(sum(C.^2));  %sum over rows & columnns of nij^2
t3=.5*(nis+njs);

%Expected index (for adjustment)
nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));

A=t1+t2-t3;     %no. agreements
D=  -t2+t3;     %no. disagreements

if t1==nc
    AR=0;           %avoid division by zero; if k=1, define Rand = 0
else
    AR=(A-nc)/(t1-nc);      %adjusted Rand - Hubert & Arabie 1985
end

RI=A/t1;            %Rand 1971      %Probability of agreement
MI=D/t1;            %Mirkin 1970    %p(disagreement)
HI=(A-D)/t1;    %Hubert 1977    %p(agree)-p(disagree)

    function Cont=Contingency(Mem1,Mem2)
        
        if nargin < 2 | min(size(Mem1)) > 1 | min(size(Mem2)) > 1
            error('Contingency: Requires two vector arguments')
            return
        end
        
        Cont=zeros(max(Mem1),max(Mem2));
        
        for i = 1:length(Mem1);
            Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
        end
    end
end

function IDXfinal=consensus_clustering_hydra_highlevel(IDX,k)
[n,~]=size(IDX);
cooc=zeros(n);
for i=1:n-1
    for j=i+1:n
        cooc(i,j)=sum(IDX(i,:)==IDX(j,:));
    end
    %cooc(i,i)=sum(IDX(i,:)==IDX(i,:))/2;
end
cooc=cooc+cooc';
L=diag(sum(cooc,2))-cooc;

Ln=eye(n)-diag(sum(cooc,2).^(-1/2))*cooc*diag(sum(cooc,2).^(-1/2));
Ln(isnan(Ln))=0;
[V,~]=eig(Ln);
try
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
catch
    disp('Complex Eigenvectors Found...Using Non-Normalized Laplacian');
    [V,~]=eig(L);
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
end

end

function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

s=mod(n,n_folds);r=n-s;
p1=ceil((1:r)/ceil(r/n_folds));
p2=randperm(n_folds);p2=p2(1:s);
p3=[p1 p2];
part=p3(randperm(size(p3,2)));
end

function [X0train,X0test]=GLMcorrection(Xtrain,Ytrain,covartrain,Xtest,covartest)

X1=Xtrain(Ytrain==-1,:);
C1=covartrain(Ytrain==-1,:);
B=[C1 ones(size(C1,1),1)];
Z=X1'*B*inv(B'*B);
X0train=(Xtrain'-Z(:,1:end-1)*covartrain')';
X0test=(Xtest'-Z(:,1:end-1)*covartest')';
end

function printhelp()
fprintf(' function returns estimated subgroups by hydra for clustering \n')
fprintf(' configurations ranging from K=1 to K=10, or another specified range of\n')
fprintf(' values. The function returns also the Adjusted Rand Index that was\n')
fprintf(' calculated across the cross-validation experiments and comparing\n')
fprintf(' respective clustering solutions.\n')
fprintf('\n')
fprintf(' INPUT\n')
fprintf('\n')
fprintf(' REQUIRED\n')
fprintf(' [--input, -i] : .csv file containing the input features. (REQUIRED)\n')
fprintf('              every column of the file contains values for a feature, with\n')
fprintf('              the exception of the first and last columns. We assume that\n')
fprintf('              the first column contains subject identifying information\n')
fprintf('              while the last column contains label information. First line\n')
fprintf('              of the file should contain header information. Label\n')
fprintf('              convention: -1 -> control group - 1 -> pathological group\n')
fprintf('              that will be partioned to subgroups\n')
fprintf(' [--outputDir, -o] : directory where the output from all folds will be saved (REQUIRED)\n')
fprintf('\n')
fprintf(' OPTIONAL\n')
fprintf('\n')
fprintf(' [--covCSV, -z] : .csv file containing values for different covariates, which\n')
fprintf('           will be used to correct the data accordingly (OPTIONAL). Every\n')
fprintf('           column of the file contains values for a covariate, with the\n')
fprintf('           exception of the first column, which contains subject\n')
fprintf('           identifying information. Correction is performed by solving a\n')
fprintf('           solving a least square problem to estimate the respective\n')
fprintf('           coefficients and then removing their effect from the data. The\n')
fprintf('           effect of ALL provided covariates is removed. If no file is\n')
fprintf('           specified, no correction is performed.\n')
fprintf('\n')
fprintf(' NOTE: featureCSV and covCSV files are assumed to have the subjects given\n')
fprintf('       in the same order in their rows\n')
fprintf('\n')
fprintf(' [--c, -c] : regularization parameter (positive scalar). smaller values produce\n')
fprintf('     sparser models (OPTIONAL - Default 0.25)\n')
fprintf(' [--reg_type, -r] : determines regularization type. 1 -> promotes sparsity in the\n')
fprintf('            estimated hyperplanes - 2 -> L2 norm (OPTIONAL - Default 1)\n')
fprintf(' [--balance, -b] : takes into account differences in the number between the two\n')
fprintf('           classes. 1-> in case there is mismatch between the number of\n')
fprintf('           controls and patient - 0-> otherwise (OPTIONAL - Default 1)\n')
fprintf(' [--init, -g] : initialization strategy. 0 : assignment by random hyperplanes\n')
fprintf('        (not supported for regression), 1 : pure random assignment, 2:\n')
fprintf('        k-means assignment, 3: assignment by DPP random\n')
fprintf('        hyperplanes (default)\n')
fprintf(' [--iter, -t] : number of iterations between estimating hyperplanes, and cluster\n')
fprintf('        estimation. Default is 50. Increase if algorithms fails to\n')
fprintf('        converge\n')
fprintf(' [--numconsensus, -n] : number of clustering consensus steps. Default is 20.\n')
fprintf('                Increase if algorithm gives unstable clustering results.\n')
fprintf(' [--kmin, -m] : determines the range of clustering solutions to evaluate\n')
fprintf('             (i.e., kmin to kmax). Default  value is 1.\n')
fprintf(' [--kmax, -k] : determines the range of clustering solutions to evaluate\n')
fprintf('             (i.e., kmin to kmax). Default  value is 10.\n')
fprintf(' [--kstep, -s] : determines the range of clustering solutions to evaluate\n')
fprintf('             (i.e., kmin to kmax, with step kstep). Default  value is 1.\n')
fprintf(' [--cvfold, -f]: number of folds for cross validation. Default value is 10.\n')
fprintf(' [--vo, -j] : verbose output (i.e., also saves input data to verify that all were\n')
fprintf('      read correctly. Default value is 0\n')
fprintf(' [--usage, -u]  Prints basic usage message.     \n');     
fprintf(' [--help, -h]  Prints help information.\n');
fprintf(' [--version, -v]  Prints information about software version.\n');
fprintf('\n')
fprintf(' OUTPUT:\n')
fprintf(' CIDX: sub-clustering assignments of the disease population (positive\n')
fprintf('       class).\n')
fprintf(' ARI: adjusted rand index measuring the overlap/reproducibility of\n')
fprintf('      clustering solutions across folds\n')
fprintf('\n')
fprintf(' NOTE: to compile this function do\n')
fprintf(' mcc -m  hydra.m\n')
fprintf('\n')
fprintf('\n')
fprintf(' EXAMPLE USE (in matlab)\n')
fprintf(' hydra(''-i'',''test.csv'',''-o'',''.'',''-k'',3,''-f'',3);\n')
fprintf(' EXAMPLE USE (in command line)\n')
fprintf(' hydra -i test.csv -o . -k 3 -f 3\n')
fprintf('======================================================================\n');
fprintf('Contact: software@cbica.upenn.edu\n');
fprintf('\n');
fprintf('Copyright (c) 2018 University of Pennsylvania. All rights reserved.\n');
fprintf('See COPYING file or http://www.med.upenn.edu/sbia/software/license.html\n');
fprintf('======================================================================\n');
end

function o=input2num(x)
if isnumeric(x)
    o=x;
else
    o = str2double(x);
end
end


%% ===================================================================================
%  HYDRA SOLVER
%  Version 1.0.0 --- January 2018
%  Section of Biomedical Image Analysis
%  Department of Radiology
%  University of Pennsylvania
%  Richard Building
%  3700 Hamilton Walk, 7th Floor
%  Philadelphia, PA 19104
%
%  Web:   https://www.med.upenn.edu/sbia/
%  Email: sbia-software at uphs.upenn.edu
%
%  Copyright (c) 2018 University of Pennsylvania. All rights reserved.
%  See https://www.med.upenn.edu/sbia/software-agreement.html or COPYING file.
%
%  Author:
%  Erdem Varol
%  software@cbica.upenn.edu


function model=hydra_solver(XK,Y,Cov,params);
%% Parameters:
% numconsensus -- (int>=0) 0 for no consensus, positive integer for number of consensus
% runs
% numiter -- (int>0) number of iterative assignment steps
% C -- (real>0) loss penalty
% k -- (int>0) number of polytope faces (final number may be less due to
% face dropping)
% kernel -- (0 (default) or 1), treat XK as X*X' solve dual problem (1), else XK is X
% solve primal(0)
% init_type -- 0 : assignment by random hyperplanes (not supported for regression), 1 : pure random
% assignment, 2: k-means assignment (default), 3: assignment by DPP random
% hyperplanes
% reg_type -- (1 or 2): 1 solves L1-SVM, 2 solves L2-SVM
%% parameters
if ~isfield(params,'numconsensus')
    params.numconsensus=50;
end
if ~isfield(params,'numiter')
    params.numiter=20;
end
if ~isfield(params,'C')
    params.C=1;
end
if ~isfield(params,'k')
    params.k=1;
end
if ~isfield(params,'kernel')
    params.kernel=0;
end
if ~isfield(params,'init_type')
    params.init_type=2;
end
if ~isfield(params,'balanceclasses')
    params.balanceclasses=0;
end
if ~isfield(params,'fixedclustering')
    params.fixedclustering=0;
end
if ~isfield(params,'fixedclusteringIDX')
    params.fixedclusteringIDX=ones(size(XK,1),1);
end
if ~isfield(params,'reg_type');
    params.reg_type=2;
end

params.type='classification';
initparams.init_type=params.init_type;
%% algorithms


switch params.type
    case 'classification'
        initparams.regression=0;
        if params.fixedclustering==1
            params.k=numel(unique(params.fixedclusteringIDX(Y==1,1)));
            [~,~,params.fixedclusteringIDX(Y==1,1)]=unique(params.fixedclusteringIDX(Y==1,1));
        end
        
    %option for l2-regularization (default)
        if params.reg_type==2;
                if params.kernel==0
                    svmX=XK;
                    svmparams='-t 0';
                    initparams.kernel=0;
                elseif params.kernel==1
                    svmX=[(1:size(XK,1))' XK];
                    svmparams='-t 4';
                    initparams.kernel=1;
                end
                
                if params.fixedclustering==0
                    IDX=zeros(size(Y(Y==1,:),1),params.numconsensus);
                    for tt=1:params.numconsensus

            %Initialization
                        W=ones(size(Y,1),params.k)/params.k;
                        W(Y==1,:)=hydra_init_v2(XK,Y,params.k,initparams);
                        S=zeros(size(W));
                        cn=zeros(1,params.k);cp=zeros(1,params.k);nrm=zeros(1,params.k);
                        for t=1:params.numiter
                            for j=1:params.k
                %Weights for negative and positive samples
                                cn(1,j)=1./mean(W(Y==-1,j),1);
                                cp(1,j)=1./mean(W(Y==1,j),1);
                                nrm(1,j)=cn(1,j)+cp(1,j);
                                cn(1,j)=cn(1,j)/nrm(1,j);
                                cp(1,j)=cp(1,j)/nrm(1,j);

                                if params.balanceclasses==1
                    %Weighted svm taking into account negative/positive imbalance to solve for polytope hyperplanes
                                    mdl{j}=w_svmtrain(XK,Y,W(:,j),params.C,cp(1,j),cn(1,j),params.kernel);
                                else
                    %Unweighted svm to solve for polytope hyperplanes
                                    mdl{j}=w_svmtrain(XK,Y,W(:,j),params.C,1,1,params.kernel);
                                end
                %Solving subject projection score along each face of the polytope
                                S(:,j)=w_svmpredict(XK,mdl{j},params.kernel);
                            end
                %Subject assignment to the face of the polytope with maximum score
                            [~,idx]=max(S(Y==1,:),[],2);
                            Wold=W;
                            W(Y==1,:)=0;
                            W(sub2ind(size(W),find(Y==1),idx))=1;
                            if norm(W-Wold,'fro')<1e-6;
                                disp('converged');
                                break
                            end
                        end
                        IDX(:,tt)=idx;
                        
                    end
                    
            %Consensus steps, solving the assignments multiple times for stability
                    if params.numconsensus>1
                        IDXfinal=consensus_clustering(IDX,params.k);
                        W=zeros(size(Y,1),params.k);
                        W(sub2ind(size(W),find(Y==1),IDXfinal))=1;
                        W(Y==-1,:)=1/params.k;
                        cn=zeros(1,params.k);cp=zeros(1,params.k);nrm=zeros(1,params.k);
                        for j=1:params.k
                            cn(1,j)=1./mean(W(Y==-1,j),1);
                            cp(1,j)=1./mean(W(Y==1,j),1);
                            nrm(1,j)=cn(1,j)+cp(1,j);
                            cn(1,j)=cn(1,j)/nrm(1,j);
                            cp(1,j)=cp(1,j)/nrm(1,j);
                            if params.balanceclasses==1
                                mdl{j}=w_svmtrain(XK,Y,W(:,j),params.C,cp(1,j),cn(1,j),params.kernel);
                            else
                                mdl{j}=w_svmtrain(XK,Y,W(:,j),params.C,1,1,params.kernel);
                            end
                        end
                        
                    else
                        IDXfinal=IDX;
                    end
                    
            %If using fixed clustering inputs, solve polytope once:
                elseif params.fixedclustering==1
                    IDXfinal=params.fixedclusteringIDX(Y==1,1);
                    W=zeros(size(Y,1),params.k);
                    W(sub2ind(size(W),find(Y==1),IDXfinal))=1;
                    W(Y==-1,:)=1/params.k;
                    cn=zeros(1,params.k);cp=zeros(1,params.k);nrm=zeros(1,params.k);
                    for j=1:params.k
                        cn(1,j)=1./mean(W(Y==-1,j),1);
                        cp(1,j)=1./mean(W(Y==1,j),1);
                        nrm(1,j)=cn(1,j)+cp(1,j);
                        cn(1,j)=cn(1,j)/nrm(1,j);
                        cp(1,j)=cp(1,j)/nrm(1,j);
                        if params.balanceclasses==1
                            mdl{j}=w_svmtrain(XK,Y,W(:,j),params.C,cp(1,j),cn(1,j),params.kernel);
                        else
                            mdl{j}=w_svmtrain(XK,Y,W(:,j),params.C,1,1,params.kernel);
                        end
                    end
                    
                end
%store models and clustering outputs
                model.mdl=mdl;
                model.S=W(Y==1,:);
                model.W=W;
                model.Yhat=Y;
                model.Yhat(Y==1)=IDXfinal;
                model.cn=cn;
                model.cp=cp;
        end
%Option for sparse regularization
        if params.reg_type==1
                if params.kernel==0
                    svmX=sparse(XK);
                    initparams.kernel=0;
                    svmparams='-B 1';
                elseif params.kernel==1
                    error('Kernel in sparse SVM not supported');
                end
                if params.fixedclustering==0
                    IDX=zeros(size(Y(Y==1,:),1),params.numconsensus);
                    for tt=1:params.numconsensus
                        W=ones(size(Y,1),params.k)/params.k;
                        W(Y==1,:)=hydra_init_v2(XK,Y,params.k,initparams);
                        S=zeros(size(W));
                        cn=zeros(1,params.k);cp=zeros(1,params.k);nrm=zeros(1,params.k);
                        for t=1:params.numiter
                            for j=1:params.k
                                cn(1,j)=1./mean(W(Y==-1,j),1);
                                cp(1,j)=1./mean(W(Y==1,j),1);
                                nrm(1,j)=cn(1,j)+cp(1,j);
                                cn(1,j)=cn(1,j)/nrm(1,j);
                                cp(1,j)=cp(1,j)/nrm(1,j);
                                if params.balanceclasses==1
                                    mdl{j}=w_train(XK,Y,W(:,j),params.C,cp(1,j),cn(1,j));
                                else
                                    mdl{j}=w_train(XK,Y,W(:,j),params.C,1,1);
                                end
                                S(:,j)=w_svmpredict(XK,mdl{j},0);
                            end
                            [~,idx]=max(S(Y==1,:),[],2);
                            Wold=W;
                            W(Y==1,:)=0;
                            W(sub2ind(size(W),find(Y==1),idx))=1;
                            if norm(W-Wold,'fro')<1e-6;
                                disp('converged');
                                break
                            end
                        end
                        IDX(:,tt)=idx;
                        
                    end
                    
                    if params.numconsensus>1
                        IDXfinal=consensus_clustering(IDX,params.k);
                        W=zeros(size(Y,1),params.k);
                        W(sub2ind(size(W),find(Y==1),IDXfinal))=1;
                        W(Y==-1,:)=1/params.k;
                        cn=zeros(1,params.k);cp=zeros(1,params.k);nrm=zeros(1,params.k);
                        for j=1:params.k
                            cn(1,j)=1./mean(W(Y==-1,j),1);
                            cp(1,j)=1./mean(W(Y==1,j),1);
                            nrm(1,j)=cn(1,j)+cp(1,j);
                            cn(1,j)=cn(1,j)/nrm(1,j);
                            cp(1,j)=cp(1,j)/nrm(1,j);
                            if params.balanceclasses==1
                                mdl{j}=w_train(XK,Y,W(:,j),params.C,cp(1,j),cn(1,j));
%                                 train(W(:,j),Y,svmX,['-s 5 -c ' num2str(params.C) ' -q -w-1 ' num2str(cn(1,j)) ' -w1 ' num2str(cp(1,j)) ' ' svmparams]);
                            else
                                mdl{j}=w_train(XK,Y,W(:,j),params.C,1,1);
%                                 train(W(:,j),Y,svmX,['-s 5 -c ' num2str(params.C) ' -q ' svmparams]);
                            end
                        end
                        
                    else
                        IDXfinal=IDX;
                    end
                elseif params.fixedclustering==1
                    IDXfinal=params.fixedclusteringIDX(Y==1,1);
                    W=zeros(size(Y,1),params.k);
                    W(sub2ind(size(W),find(Y==1),IDXfinal))=1;
                    W(Y==-1,:)=1/params.k;
                    cn=zeros(1,params.k);cp=zeros(1,params.k);nrm=zeros(1,params.k);
                    for j=1:params.k
                        cn(1,j)=1./mean(W(Y==-1,j),1);
                        cp(1,j)=1./mean(W(Y==1,j),1);
                        nrm(1,j)=cn(1,j)+cp(1,j);
                        cn(1,j)=cn(1,j)/nrm(1,j);
                        cp(1,j)=cp(1,j)/nrm(1,j);
                        if params.balanceclasses==1
                            mdl{j}=w_train(XK,Y,W(:,j),params.C,cp(1,j),cn(1,j));
%                             train(W(:,j),Y,svmX,['-s 5 -c ' num2str(params.C) ' -q -w-1 ' num2str(cn(1,j)) ' -w1 ' num2str(cp(1,j)) ' ' svmparams]);
                        else
                            mdl{j}=w_train(XK,Y,W(:,j),params.C,1,1);
%                             train(W(:,j),Y,svmX,['-s 5 -c ' num2str(params.C) ' -q ' svmparams]);
                        end
                    end
                    
                end
                model.mdl=mdl;
                model.S=W(Y==1,:);
                model.W=W;
                model.Yhat=Y;
                model.Yhat(Y==1)=IDXfinal;
                model.cn=cn;
                model.cp=cp;
        end
    
end

model.params=params;
end

function IDXfinal=consensus_clustering(IDX,k)
%Function performs consensus clustering on a co-occurence matrix
[n,~]=size(IDX);
cooc=zeros(n);
for i=1:n-1
    for j=i+1:n
        cooc(i,j)=sum(IDX(i,:)==IDX(j,:));
    end
    %cooc(i,i)=sum(IDX(i,:)==IDX(i,:))/2;
end
cooc=cooc+cooc';
L=diag(sum(cooc,2))-cooc;

Ln=eye(n)-diag(sum(cooc,2).^(-1/2))*cooc*diag(sum(cooc,2).^(-1/2));
Ln(isnan(Ln))=0;
[V,~]=eig(Ln);
try
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
catch
    disp('Complex Eigenvectors Found...Using Non-Normalized Laplacian');
    [V,~]=eig(L);
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
end

end

function [S,Yhat]=hydra_init_v2(XK,Y,k,params)
%Function performs initialization for supervised clustering
nker=@(K)(K./sqrt(diag(K)*diag(K)'));
init_type=params.init_type;
if params.regression==0
    if params.kernel==0
        X=XK;
        if init_type==0; %% Random hyperplanes
            idxp=find(Y==1);
            idxn=find(Y==-1);
            prob=zeros(size(X(Y==1,:),1),k);
            for j=1:k
                ip=randi(length(idxp));
                in=randi(length(idxn));
                w0=(X(idxp(ip),:)-X(idxn(in),:));
                w0=w0/norm(w0);
                prob(:,j)=bsxfun(@times,X(Y==1,:),1./norms(X(Y==1,:),2,2))*w0';
            end
            l=min(prob-1,0);
            d=prob-1;
            S=LP1(l,d);
        elseif init_type==1; %% Random assignment
            S=drchrnd(ones(1,k),size(X(Y==1,:),1));
        elseif init_type==2; %% K-means
            IDX=kmeans(X(Y==1,:),k,'replicates',20);
            S=zeros(size(X(Y==1,:),1),k);
            S(sub2ind(size(S),(1:size(S,1))',IDX))=1;
        elseif init_type==3; %% DPP random hyperplanes
            idxp=find(Y==1);
            idxn=find(Y==-1);
            num=size(X,1);
            W=zeros(num,size(X,2));
            for j=1:num
                ip=randi(length(idxp));
                in=randi(length(idxn));
                W(j,:)=(X(idxp(ip),:)-X(idxn(in),:));
            end
            KW=W*W';
            KW=KW./sqrt(diag(KW)*diag(KW)');
            Widx = sample_dpp(decompose_kernel(KW),k);
            prob=zeros(size(X(Y==1,:),1),k);
            for j=1:k
                prob(:,j)=bsxfun(@times,X(Y==1,:),1./norms(X(Y==1,:),2,2))*(W(Widx(j),:))';
            end
            l=min(prob-1,0);
            d=prob-1;
            S=LP1(l,d);
        end
        Yhat=-ones(size(Y));
        [~,Yhat(Y==1)]=max(S,[],2);
    elseif params.kernel==1
        K=XK;
        if init_type==0
            Kn=nker(K);
            idxp=find(Y==1);
            idxn=find(Y==-1);
            prob=zeros(size(K(Y==1,:),1),k);
            for j=1:k
                ip=randi(length(idxp));
                in=randi(length(idxn));
                prob(:,j)=Kn(:,idxp(ip))-Kn(:,idxn(in));
            end
            l=min(prob-1,0);
            d=prob-1;
            S=LP1(l,d);
        elseif init_type==1
            S=drchrnd(ones(1,k),size(K(Y==1,:),1));
        elseif init_type==2
            IDX=knkmeans(K(Y==1,Y==1),k,20);
            S=zeros(size(K(Y==1,:),1),k);
            S(sub2ind(size(S),(1:size(S,1))',IDX))=1;
        elseif init_type==3;
            Kn=nker(K);
            idxp=find(Y==1);
            idxn=find(Y==-1);
            num=size(K,1);
            KW=zeros(num,num);
            KWidxp=zeros(num,1);
            KWidxn=zeros(num,1);
            for i=1:num
                KWidxp(i,1)=randi(length(idxp));
                KWidxn(i,1)=randi(length(idxn));
            end
            for i=1:num
                for j=i:num
                    KW(i,j)=K(idxp(KWidxp(i,1)),idxp(KWidxp(j,1)))+K(idxn(KWidxn(i,1)),idxn(KWidxn(j,1)))-K(idxp(KWidxp(i,1)),idxn(KWidxn(j,1)))-K(idxn(KWidxn(i,1)),idxp(KWidxp(j,1)));
                    KW(j,i)=KW(i,j);
                end
            end
            KW=KW./sqrt(diag(KW)*diag(KW)');
            Widx = sample_dpp(decompose_kernel(KW),k);
            prob=zeros(size(K(Y==1,:),1),k);
            for j=1:k
                prob(:,j)=Kn(Y==1,idxp(KWidxp(Widx(j))))-Kn(Y==1,idxn(KWidxn(Widx(j))));
            end
            l=min(prob-1,0);
            d=prob-1;
            S=LP1(l,d);
        end
        Yhat=-ones(size(Y));
        [~,Yhat(Y==1)]=max(S,[],2);
    end
end
end

function s=LP1(l,d)
% Proportional assignment based on margin
invL=1./l;
idx=find(invL==Inf);
invL(idx)=d(idx);
for i=1:size(invL,1)
    pos=find(invL(i,:)>0); %#ok<*EFIND>
    neg=find(invL(i,:)<0);
    if ~isempty(pos)
        invL(i,neg)=0; %#ok<*FNDSB>
    else
        invL(i,:)=invL(i,:)/min(invL(i,:),[],2);
        invL(i,invL(i,:)<1)=0;
    end
end
s=bsxfun(@times,invL,1./sum(invL,2));
end

function epsilon=svr_parameter_selection(XK,Y,params)
%Function selects epsilon for svr
sigma=noise_estimator(XK,Y,params);
epsilon=3*sigma*sqrt(log(size(XK,1))/size(XK,1));
end

function sigma=noise_estimator(XK,Y,params)

if params.kernel==1
    Ypred=loo_kernel_knn(XK,Y,5);
elseif params.kernel==0
    K=XK*XK';
    Ypred=loo_kernel_knn(K,Y,5);
end

sigma=sqrt((5/4)*(1/size(XK,1))*sum((Y-Ypred).^2));
end

function Ypred=loo_kernel_knn(K,Y,k)
[n,~]=size(K);
D=kernel2dist(K);
Ypred=zeros(n,1);
for i=1:n
    Yi=Y((1:n)~=i);
    [~,idx]=sort(D(i,(1:n)~=i),2,'ascend');
    Ypred(i,1)=mean(Yi(idx(1:k)));
end
end

function D=kernel2dist(K)
D=zeros(size(K));
for i=1:size(K,1)-1
    for j=i+1:size(K,1)
        D(i,j)=K(i,i)+K(j,j)-2*K(i,j);
    end
end
D=D+D';
end

function Y = sample_dpp(L,k)
% sample a set Y from a dpp.  L is a decomposed kernel, and k is (optionally)
% the size of the set to return.

if ~exist('k','var')
    % choose eigenvectors randomly
    D = L.D ./ (1+L.D);
    v = find(rand(length(D),1) <= D);
else
    % k-DPP
    v = sample_k(L.D,k);
end
k = length(v);
V = L.V(:,v);

% iterate
Y = zeros(k,1);
for i = k:-1:1
    
    % compute probabilities for each item
    P = sum(V.^2,2);
    P = P / sum(P);
    
    % choose a new item to include
    Y(i) = find(rand <= cumsum(P),1);
    
    % choose a vector to eliminate
    j = find(V(Y(i),:),1);
    Vj = V(:,j);
    V = V(:,[1:j-1 j+1:end]);
    
    % update V
    V = V - bsxfun(@times,Vj,V(Y(i),:)/Vj(Y(i)));
    
    % orthogonalize
    for a = 1:i-1
        for b = 1:a-1
            V(:,a) = V(:,a) - V(:,a)'*V(:,b)*V(:,b);
        end
        V(:,a) = V(:,a) / norm(V(:,a));
    end
    
end

Y = sort(Y);
end

function L = decompose_kernel(M)
L.M = M;
[V,D] = eig(M);
L.V = real(V);
L.D = real(diag(D));
end

function S = sample_k(lambda,k)
% pick k lambdas according to p(S) \propto prod(lambda \in S)

% compute elementary symmetric polynomials
E = elem_sympoly(lambda,k);

% iterate
i = length(lambda);
remaining = k;
S = zeros(k,1);
while remaining > 0
    
    % compute marginal of i given that we choose remaining values from 1:i
    if i == remaining
        marg = 1;
    else
        marg = lambda(i) * E(remaining,i) / E(remaining+1,i+1);
    end
    
    % sample marginal
    if rand < marg
        S(remaining) = i;
        remaining = remaining - 1;
    end
    i = i-1;
end
end

function E = elem_sympoly(lambda,k)
% given a vector of lambdas and a maximum size k, determine the value of
% the elementary symmetric polynomials:
%   E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)

N = length(lambda);
E = zeros(k+1,N+1);
E(1,:) = 1;
for l = (1:k)+1
    for n = (1:N)+1
        E(l,n) = E(l,n-1) + lambda(n-1)*E(l-1,n-1);
    end
end
end

function [label, energy,LABEL,ENERGY] = knkmeans(K,init,replicates)
% Perform kernel k-means clustering.
%   K: kernel matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k)
% Reference: [1] Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Michael Chen (sth4nth@gmail.com).
if nargin<3
    replicates=20;
end
LABEL=zeros(size(K,1),replicates);
ENERGY=zeros(1,replicates);
for TT=1:replicates
    n = size(K,1);
    if length(init) == 1
        label = ceil(init*rand(1,n));
    elseif size(init,1) == 1 && size(init,2) == n
        label = init;
    else
        error('ERROR: init is not valid.');
    end
    last = 0;
    while any(label ~= last)
        [u,~,label] = unique(label,'legacy');   % remove empty clusters
        k = length(u);
        E = sparse(label,1:n,1,k,n,n);
        E = bsxfun(@times,E,1./sum(E,2));
        T = E*K;
        Z = repmat(diag(T*E'),1,n)-2*T;
        last = label;
        [val, label] = min(Z,[],1);
    end
    [~,~,label] = unique(label,'legacy');   % remove empty clusters
    LABEL(:,TT)=label';
    ENERGY(:,TT) = sum(val)+trace(K);
end
[energy,IDX]=min(ENERGY,[],2);
label=LABEL(:,IDX);
end

function r = drchrnd(a,n)
% take a sample from a dirichlet distribution
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end

function o = norms( x, p, dim )
%Function computes vector norms
switch p,
    case 1,
        o = sum( abs( x ), dim );
    case 2,
        o = sqrt( sum( x .* conj( x ), dim ) );
    case Inf,
        o = max( abs( x ), [], dim );
    otherwise,
        o = sum( abs( x ) .^ p, dim ) .^ ( 1 / p );
end
end

function mdl=w_svmtrain(X,Y,W,C,Cp,Cn,dual)
%Function solves weighted l2-svm, requires matlab optimization toolbox version 2014+
if any(isnan([Cp Cn]))
    mdl.w=zeros(size(X,2),1);
    mdl.b=0;
    warning('Cluster dropped');
    return
end
if dual==0
    X=X;
elseif dual==1
    [U,S,~]=svd(X);
    X=U*sqrt(S);
    
end
idxp=find(Y==1);
idxn=find(Y==-1);
Cw=zeros(size(Y));
Cw(idxp)=Cp;
Cw(idxn)=Cn;
[n,d] = size(X);
H = diag([ones(1, d), zeros(1, n + 1)]);
f = [zeros(1,d+1) C*(ones(1,n).*W'.*Cw')]';
p = diag(Y) * X;
A = -[p Y eye(n)];
B = -ones(n,1);
lb = [-inf * ones(d+1,1) ;zeros(n,1)];
options=optimoptions('quadprog','Display','off','OptimalityTolerance',1e-8);
z = quadprog(H,f,A,B,[],[],lb,[],[],options);

mdl.w = z(1:d,:);
mdl.b = z(d+1:d+1,:);
mdl.eps = z(d+2:d+n+1,:);
end

function mdl=w_train(X,Y,W,C,Cp,Cn)
%Function solves weighted l1-svm, requires matlab optimization toolbox version 2014+
if any(isnan([Cp Cn]))
    mdl.w=zeros(size(X,2),1);
    mdl.b=0;
    %warning('Cluster dropped');
    return
end
idxp=find(Y==1);
idxn=find(Y==-1);
Cw=zeros(size(Y,1),1);
Cw(idxp)=Cp;
Cw(idxn)=Cn;
[n,d]=size(X);
H=blkdiag(zeros(d),zeros(d),diag(C*W.*Cw));
f=[ones(d,1);ones(d,1);zeros(n,1)];
A=-[diag(Y)*X -diag(Y)*X eye(n)];
b=-ones(n,1);
lb=[zeros(d,1);zeros(d,1);zeros(n,1)];
ub=[inf(d,1);inf(d,1);inf(n,1)];
options=optimoptions('quadprog','Display','off','OptimalityTolerance',1e-8);
v = quadprog(H,f,A,b,[],[],lb,ub,[],options);

mdl.w=v(1:d)-v(d+1:2*d);
mdl.b=0;
end

function S=w_svmpredict(X,mdl,dual)
%Function makes svm prediction using model
if dual==0
    X=X;
elseif dual==1
    [U,S,~]=svd(X);
    X=U*sqrt(S);
    
end

S=X*mdl.w+mdl.b;

end

