function lc_roi_segmentation_special_clustering(varargin)
% LC_ROI_SEGMENTATION
% Usage: see EXAMPLE below.
% GOAL: This function is used to segment a ROI into K sub-regions
% according to its function connectivity with other voxel by using kmeans clustering.
% INPUT:
%       REQUIRED:
%           [--data_dir, -dd]: directory of all 4D files, preprocessed using software like DPABI (e.g., FunImgARWSFC).
%           [--roi2segment,-rs]: image file of roi need to segment, .nii or .img
%           
%       OPTIONAL
%           [--num_of_subregion, -ns]: number of sub-regions you want to segment the roi2segment, default is 3;
%           [--mask_file, -mf]:  mask file for filtering data, .nii or .img
%           [--out_dir, -od]: output directory
%           [--n_workers, -nw]: How many threads(CPU) to use.
%           [--n_replicate, -nr]  Number of times to repeat clustering using new initial cluster centroid positions, default is 100.
%           [--is_pca, -ip] Whether perform PCA to reduce dimension,
%           default is 1, 0 OR 1
%           [--explained_cov, -ec]: how many explained variance to retain, default is 0.90, range = (0, 1]
%
% OUTPUT:
%       All subject level segmentation and one group level segmentation.
% EXAMPLE 1 :
% lc_roi_segmentation_special_clustering('-dd', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\data',...
%                           '-rs', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\Amygdala_3_3_3.nii',...
%                           '-ns', 3,... 
%                            '-nr', 500,...
%                            '-ip',0,...
%                            '-ec', 0.80,... 
%                           '-mf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\sorted_brainnetome_atalas_3mm.nii',...
%                           '-od', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan');
% EXAMPLE 2(without pca) :
% lc_roi_segmentation_special_clustering('-dd', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\signals',...
%                           '-rs', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\Amygdala_3_3_3.nii',...
%                           '-ns', 3,... 
%                            '-nr', 500,...
%                            '-ip',0,...
%                           '-mf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\sorted_brainnetome_atalas_3mm.nii',...
%                           '-od', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan');
% REFERENCE:
%   <Individual-specific functional connectivity of the roi: A substrate for precision psychiatry>
% @author: Li Chao
% Email: lichao19870617@gmail.com

help lc_roi_segmentation
fprintf('-----------------------------------------------------------\n');

if nargin == 0
    help lc_roi_segmentation
    return
end

% Varargin parser
num_of_subregion = 3;
mask_file = '';
out_dir = pwd;
n_workers = 4;
n_replicate = 100;
is_pca = 1;
explained_cov = 0.90;

if( sum(or(strcmpi(varargin,'--data_dir'),strcmpi(varargin,'-dd')))==1)
    data_dir = varargin{find(or(strcmpi(varargin,'--data_dir'),strcmp(varargin,'-dd')))+1};
else
    help lc_roi_segmentation
    error('Please specify data_dir!');
end

if( sum(or(strcmpi(varargin,'--roi2segment'),strcmpi(varargin,'-rs')))==1)
    roi2segment = varargin{find(or(strcmpi(varargin,'--roi2segment'),strcmp(varargin,'-rs')))+1};
else
    help lc_roi_segmentation
    error('Please specify roi2segment!');
end

if( sum(or(strcmpi(varargin,'--num_of_subregions'),strcmpi(varargin,'-ns')))==1)
    num_of_subregion = varargin{find(or(strcmpi(varargin,'--num_of_subregions'),strcmp(varargin,'-ns')))+1};
end

if( sum(or(strcmpi(varargin,'--mask_file'),strcmpi(varargin,'-mf')))==1)
    mask_file = varargin{find(or(strcmpi(varargin,'--mask_file'),strcmp(varargin,'-mf')))+1};
end

if( sum(or(strcmpi(varargin,'--out_dir'),strcmpi(varargin,'-od')))==1)
    out_dir = varargin{find(or(strcmpi(varargin,'--out_dir'),strcmp(varargin,'-od')))+1};
end

if( sum(or(strcmpi(varargin,'--n_workers'),strcmpi(varargin,'-nw')))==1)
    n_workers = varargin{find(or(strcmpi(varargin,'--n_workers'),strcmp(varargin,'-nw')))+1};
end

if( sum(or(strcmpi(varargin,'--n_replicate'),strcmpi(varargin,'-nr')))==1)
    n_replicate = varargin{find(or(strcmpi(varargin,'--n_replicate'),strcmp(varargin,'-nr')))+1};
end

if( sum(or(strcmpi(varargin,'--is_pca'),strcmpi(varargin,'-ip')))==1)
    is_pca = varargin{find(or(strcmpi(varargin,'--is_pca'),strcmp(varargin,'-ip')))+1};
end

if( sum(or(strcmpi(varargin,'--explained_cov'),strcmpi(varargin,'-ec')))==1)
    explained_cov = varargin{find(or(strcmpi(varargin,'--explained_cov'),strcmp(varargin,'-ec')))+1};
end

% Make output directory
if ~exist(out_dir)
    mkdir(out_dir)
end

% Load roi roi
[roi, header_roi] = y_Read(roi2segment);
[dim1, dim2, dim3] = size(roi);
roi_mask = reshape(roi, dim1*dim2*dim3, [])' ~= 0;

% Load source data path
data_strut = dir(data_dir);
data_path = fullfile(data_dir, {data_strut.name});
data_path = data_path(3:end)';

n_sub = length(data_path);
datafile_path = cell(n_sub,1);
for i = 1: n_sub
    one_data_strut = dir(data_path{i});
    one_data_path = fullfile(data_path{i}, {one_data_strut.name});
    one_data_path = one_data_path(3:end);
    datafile_path(i) = one_data_path(1);
end

% Load mask
if ~strcmp(mask_file, '')
    mask = y_Read(mask_file) ~= 0;
    if (~all(size(mask) == size(roi)))
        disp('Dimension of the mask did not match roi!');
        return;
    end
else
    mask = ones(dim1, dim2, dim3) == 1;
end
mask = reshape(mask, dim1*dim2*dim3, [])' ~= 0;

% iteration
data_target = y_Read(datafile_path{1});
data_target = reshape(data_target, dim1*dim2*dim3, [])';
roi_signal = data_target(:, roi_mask);
non_roi_signal = data_target(:, (~roi_mask) & mask);
fc_all = zeros(size(roi_signal, 2), size(roi_signal, 2));

try
    parpool(n_workers)
catch
    fprintf('Parpool already running!\n')
end

% Kmeans
disp('Kmeans...');
stream = RandStream('mlfg6331_64');
options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
idx_all = zeros(size(roi_signal, 2),n_sub);
for i = 1:n_sub
    fprintf('Running %d/%d\n', i, n_sub);
    disp('----------------------------------');
    % Extract roi
    data_target = y_Read(datafile_path{i});
    data_target = reshape(data_target, dim1*dim2*dim3, [])';
    roi_signal = data_target(:, roi_mask);
 
    % Extract non-roi signals
    non_roi_signal = data_target(:, (~roi_mask) & mask);
    
    % Fillmissing of roi signals
    roi_signal(isinf(roi_signal)) = NaN;
    non_roi_signal(isinf(non_roi_signal)) = NaN;
    x = 1:size(roi_signal);
    [roi_signal_intered,~] = fillmissing(roi_signal,'linear','SamplePoints',x);
    [non_roi_signal_intered, ~] =  fillmissing(non_roi_signal,'linear','SamplePoints',x);

    % Step 3 is to calculate the partial correlations
    fc = corr(roi_signal_intered, non_roi_signal_intered, 'Type','Pearson', 'rows', 'pairwise');

    % Fillmissing of roi signals
    fc(isinf(fc)) = NaN;
    x = 1:size(fc, 2);
    [fc,~] = fillmissing(fc','linear','SamplePoints',x);
    fc = fc';
    
    % PCA
    if is_pca
        [~, fc_reduced,~,~,explained] = pca(fc);
        n_comp = numel(explained);
        cum_ex_list = zeros(n_comp, 1);
        cum_ex = 0;
        for j = 1:n_comp
            cum_ex = cum_ex + explained(j);
            cum_ex_list(j) = cum_ex;
        end
        loc_cutoff_cum_ex = find(cum_ex_list >= explained_cov*100);
        if numel(loc_cutoff_cum_ex) >= 1
            loc_cutoff_cum_ex = loc_cutoff_cum_ex(1);
            fc_reduced = fc_reduced(:,1:loc_cutoff_cum_ex);
        else
            fc_reduced = fc;
        end
    else
        fc_reduced = fc;
    end
    
    fc_reduced(isnan(fc_reduced)) = 0;
    fc_reduced(isinf(fc_reduced)) = 1;
    W = corr(fc_reduced');
    
    if i == 1
        fc_all = fc;
    else
        fc_all = cat(1, fc_all, fc);
    end

    idx = special_clustering(W, num_of_subregion, n_replicate);
    idx_all(:,i) = idx;
    % Segment the target region into several sub-regions.
    segmentation = zeros(size(roi_mask));
    segmentation(roi_mask) = idx;
    segmentation_3d = reshape(segmentation, size(roi));
    [~, sub_name] = fileparts(fileparts(datafile_path{i}));
    header = header_roi;
    header.descrip = 'segmentated using kmeans clustering to function connectivity';
    y_Write(segmentation_3d, header, fullfile(out_dir, [sub_name, '.nii']));
end

clear fc_reduced idx roi_signal_intered data_target

% Group segmentation
% PCA
if is_pca
    [~, fc_all_reduced,~,~,explained] = pca(fc_all);
    n_comp = numel(explained);
    cum_ex_list = zeros(n_comp, 1);
    cum_ex = 0;
    for j = 1:n_comp
        cum_ex = cum_ex + explained(j);
        cum_ex_list(j) = cum_ex;
    end
    loc_cutoff_cum_ex = find(cum_ex_list >= explained_cov*100);
    loc_cutoff_cum_ex = loc_cutoff_cum_ex(1);
    fc_all_reduced = fc_all_reduced(:,1:loc_cutoff_cum_ex);
else
    fc_all_reduced = fc_all;
end

W_all = corr(fc_all_reduced')';
idx_all = special_clustering(W_all, num_of_subregion, n_replicate);
idx_all = reshape(idx_all, [],n_sub);
save(fullfile(out_dir,'idx_group.mat'), 'idx_all');
n_voxel_in_roi = size(roi_signal, 2);
idx_subgoup = 1:num_of_subregion;
idx_group = zeros(n_voxel_in_roi,1);
freq_all = cell(n_voxel_in_roi,1);
for i = 1:n_voxel_in_roi
    comp = arrayfun(@(x) idx_all(i,:)-x, idx_subgoup, 'UniformOutput', false);
    freq = cell2mat(cellfun(@(x) sum(x==0)/n_sub, comp, 'UniformOutput', false));
    freq_all{i} = freq;
    ig = find(freq==max(freq));
    randig = randperm(numel(ig));
    randig = randig(1);
    idx_group(i) = ig(randig);
end

segmentation_group = zeros(size(roi_mask));
segmentation_group(roi_mask) = idx_group;
segmentation_group_3d = reshape(segmentation_group, size(roi));
header = header_roi;
header.descrip = 'Group segmentated using kmeans clustering to function connectivity';
y_Write(segmentation_group_3d, header, fullfile(out_dir, 'group_segmentation.nii'));
    
disp('Done!');
end

function [ idx, L, D, Q, V ] = special_clustering(W, k, n_replicate)
% spectral clustering algorithm
% Input: 
% -----
%   W: adjacency matrix with N * N dimension, N is the number of samples; 
%   k: number of cluster k 

% return: 
% ------
%   idx: cluster indicator vectors as columns in 
%   L: unnormalized Laplacian
%   D: degree matrix
%   Q: eigenvectors matrix
%   V: eigenvalues matrix
% NOTE. This function is revised from https://www.cnblogs.com/FengYan/archive/2012/06/21/2553999.html

% Calculate degree matrix
degs = sum(W, 2);
D = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = D - W;

order_occurence = 1:size(L,1);
[L,~] = fillmissing(L,'linear','SamplePoints',order_occurence);

% compute the eigenvectors corresponding to the k smallest eigenvalues
% diagonal matrix V is NcutL's k smallest magnitude eigenvalues 
% matrix Q whose columns are the corresponding eigenvectors.
[Q, V] = eigs(L, k, 'SA');

% use the k-means algorithm to cluster V row-wise
% C will be a n-by-1 matrix containing the cluster number for each data point
stream = RandStream('mlfg6331_64');
options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
[idx, C,sumd, D] = kmeans(Q, k, 'Distance', 'cityblock', 'Options', options, 'replicate',n_replicate, 'Display','iter', 'emptyaction', 'singleton');

% Sort idx so each subject has the same order
order_occurence = idx(1);
count =1;
for i = 1:numel(idx)
    if ~(ismember( idx(i),order_occurence))
        order_occurence(count+1) = idx(i);
        count = count +1;
    end
end   
niter = numel(order_occurence);
maxnum = max(order_occurence);
for i = 1: niter
    idx(idx==order_occurence(i)) = maxnum + i;
end
idx = idx - maxnum;
end