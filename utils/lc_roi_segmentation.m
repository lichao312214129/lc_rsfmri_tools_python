function lc_roi_segmentation(varargin)
% LC_ROI_SEGMENTATION
% Usage: see EXAMPLE below.
% GOAL: This function is used to segment a ROI into K sub-regions
% according to its function connectivity with other voxel by using kmeans clustering.
% INPUT:
%       REQUIRED:
%           [--data_dir, -dd]: directory of all 4D files, preprocessed using software like DPABI (e.g., FunImgARWSFC).
%           [--roi2segment,-rs]: image file of roi need to segment, .nii or .img
%           [--num_of_subregion, -ns]: number of sub-regions you want to segment the roi2segment, default is 3;
%           
%       OPTIONAL
%           [--mask_file, -mf]:  mask file for filtering data, .nii or .img
%           [--out_dir, -od]: output directory
%           [--n_workers, -nw]: How many threads(CPU) to use.
%
% OUTPUT:
%       All subject level segmentation and one group level segmentation.
% EXAMPLE 2 :
% lc_roi_segmentation('-dd', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\signals',...
%                           '-rs', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\Amygdala_3_3_3.nii',...
%                           '-ns', 3,... 
%                           '-mf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\sorted_brainnetome_atalas_3mm.nii',...
%                           '-od', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan');
% 
% REFERENCE:
%   <Individual-specific functional connectivity of the roi: A substrate for precision psychiatry>
% @author: Li Chao
% Email: lichao19870617@gmail.com

% EXAMPLE 1:
% lc_roi_segmentation('-dd', 'D:\FunImgARWSFC',...
%                     '-rs', 'D:\roi_3_3_3.nii',...
%                     '-ns', 3,... 
%                     '-mf', 'D:\sorted_brainnetome_atalas_3mm.nii',...
%                     '-od', 'D:\Results');

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
fc_all = zeros(size(roi_signal, 2), size(non_roi_signal, 2));

try
    parpool(n_workers)
catch
    fprintf('Parpool already running!\n')
end

parfor i = 1:n_sub
    fprintf('Running %d/%d\n', i, n_sub);
    disp('----------------------------------');
    % Extract roi
    data_target = y_Read(datafile_path{i});
    data_target = reshape(data_target, dim1*dim2*dim3, [])';
    roi_signal = data_target(:, roi_mask);
    
    % Extract non-roi signals
    non_roi_signal = data_target(:, (~roi_mask) & mask);
    
    % Step 3 is to calculate the partial correlations
    fc = corr(roi_signal, non_roi_signal, 'Type','Pearson', 'rows', 'pairwise');
    % 
    nannum = isnan(fc);
    infnum = isinf(fc);
    if (sum(nannum(:)) == 0) && (sum(infnum(:)) == 0)
        fc_all = fc_all + fc;
    end
    
    % Kmeans
    disp('Kmeans...');
    stream = RandStream('mlfg6331_64');
    options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
    fc(isnan(fc)) = 0;
    fc(isinf(fc)) = 1;
    [idx, C, sumd, D] = kmeans(fc, num_of_subregion, 'Distance', 'cityblock', 'Options', options, 'replicate',10, 'Display','iter', 'emptyaction', 'singleton');
    
    % Segment the target region into several sub-regions.
    segmentation = zeros(size(roi_mask));
    segmentation(roi_mask) = idx;
    segmentation_3d = reshape(segmentation, size(roi));
    [~, sub_name] = fileparts(fileparts(datafile_path{i}));
    header = header_roi;
    header.descrip = 'segmentated using kmeans clustering to function connectivity';
    y_Write(segmentation_3d, header, fullfile(out_dir, [sub_name, '.nii']));
end

% Group segmentation
fc_all_mean = fc_all./n_sub;
stream = RandStream('mlfg6331_64');
options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
[idx_group, C, sumd, D] = kmeans(fc_all_mean, num_of_subregion, 'Distance', 'cityblock', 'Options', options, 'replicate',10, 'Display','iter', 'emptyaction', 'singleton');
segmentation_group = zeros(size(roi_mask));
segmentation_group(roi_mask) = idx_group;
segmentation_group_3d = reshape(segmentation_group, size(roi));
header = header_roi;
header.descrip = 'Group segmentated using kmeans clustering to function connectivity';
y_Write(segmentation_group_3d, header, fullfile(out_dir, 'group_segmentation.nii'));
    
disp('Done!');
end
