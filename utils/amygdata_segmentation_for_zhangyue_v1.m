function amygdata_segmentation_for_zhangyue_v1(varargin)
% AMGYDALA_SEGMENTATION_FOR_ZHANGYUE
% Usage: see EXAMPLE below.
% GOAL: This function is used to segment amgydala into three sub-regions
% according to its function connectivity with other voxel by using kmeans clustering.
% INPUT:
%       REQUIRED:
%           [--data_dir, -dd]: directory of all 4D files
%           [--amygdala_roi_file,-arf]: file of amygdala roi
%           
%       OPTIONAL
%           [--mask_file, -mf]:  mask file for filtering data
%           [--out_dir, -od]: output directory
%
% OUTPUT:
%       All subject level segmentation and one group level segmentation.
% EXAMPLE:
% amygdata_segmentation_for_zhangyue_v1('-dd', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\signals',...
%                           '-arf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\Amygdala_3_3_3.nii',...
%                           '-mf', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan\sorted_brainnetome_atalas_3mm.nii',...
%                           '-od', 'D:\workstation_b\ZhangYue_Guangdongshengzhongyiyuan');
% REFERENCE:
%   <Individual-specific functional connectivity of the amygdala: A substrate for precision psychiatry>
% @author: Li Chao
% Email: lichao19870617@gmail.com

if nargin == 0
    help amgydala_segmentation_for_zhangyue_v1
    return
end

% Varargin parser
mask_file = '';
out_dir = pwd;

if( sum(or(strcmpi(varargin,'--data_dir'),strcmpi(varargin,'-dd')))==1)
    data_dir = varargin{find(or(strcmpi(varargin,'--data_dir'),strcmp(varargin,'-dd')))+1};
else
    error('Please specify data_dir!');
end

if( sum(or(strcmpi(varargin,'--amygdala_roi_file'),strcmpi(varargin,'-arf')))==1)
    amygdala_roi_file = varargin{find(or(strcmpi(varargin,'--amygdala_roi_file'),strcmp(varargin,'-arf')))+1};
else
    error('Please specify amygdala_roi_file!');
end

if( sum(or(strcmpi(varargin,'--mask_file'),strcmpi(varargin,'-mf')))==1)
    mask_file = varargin{find(or(strcmpi(varargin,'--mask_file'),strcmp(varargin,'-mf')))+1};
end

if( sum(or(strcmpi(varargin,'--out_dir'),strcmpi(varargin,'-od')))==1)
    out_dir = varargin{find(or(strcmpi(varargin,'--out_dir'),strcmp(varargin,'-od')))+1};
end


% Load amygdala roi
[amygdala, header_amygdala] = y_Read(amygdala_roi_file);
[dim1, dim2, dim3] = size(amygdala);
amygdala_mask = reshape(amygdala, dim1*dim2*dim3, [])' ~= 0;

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
    if (~all(size(mask) == size(amygdala)))
        disp('Dimension of the mask did not match amygdala!');
        return;
    end
else
    mask = ones(dim1, dim2, dim3) == 1;
end
mask = reshape(mask, dim1*dim2*dim3, [])' ~= 0;

% iteration
for i = 1:n_sub
    fprintf('Running %d/%d\n', i, n_sub);
    disp('----------------------------------');
    % Extract amygdala
    data_target = y_Read(datafile_path{i});
    data_target = reshape(data_target, dim1*dim2*dim3, [])';
    amygdala_signal = data_target(:, amygdala_mask);
    
    % Extract non-amygdala signals
    non_amygdala_signal = data_target(:, (~amygdala_mask) & mask);
    
    % Step 3 is to calculate the partial correlations
    fc = corr(amygdala_signal, non_amygdala_signal, 'Type','Pearson');
    if i == 1
        fc_all = zeros(size(amygdala_signal, 2), size(non_amygdala_signal, 2));
    end
    fc_all = fc_all + fc;
    
    % Kmeans
    disp('Kmeans...');
    stream = RandStream('mlfg6331_64');
    options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
    fc(isnan(fc)) = 0;
    [idx, C, sumd, D] = kmeans(fc, 3, 'Distance', 'cityblock', 'Options', options, 'replicate',10, 'Display','iter');
    
    % Segment the target region into several sub-regions.
    segmentation = zeros(size(amygdala_mask));
    segmentation(amygdala_mask) = idx;
    segmentation_3d = reshape(segmentation, size(amygdala));
    [~, sub_name] = fileparts(fileparts(datafile_path{i}));
    header = header_amygdala;
    header.descrip = 'segmentated using kmeans clustering to function connectivity';
    y_Write(segmentation_3d, header, fullfile(out_dir, [sub_name, '.nii']));
end

% Group segmentation
fc_all_mean = fc_all./n_sub;
[idx_group, C, sumd, D] = kmeans(fc_all_mean, 3, 'Distance', 'cityblock', 'Options', options, 'replicate',10, 'Display','iter');
segmentation_group = zeros(size(amygdala_mask));
segmentation_group(amygdala_mask) = idx_group;
segmentation_group_3d = reshape(segmentation_group, size(amygdala));
header = header_amygdala;
header.descrip = 'Group segmentated using kmeans clustering to function connectivity';
y_Write(segmentation_group_3d, header, fullfile(out_dir, 'group_segmentation.nii'));
    
disp('Done!');
end
