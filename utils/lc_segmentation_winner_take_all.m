function lc_segmentation_winner_take_all(varargin)
% LC_ROI_SEGMENTATION
% Usage: see EXAMPLE below.
% GOAL: This function is used to segment a ROI into K sub-regions
% according to its function connectivity with other voxel by using kmeans clustering.
% INPUT:
%       REQUIRED:
%           [--data_dir, -dd]: directory of all 4D files, preprocessed using software like DPABI (e.g., FunImgARWSFC).
%           [--atalas, -as]: atalas of network (cortex)
%           [--network_index, -ni]: network index of atalas
%           [--roi2segment,-rs]: image file of roi need to segment, .nii or .img
%           
%       OPTIONAL
%           [--mask_file, -mf]:  mask file for filtering data, .nii or .img
%           [--out_dir, -od]: output directory
%           [--n_workers, -nw]: How many threads(CPU) to use.
%
% OUTPUT:
%       All subject level segmentation and one group level segmentation.
% EXAMPLE:
% lc_segmentation_winner_take_all('-dd', 'D:\yueyingkeji\tange\data',...
%                           '-as', 'G:\BranAtalas\Template_Yeo2011\Yeo2011_7Networks_N1000.split_components.FSL_MNI152_3mm.nii',...
%                           '-ni', 'D:\workstation_b\tange\netIndex.mat',...
%                           '-rs', 'D:\workstation_b\tange\Amygdala_3_3_3.nii',...
%                           '-mf', 'D:\WorkStation_2018\SZ_classification\Data\Atalas\sorted_brainnetome_atalas_3mm.nii',...
%                           '-od', 'D:\yueyingkeji\tange');
% 

help lc_segmentation_winner_take_all
fprintf('-----------------------------------------------------------\n');

if nargin == 0
    help lc_segmentation_winner_take_all
    return
end

% Varargin parser
mask_file = '';
out_dir = pwd;
n_workers = 4;

if( sum(or(strcmpi(varargin,'--data_dir'),strcmpi(varargin,'-dd')))==1)
    data_dir = varargin{find(or(strcmpi(varargin,'--data_dir'),strcmp(varargin,'-dd')))+1};
else
    help lc_segmentation_winner_take_all
    error('Please specify data_dir!');
end

if( sum(or(strcmpi(varargin,'--atalas'),strcmpi(varargin,'-as')))==1)
    atalas = varargin{find(or(strcmpi(varargin,'--atalas'),strcmp(varargin,'-as')))+1};
else
    help lc_segmentation_winner_take_all
    error('Please specify atalas!');
end

if( sum(or(strcmpi(varargin,'--network_index'),strcmpi(varargin,'-ni')))==1)
    network_index = varargin{find(or(strcmpi(varargin,'--network_index'),strcmp(varargin,'-ni')))+1};
else
    help lc_segmentation_winner_take_all
    error('Please specify network_index!');
end

if( sum(or(strcmpi(varargin,'--roi2segment'),strcmpi(varargin,'-rs')))==1)
    roi2segment = varargin{find(or(strcmpi(varargin,'--roi2segment'),strcmp(varargin,'-rs')))+1};
else
    help lc_segmentation_winner_take_all
    error('Please specify roi2segment!');
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

% Load ROI
[roi, header_roi] = y_Read(roi2segment);
[dim1, dim2, dim3] = size(roi);
roi_mask = reshape(roi, dim1*dim2*dim3, [])' ~= 0;

% Load external mask
if ~strcmp(mask_file, '')
    mask_org = y_Read(mask_file) ~= 0;
    if (~all(size(mask_org) == size(roi)))
        disp('Dimension of the mask did not match roi!');
        return;
    end
else
    mask_org = ones(dim1, dim2, dim3) == 1;
end
mask = reshape(mask_org, dim1*dim2*dim3, [])' ~= 0;
roi_mask = logical(roi_mask .* mask);

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

% Load atalas
[atalas, header_atalas] = y_Read(atalas);
atalas = atalas .* mask_org;

% Load network_index
network_index = importdata(network_index);

try
    parpool(n_workers)
catch
    fprintf('Parpool already running!\n')
end

% Get one data dimension
data_target = y_Read(datafile_path{1});
data_target = reshape(data_target, dim1*dim2*dim3, [])';
    
coef_all = 0;
parfor i = 1:n_sub
    fprintf('Running %d/%d\n', i, n_sub);
    disp('----------------------------------');

    % Step 1: Extract target signals
    data_target = y_Read(datafile_path{i});
    data_target = reshape(data_target, dim1*dim2*dim3, [])';
    roi_signal = data_target(:, roi_mask);
    
    % Step 2 is to extract the the average time series of the other regions (some regions are combined with one functional region).
    network_index_excluded_target_id = setdiff(network_index, []);
    other_regions_id = arrayfun(@(id)(find(network_index == id)), network_index_excluded_target_id, 'UniformOutput',false);
    
    num_other_network = length(network_index_excluded_target_id);
    atalas_combined_all_all=0;
    for j = 1:num_other_network
        atalas_combined = arrayfun(@(id)(atalas == id), other_regions_id{j}, 'UniformOutput',false);
        num_id = length(atalas_combined);
        atalas_combined_all = 0;
        for k = 1: num_id
            atalas_combined_all = atalas_combined_all + atalas_combined{k};
        end
        atalas_combined_all_all = atalas_combined_all_all + atalas_combined_all .* network_index_excluded_target_id(j);
    end
    average_signals_other_regions = y_ExtractROISignal_copy(datafile_path{i}, {atalas_combined_all_all},[], atalas, 1);
    
    % Step 3 is to calculate the partial correlations between time series of all voxels in the target brain region and average time series of the other regions.
    coef = zeros(num_other_network, size(roi_signal,2));
    for j = 1 : num_other_network
        cov = average_signals_other_regions;
        cov(:,j) = [];
        coef(j,:) = partialcorr(roi_signal, average_signals_other_regions(:,j), cov, 'Type','Pearson');
    end
    
    % Step 4 is to average the partial correlations across all participants (sum then be devided by nsub).
    coef_all = coef_all + coef;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% Step 4:continue
coef = coef_all ./ n_sub;
 
% Step 5 is to segment the target region into several sub-regions.
coef_max = max(abs(coef));
nVoxInROI = sum(roi_mask(:));
segmentation = zeros(1,nVoxInROI);
for j = 1: nVoxInROI
    segmentation(j) = find(abs(coef(:,j)) == coef_max(j));
end
% clear atalas  atalas_combined atalas_combined_all average_signals_other_regions coef coef_all

% Step 6 is to save the sub-regions.
seg = zeros(1,dim1 * dim2 * dim3);
seg(roi_mask) = segmentation;
segmentation = reshape(seg, dim1, dim2, dim3);
header = header_atalas;
header.descrip = 'region segmentation';
y_Write(segmentation, header, fullfile(out_dir,'segmentation.nii'));
disp('Done!');
end


function [ROISignals] = y_ExtractROISignal_copy(AllVolume, ROIDef, OutputName, MaskData, IsMultipleLabel, IsNeedDetrend, Band, TR, TemporalMask, ScrubbingMethod, ScrubbingTiming, Header, CUTNUMBER)             
% NOTE. This function is modified from DPABI.
% Written by YAN Chao-Gan 120216 based on fc.m.
% The Nathan Kline Institute for Psychiatric Research, 140 Old Orangeburg Road, Orangeburg, NY 10962, USA
% Child Mind Institute, 445 Park Avenue, New York, NY 10022, USA
% The Phyllis Green and Randolph Cowen Institute for Pediatric Neuroscience, New York University Child Study Center, New York, NY 10016, USA
% ycg.yan@gmail.com

if ~exist('IsMultipleLabel','var')
    IsMultipleLabel = 0;
end

if ~exist('CUTNUMBER','var')
    CUTNUMBER = 10;
end

theElapsedTime =cputime;
% fprintf('\n\t Extracting ROI signals...');

if ~isnumeric(AllVolume)
    [AllVolume,VoxelSize,theImgFileList, Header] =y_ReadAll(AllVolume);
end

AllVolume(find(isnan(AllVolume))) = 0; %YAN Chao-Gan, 171022. Set the NaN voxels to 0.

[nDim1 nDim2 nDim3 nDimTimePoints]=size(AllVolume);
BrainSize = [nDim1 nDim2 nDim3];
VoxelSize = sqrt(sum(Header.mat(1:3,1:3).^2));


if ischar(MaskData)
    if ~isempty(MaskData)
        [MaskData,MaskVox,MaskHead]=y_ReadRPI(MaskData);
    else
        MaskData=ones(nDim1,nDim2,nDim3);
    end
end

% Convert into 2D
AllVolume=reshape(AllVolume,[],nDimTimePoints)';
% AllVolume=permute(AllVolume,[4,1,2,3]); % Change the Time Course to the first dimention
% AllVolume=reshape(AllVolume,nDimTimePoints,[]);

MaskDataOneDim=reshape(MaskData,1,[]);
MaskIndex = find(MaskDataOneDim);
AllVolume=AllVolume(:,MaskIndex);

% Scrubbing
if exist('TemporalMask','var') && ~isempty(TemporalMask) && ~strcmpi(ScrubbingTiming,'AfterFiltering')
    if ~all(TemporalMask)
        AllVolume = AllVolume(find(TemporalMask),:); %'cut'
        if ~strcmpi(ScrubbingMethod,'cut')
            xi=1:length(TemporalMask);
            x=xi(find(TemporalMask));
            AllVolume = interp1(x,AllVolume,xi,ScrubbingMethod);
        end
        nDimTimePoints = size(AllVolume,1);
    end
end


% Detrend
if exist('IsNeedDetrend','var') && IsNeedDetrend==1
    %AllVolume=detrend(AllVolume);
    fprintf('\n\t Detrending...');
    SegmentLength = ceil(size(AllVolume,2) / CUTNUMBER);
    for iCut=1:CUTNUMBER
        if iCut~=CUTNUMBER
            Segment = (iCut-1)*SegmentLength+1 : iCut*SegmentLength;
        else
            Segment = (iCut-1)*SegmentLength+1 : size(AllVolume,2);
        end
        AllVolume(:,Segment) = detrend(AllVolume(:,Segment));
        fprintf('.');
    end
end

% Filtering
if exist('Band','var') && ~isempty(Band)
    fprintf('\n\t Filtering...');
    SegmentLength = ceil(size(AllVolume,2) / CUTNUMBER);
    for iCut=1:CUTNUMBER
        if iCut~=CUTNUMBER
            Segment = (iCut-1)*SegmentLength+1 : iCut*SegmentLength;
        else
            Segment = (iCut-1)*SegmentLength+1 : size(AllVolume,2);
        end
        AllVolume(:,Segment) = y_IdealFilter(AllVolume(:,Segment), TR, Band);
        fprintf('.');
    end
end



% Scrubbing after filtering
if exist('TemporalMask','var') && ~isempty(TemporalMask) && strcmpi(ScrubbingTiming,'AfterFiltering')
    if ~all(TemporalMask)
        AllVolume = AllVolume(find(TemporalMask),:); %'cut'
        if ~strcmpi(ScrubbingMethod,'cut')
            xi=1:length(TemporalMask);
            x=xi(find(TemporalMask));
            AllVolume = interp1(x,AllVolume,xi,ScrubbingMethod);
        end
        nDimTimePoints = size(AllVolume,1);
    end
end


% Extract the Seed Time Courses

SeedSeries = [];
MaskROIName=[];

for iROI=1:length(ROIDef)
    IsDefinedROITimeCourse =0;
    if strcmpi(int2str(size(ROIDef{iROI})),int2str([nDim1, nDim2, nDim3]))  %ROI Data
        MaskROI = ROIDef{iROI};
        MaskROIName{iROI} = sprintf('Mask Matrix definition %d',iROI);
    elseif size(ROIDef{iROI},1) == nDimTimePoints %Seed series% strcmpi(int2str(size(ROIDef{iROI})),int2str([nDimTimePoints, 1])) %Seed series
        SeedSeries{1,iROI} = ROIDef{iROI};
        IsDefinedROITimeCourse =1;
        MaskROIName{iROI} = sprintf('Seed Series definition %d',iROI);
    elseif strcmpi(int2str(size(ROIDef{iROI})),int2str([1, 4]))  %Sphere ROI definition: CenterX, CenterY, CenterZ, Radius
        MaskROI = y_Sphere(ROIDef{iROI}(1:3), ROIDef{iROI}(4), Header);
        MaskROIName{iROI} = sprintf('Sphere definition (CenterX, CenterY, CenterZ, Radius): %g %g %g %g.',ROIDef{iROI});
    elseif exist(ROIDef{iROI},'file')==2    % Make sure the Definition file exist
        [pathstr, name, ext] = fileparts(ROIDef{iROI});
        if strcmpi(ext, '.txt'),
            TextSeries = load(ROIDef{iROI});
            if IsMultipleLabel == 1
                for iElement=1:size(TextSeries,2)
                    MaskROILabel{1,iROI}{iElement,1} = ['Column ',num2str(iElement)];
                end
                SeedSeries{1,iROI} = TextSeries;
            else
                SeedSeries{1,iROI} = mean(TextSeries,2);
            end
            IsDefinedROITimeCourse =1;
            MaskROIName{iROI} = ROIDef{iROI};
        elseif strcmpi(ext, '.img') || strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
            %The ROI definition is a mask file
            
            MaskROI=y_ReadRPI(ROIDef{iROI});
            if ~strcmpi(int2str(size(MaskROI)),int2str([nDim1, nDim2, nDim3]))
                error(sprintf('\n\tMask does not match.\n\tMask size is %dx%dx%d, not same with required size %dx%dx%d',size(MaskROI), [nDim1, nDim2, nDim3]));
            end

            MaskROIName{iROI} = ROIDef{iROI};
        else
            error(sprintf('Wrong ROI file type, please check: \n%s', ROIDef{iROI}));
        end
        
    else
        error(sprintf('File doesn''t exist or wrong ROI definition, please check: %s.\n', ROIDef{iROI}));
    end

    if ~IsDefinedROITimeCourse
        % Speed up! YAN Chao-Gan 101010.
        MaskROI=reshape(MaskROI,1,[]);
        MaskROI=MaskROI(MaskIndex); %Apply the brain mask
        
        if IsMultipleLabel == 1
            Element = unique(MaskROI);
            Element(find(isnan(Element))) = []; % ignore background if encoded as nan. Suggested by Dr. Martin Dyrba
            Element(find(Element==0)) = []; % This is the background 0
            SeedSeries_MultipleLabel = zeros(nDimTimePoints,length(Element));
            for iElement=1:length(Element)
                
                SeedSeries_MultipleLabel(:,iElement) = mean(AllVolume(:,find(MaskROI==Element(iElement))),2);
                
                MaskROILabel{1,iROI}{iElement,1} = num2str(Element(iElement));

            end
            SeedSeries{1,iROI} = SeedSeries_MultipleLabel;
        else
            SeedSeries{1,iROI} = mean(AllVolume(:,find(MaskROI)),2);
        end
    end
end


%Merge the seed series cell into seed series matrix
ROISignals = double(cell2mat(SeedSeries)); %Suggested by H. Baetschmann.    %ROISignals = cell2mat(SeedSeries);

theElapsedTime = cputime - theElapsedTime;
% fprintf('\n\t Extracting ROI signals finished, elapsed time: %g seconds.\n', theElapsedTime);
end