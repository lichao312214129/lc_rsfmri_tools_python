%% 将ALFF map 标准化，方便dpabi统计
function lc_standardization_3D(inpath,outpath,method,mask)
if nargin<1
    inpath='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\validation_wholeBrain_all';
    outpath='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\mALFF_All';
    method='demean';
    mask='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\code_workstation2018_ALFF_clustering\Group_mask.nii';
end
%
mkdir(outpath);
mask=y_Read(mask)==1;
%
[path_subj,name_subj]=load_all_nii(inpath);

[data_all,Header]=nii_to_one_mat(path_subj);

standardization(data_all,name_subj,method,outpath,Header,mask);

end

function [path_subj,name_subj]=load_all_nii(inpath)
path_struct=dir(fullfile(inpath,'*.nii'));
path_subj=fullfile(inpath,{path_struct.name}');
name_subj={path_struct.name}';
end

function [data_all,Header]=nii_to_one_mat(path_subj)
% Header=每个被试header
[data_all]=y_ReadAll(path_subj);
Header=cell(length(path_subj),1);
for isubj=1:length(path_subj)
    [~,Header{isubj}]=y_Read(path_subj{isubj});
end
end

function standardization(data_all,name_subj,method,outpath,Header,mask)

[dim1,dim2,dim3,n_subj]=size(data_all);

if nargin<6
    mask=ones(dim1,dim2,dim3)==1;
end

if strcmp(method,'zscore')
    for isubj=1:n_subj
        fprintf('%d/%d\n',isubj,n_subj);
        data_one_subj=data_all(:,:,:,isubj);
        z_data_one_subj=(data_one_subj-mean(data_one_subj(mask)))./std(data_one_subj(mask));
        
        % write to nii
        y_Write(z_data_one_subj,Header{isubj},fullfile(outpath,name_subj{isubj}));
    end
elseif strcmp(method,'demean')
    for isubj=1:n_subj
        fprintf('%d/%d\n',isubj,n_subj);
        data_one_subj=data_all(:,:,:,isubj);
        z_data_one_subj=data_one_subj./mean(data_one_subj(mask));
        
        % write to nii
        y_Write(z_data_one_subj,Header{isubj},fullfile(outpath,name_subj{isubj}));
    end
end
end