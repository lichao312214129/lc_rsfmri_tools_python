function move_subject_files_according_predict_label()
% 根据预测标签，将被试的图象分别移动到相应文件夹下
% input
imgPath='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\multilabel\smALFF_all';
labelFile='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\multilabel\predict_label_multilabel.xlsx';
output_folder='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\multilabel\smALFF_e';
which_label=5;
%%
prdLabel=xlsread(labelFile);

%
subStrut=dir(fullfile(imgPath,'*.nii'));
subj={subStrut.name}';
% subj=subj(1:282);
subj_a=subj(prdLabel==which_label);

% copy subject alff
n=numel(subj_a);
for i=1:n
    fprintf('%d/%d\n',i,n)
    beCopyFile=fullfile(imgPath,subj_a{i});
    copyfile(beCopyFile,output_folder);
end
end