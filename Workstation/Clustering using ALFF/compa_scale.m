%% 比较被聚类模型分为2类的被试的量表差异
%% input
scale_path_test='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\REST-meta-MDD-PhenotypicData_WithHAMDSubItem_S20.xlsx';
predict_label_path='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Machine_Learning\predictLabel_testData.xlsx';
%% load
[scale_data,header]=xlsread(scale_path_test);
header=header(1,2:end);
[predict_label]=xlsread(predict_label_path);

%% 分组
scale_data_a=scale_data(predict_label==1,:);
scale_data_noa=scale_data(predict_label==0,:);


%% 各个量表的ind,筛选
[~,ind_age]=ismember('年龄',header);
[~,ind_edu]=ismember('教育年限',header);
[~,ind_duration]=ismember('病程（月）',header);
[~,ind_hamd]=ismember('HAMD',header);
[~,ind_hama]=ismember('HAMA',header);
[~,ind_first]=ismember('是否首发',header);
[~,ind_drug]=ismember('是否正在用药',header);

% age
age_a=scale_data_a(:,ind_age);
age_noa=scale_data_noa(:,ind_age);

% edu
edu_a=scale_data_a(:,ind_edu);
edu_noa=scale_data_noa(:,ind_edu);
%duration
duration_a=scale_data_a(:,ind_duration);
duration_noa=scale_data_noa(:,ind_duration);
%hamd
a_hamd=scale_data_a(:,ind_hamd);
noa_hamd=scale_data_noa(:,ind_hamd);
%hama
a_hama=scale_data_a(:,ind_hama);
noa_hama=scale_data_noa(:,ind_hama);
%first
first_a=scale_data_a(:,ind_first);
first_noa=scale_data_noa(:,ind_first);
%drug
a_drug=scale_data_a(:,ind_drug);
noa_drug=scale_data_noa(:,ind_drug);

%% save 
% xlswrite({'group','age','edu','duration','hamd','hama','first','drug'})
% xlswrite('scale.xlsx',[age_a;edu_a,duration_a,hamd_a,hama_a,first_a,drug_a;
%                     age_a;edu_a,duration_a,hamd_a,hama_a,first_a,drug_a]);

%% compare
age_describe=[[mean(age_a),std(age_a)];[mean(age_noa),std(age_noa)]];
[h_age,p_age]=ttest2(age_a(~isnan(age_a)),age_noa(~isnan(age_noa)));

edu_describe=[[mean(edu_a),std(edu_a)];[mean(edu_noa),std(edu_noa)]];
[h_edu,p_edu]=ttest2(edu_a(~isnan(edu_a)),edu_noa(~isnan(edu_noa)));

duration_describe=[mean(duration_a(~isnan(duration_a))),std(duration_a(~isnan(duration_a)));
                    mean(duration_noa(~isnan(duration_noa))),std(duration_noa(~isnan(duration_noa)))];
[h_duration,p_duration]=ttest2(duration_a(~isnan(duration_a)),duration_noa(~isnan(duration_noa)));

hamd_describe=[mean(a_hamd(~isnan(a_hamd))),std(a_hamd(~isnan(a_hamd)));mean(noa_hamd(~isnan(noa_hamd))),std(noa_hamd(~isnan(noa_hamd)))];
[h_hamd,p_hamd]=ttest2(a_hamd(~isnan(a_hamd)),noa_hamd(~isnan(noa_hamd)));

hama_describe=[mean(a_hama(~isnan(a_hama))),std(a_hama(~isnan(a_hama)));
                mean(noa_hama(~isnan(noa_hama))),std(noa_hama(~isnan(noa_hama)))];
[h_hama,p_hama]=ttest2(a_hama(~isnan(a_hama)),noa_hama(~isnan(noa_hama)));

% first_describe=[sum(first_a==1),sum(first_a==-1); sum(first_noa==1),sum(first_noa==-1)];
% [p_first, Q_first]= chi2test_LiuFeng([sum(first_a==1),sum(first_a==-1); sum(first_noa==1),sum(first_noa==-1)]);

drug_describe=[sum(a_drug==1),sum(a_drug==-1); sum(noa_drug==1),sum(noa_drug==-1)];
[p_drug, Q_drug]= chi2test_LiuFeng([sum(a_drug==1),sum(a_drug==-1); sum(noa_drug==1),sum(noa_drug==-1)]);
% multi corr
[Results] = multcomp_bonferroni([p_age,p_edu,p_duration,p_hamd,p_hama,p_drug]);

%% 用药与未用药分开
% a
a_hamd_drug=a_hamd(a_drug==1);
a_hamd_no_drug=a_hamd(a_drug==-1);

a_hama_drug=a_hama(a_drug==1);
a_hama_no_drug=a_hama(a_drug==-1);

a_duration_drug=duration_a(a_drug==1);
a_duration_no_drug=duration_a(a_drug==-1);

% non-a
noa_hamd_drug=noa_hamd(noa_drug==1);
noa_hamd_no_drug=noa_hamd(noa_drug==-1);

noa_hama_drug=noa_hama(noa_drug==1);
noa_hama_no_drug=noa_hama(noa_drug==-1);

noa_duration_drug=duration_noa(noa_drug==1);
noa_duration_no_drug=duration_noa(noa_drug==-1);

%% ttest2
% a
[a_hamd_h,a_hamd_p]=ttest2(a_hamd_drug,a_hamd_no_drug);
[a_hama_h,a_hama_p]=ttest2(a_hama_drug,a_hama_no_drug);
[a_h_duration,a_p_duration]=ttest2(a_duration_drug,a_duration_no_drug);

% noa
[noa_hamd_h,noa_hamd_p]=ttest2(noa_hamd_drug,noa_hamd_no_drug);
[noa_hama_h,noa_hama_p]=ttest2(noa_hama_drug,noa_hama_no_drug);
[noa_h_duration,noa_p_duration]=ttest2(noa_duration_drug,noa_duration_no_drug);
% hamd~label*medication
all_hamd=[a_hamd_drug;a_hamd_no_drug;noa_hamd_drug;noa_hamd_no_drug];
all_label=[ones(length([a_hamd_drug;a_hamd_no_drug]),1);zeros(length([noa_hamd_drug;noa_hamd_no_drug]),1)];
all_drug=[ones(length(a_hamd_drug),1);zeros(length(a_hamd_no_drug),1);...
          ones(length(noa_hamd_drug),1);zeros(length(noa_hamd_no_drug),1)];

%
ind_hamd_nan=isnan(all_hamd);
all_hamd=all_hamd(~ind_hamd_nan);
all_label=all_label(~ind_hamd_nan);
all_drug=all_drug(~ind_hamd_nan);
Dataset=dataset(all_hamd,all_label,all_drug);
      
% modelspec = 'all_hamd ~ all_label:all_drug';
T = [0 0 0;1 0 0;0 1 0;1 1 0];
mdl = fitglm(Dataset,T);
% 'MPG ~ Acceleration + Weight + Acceleration:Weight + Weight^2';

% mdl = fitlm(Dataset(:,2:3),Dataset(:,1),T);

%% plot bar
%mean and std
a_mean=[mean(a_hamd_drug(~isnan(a_hamd_drug))),mean(a_hamd_no_drug(~isnan(a_hamd_no_drug)));
       mean(a_hama_drug(~isnan(a_hama_drug))),mean(a_hama_no_drug(~isnan(a_hama_no_drug)));
       mean(a_duration_drug(~isnan(a_duration_drug))),mean(a_duration_no_drug(~isnan(a_duration_no_drug)));
       mean(noa_hamd_drug(~isnan(noa_hamd_drug))),mean(noa_hamd_no_drug(~isnan(noa_hamd_no_drug)));
       mean(noa_hama_drug(~isnan(noa_hama_drug))),mean(noa_hama_no_drug(~isnan(noa_hama_no_drug)));
       mean(noa_duration_drug(~isnan(noa_duration_drug))),mean(noa_duration_no_drug(~isnan(noa_duration_no_drug)))];

a_std=[std(a_hamd_drug(~isnan(a_hamd_drug))),std(a_hamd_no_drug(~isnan(a_hamd_no_drug)));
       std(a_hama_drug(~isnan(a_hama_drug))),std(a_hama_no_drug(~isnan(a_hama_no_drug)));
       std(a_duration_drug(~isnan(a_duration_drug))),std(a_duration_no_drug(~isnan(a_duration_no_drug)));
       std(noa_hamd_drug(~isnan(noa_hamd_drug))),std(noa_hamd_no_drug(~isnan(noa_hamd_no_drug)));
       std(noa_hama_drug(~isnan(noa_hama_drug))),std(noa_hama_no_drug(~isnan(noa_hama_no_drug)));
       std(noa_duration_drug(~isnan(noa_duration_drug))),std(noa_duration_no_drug(~isnan(noa_duration_no_drug)))];

 % plot
bar_errorbar_innder(a_mean,a_std)
