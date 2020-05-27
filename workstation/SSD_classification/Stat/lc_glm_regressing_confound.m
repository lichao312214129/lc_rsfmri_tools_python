% This script is used to regressing confounds: age, sex, headmotion and site.
% Before running this script, make sure you have run lc_demographic_information_statistics.py

% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';

% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

%% GLM: Building model of site in HC, and applied to all
independent_variables_all = demographic(:,2:end);
loc_all_hc = (independent_variables_all(:,1)==0);

independent_variables_all_hc = independent_variables_all(loc_all_hc,:);
dependent_variables_all_hc = data(loc_all_hc,3:end);

% Regressing site
beta_value_all_hc = independent_variables_all_hc(:,[3 4 5])\dependent_variables_all_hc;
resid_all =  data(:,3:end) - independent_variables_all(:,[3 4 5])*beta_value_all_hc;

% Regressing age, sex and headmotion
beta_value = independent_variables_all(:,[3,4])\resid_all;
resid_all=resid_all-independent_variables_all(:,[3,4])*beta_value;

% Concat
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\resid_all.mat', 'resid_all');

resid_all = cat(2, data(:,1:2),demographic(:,end), demographic(:,[3,4,5]));
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\resid_all.mat', 'resid_all');

%% GLM separately (Patient and HC are separate)
independent_variables_all = demographic(:,2:end);

loc_tr_p = (independent_variables_all(:,1)==1) & (independent_variables_all(:,5)~=0);
loc_tr_hc = (independent_variables_all(:,1)==0) & (independent_variables_all(:,5)~=0);
loc_te_p = (independent_variables_all(:,1)==1) & (independent_variables_all(:,5)==0);
loc_te_hc = (independent_variables_all(:,1)==0) & (independent_variables_all(:,5)==0);

independent_variables_train_patient = independent_variables_all(loc_tr_p,:);
independent_variables_train_hc = independent_variables_all(loc_tr_hc,:);
dependent_variables_train_patient = data(loc_tr_p,3:end);
dependent_variables_train_hc = data(loc_tr_hc,3:end);

independent_variables_test_patient = independent_variables_all(loc_te_p,:);
independent_variables_test_hc = independent_variables_all(loc_te_hc,:);
dependent_variables_test_patient = data(loc_te_p,3:end);
dependent_variables_test_hc = data(loc_te_hc,3:end);

beta_value_patient=independent_variables_train_patient(:,[3,4])\dependent_variables_train_patient;
beta_value_hc = independent_variables_train_hc(:,[3,4])\dependent_variables_train_hc;


% Get residual error
resid_train_patient=dependent_variables_train_patient-independent_variables_train_patient(:,[3,4])*beta_value_patient;
resid_train_hc=dependent_variables_train_hc-independent_variables_train_hc(:,[3,4])*beta_value_hc;

resid_test_patient=dependent_variables_test_patient-independent_variables_test_patient(:,[3,4])*beta_value_patient;
resid_test_hc=dependent_variables_test_hc-independent_variables_test_hc(:,[3,4])*beta_value_hc;

corr(mean(resid_train_patient)', mean(resid_test_patient)')
corr(mean(dependent_variables_train_patient)', mean(dependent_variables_test_patient)')
corr(mean(resid_train_hc)', mean(resid_test_hc)')
corr(mean(dependent_variables_train_hc)', mean(dependent_variables_test_hc)')

% Concat
resid_all = zeros(size(data(:,3:end)));
resid_all(loc_tr_p,:) = resid_train_patient;
resid_all(loc_tr_hc,:) = resid_train_hc;
resid_all(loc_te_p,:) = resid_test_patient;
resid_all(loc_te_hc,:) = resid_test_hc;

resid_all = cat(2, data(:,1:2), demographic(:,[6]), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\resid_all.mat', 'resid_all');

%% GLM separately (Patient and HC are together)
independent_variables_all = demographic(:,2:end);

loc_p = (independent_variables_all(:,1)==1);
loc_hc = (independent_variables_all(:,1)==0);

independent_variables_patient = independent_variables_all(loc_p,:);
independent_variables_hc = independent_variables_all(loc_hc,:);
dependent_variables_patient = data(loc_p,3:end);
dependent_variables_hc = data(loc_hc,3:end);

beta_value_patient = independent_variables_patient(:,3:end-1)\dependent_variables_patient;
beta_value_hc = independent_variables_hc(:,3:end-1)\dependent_variables_hc;

% Get residual error
resid_patient=dependent_variables_patient-independent_variables_patient(:,3:end-1)*beta_value_patient;
resid_hc=dependent_variables_hc-independent_variables_hc(:,3:end-1)*beta_value_hc;

% Concat
corr(mean(resid_patient)',mean(resid_hc)')
resid_all = zeros(size(data(:,3:end)));
resid_all(loc_p,:) = resid_patient;
resid_all(loc_hc,:) = resid_hc;

resid_all = cat(2, data(:,1:2), demographic(:,[6]), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\resid_all.mat', 'resid_all');


%% GLM all
independent_variables_all = demographic(:,2:end);
dependent_variables_all = data(:,3:end);
beta_value = independent_variables_all(:,2:end)\dependent_variables_all;

% Get residual error
resid_all=dependent_variables_all-independent_variables_all(:,2:end)*beta_value;
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\resid_all.mat', 'resid_all');

%% Compare
independent_variables_all = demographic(:,end);
dependent_variables_all = data(:,3:end);
design = zeros(1077,4);
for  i = 1:4
    design(:,i) = independent_variables_all == (i-1);
end
design = cat(2, design, demographic(:,3:end-1));

contrast = [0 0 0 0 0 0 1];
test_type = 'ftest';
[tstat,pvalue, beta_value] = el_glm(design, dependent_variables_all, contrast, test_type);


%% Compare age, sex, headmotion
independent_variables_all = demographic(:,end);
dependent_variables_all =  demographic(:,3:5);
design = zeros(1077,4);
for  i = 1:4
    design(:,i) = independent_variables_all == (i-1);
end

contrast = [1 0 0 -1];
test_type = 'ttest';
[tstat, pvalue, beta_value] = el_glm(design, dependent_variables_all, contrast, test_type);

%% Take cov as features
cov = cat(2, data, demographic(:,[3,4,5]));