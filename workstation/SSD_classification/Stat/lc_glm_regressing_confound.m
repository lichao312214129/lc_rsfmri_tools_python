% This script is used to regressing confounds: age, sex, headmotion and site.
% Before running this script, make sure you have run lc_demographic_information_statistics.py

% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';

% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% % Headmotion compare
% hm = demographic(:,5);
% site = demographic(:,end);
% [h,p]=ttest2(hm(site==0), hm(site~=0))
% 
% mean(hm(site==0))
% mean(hm(site~=0))

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

% Regress out site in HC, and applied to all
independent_variables_all = demographic(:,2:end);
loc_all_hc = (independent_variables_all(:,1)==0);
independent_variables_all_hc = independent_variables_all(loc_all_hc,:);
dependent_variables_all_hc = data(loc_all_hc,3:end);
beta_value_all_hc = independent_variables_all_hc(:,[5])\dependent_variables_all_hc;
resid_all =  data(:,3:end) - independent_variables_all(:,[5])*beta_value_all_hc;

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

beta_value_patient=independent_variables_train_patient(:,[2,3,4])\dependent_variables_train_patient;
beta_value_hc = independent_variables_train_hc(:,[2,3,4])\dependent_variables_train_hc;

% Get residual error
resid_train_patient=dependent_variables_train_patient-independent_variables_train_patient(:,[2,3,4])*beta_value_patient;
resid_train_hc=dependent_variables_train_hc-independent_variables_train_hc(:,[2,3,4])*beta_value_hc;

resid_test_patient=dependent_variables_test_patient-independent_variables_test_patient(:,[2, 3,4])*beta_value_patient;
resid_test_hc=dependent_variables_test_hc-independent_variables_test_hc(:,[2,3,4])*beta_value_hc;

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

%% GLM with each site separately, but exception of the site

% Regress out site in HC, and applied to all
site_design = zeros(size(demographic,1),4);
for i = 1:4
    site_design(:,i) = demographic(:,end) == i-1;
end
independent_variables_all = site_design;
loc_all_hc = (independent_variables_all(:,1)==0);
independent_variables_all_hc = site_design(loc_all_hc,:);
dependent_variables_all_hc = data(loc_all_hc,3:end);
beta_value_all_hc = independent_variables_all_hc\dependent_variables_all_hc;
resid_all =  data(:,3:end) - independent_variables_all*beta_value_all_hc;

% Regress out age, sex and headmotion in each site
sex_design = zeros(size(demographic,1),2);
for i = 1:2
    sex_design(:,i) = demographic(:,4) == i-1;
end

loc_site1 = (demographic(:,6)==0);
loc_site2 = (demographic(:,6)==1);
loc_site3 = (demographic(:,6)==2);
loc_site4 = (demographic(:,6)==3);

independent_variables_site1 = [sex_design(loc_site1,:), demographic(loc_site1,[3 5])];
independent_variables_site2 = [sex_design(loc_site2,:), demographic(loc_site2,[3 5])];
independent_variables_site3 = [sex_design(loc_site3,:), demographic(loc_site3,[3 5])];
independent_variables_site4 = [sex_design(loc_site4,:), demographic(loc_site4,[3 5])];

dependent_variables_site1 = resid_all(loc_site1,:);
dependent_variables_site2 = resid_all(loc_site2,:);
dependent_variables_site3 = resid_all(loc_site3,:);
dependent_variables_site4 = resid_all(loc_site4,:);

beta_value_site1=independent_variables_site1\dependent_variables_site1;
beta_value_site2=independent_variables_site2\dependent_variables_site2;
beta_value_site3=independent_variables_site3\dependent_variables_site3;
beta_value_site4=independent_variables_site4\dependent_variables_site4;

% Get residual error
resid_site1=dependent_variables_site1-independent_variables_site1*beta_value_site1;
resid_site2=dependent_variables_site2-independent_variables_site2*beta_value_site2;
resid_site3=dependent_variables_site3-independent_variables_site3*beta_value_site3;
resid_site4=dependent_variables_site4-independent_variables_site4*beta_value_site4;

% Check
corr(data(:,4), resid_all(:,2))
corr(mean(resid_site1)', mean(resid_site2)')
corr(mean(dependent_variables_site1)', mean(dependent_variables_site2)')
corr(mean(resid_site2)', mean(resid_site4)')
corr(mean(dependent_variables_site4)', mean(dependent_variables_site2)')

% Concat
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

%% GLM on each diagnosis, each site separately, but exception of the site

% Regress out site in HC, and applied to all
site_design = zeros(size(demographic,1),4);
for i = 1:4
    site_design(:,i) = demographic(:,end) == i-1;
end
independent_variables_all = site_design;
loc_all_hc = (independent_variables_all(:,1)==0);
independent_variables_all_hc = site_design(loc_all_hc,:);
dependent_variables_all_hc = data(loc_all_hc,3:end);
beta_value_all_hc = independent_variables_all_hc\dependent_variables_all_hc;
resid_all =  data(:,3:end) - independent_variables_all*beta_value_all_hc;

% Regress out age, sex and headmotion in each site
sex_design = zeros(size(demographic,1),2);
for i = 1:2
    sex_design(:,i) = demographic(:,4) == i-1;
end

loc_site1_patient = (demographic(:,2)==1 & demographic(:,6)==0);
loc_site2_patient = (demographic(:,2)==1 & demographic(:,6)==1);
loc_site3_patient = (demographic(:,2)==1 & demographic(:,6)==2);
loc_site4_patient = (demographic(:,2)==1 & demographic(:,6)==3);
loc_site1_hc = (demographic(:,2)==0 & demographic(:,6)==0);
loc_site2_hc = (demographic(:,2)==0 & demographic(:,6)==1);
loc_site3_hc = (demographic(:,2)==0 & demographic(:,6)==2);
loc_site4_hc = (demographic(:,2)==0 & demographic(:,6)==3);

independent_variables_site1_patient = [sex_design(loc_site1_patient,:), demographic(loc_site1_patient,[3 5])];
independent_variables_site2_patient = [sex_design(loc_site2_patient,:), demographic(loc_site2_patient,[3 5])];
independent_variables_site3_patient = [sex_design(loc_site3_patient,:), demographic(loc_site3_patient,[3 5])];
independent_variables_site4_patient = [sex_design(loc_site4_patient,:), demographic(loc_site4_patient,[3 5])];
independent_variables_site1_hc = [sex_design(loc_site1_hc,:), demographic(loc_site1_hc,[3 5])];
independent_variables_site2_hc = [sex_design(loc_site2_hc,:), demographic(loc_site2_hc,[3 5])];
independent_variables_site3_hc = [sex_design(loc_site3_hc,:), demographic(loc_site3_hc,[3 5])];
independent_variables_site4_hc = [sex_design(loc_site4_hc,:), demographic(loc_site4_hc,[3 5])];

dependent_variables_site1_patient = resid_all(loc_site1_patient,:);
dependent_variables_site2_patient = resid_all(loc_site2_patient,:);
dependent_variables_site3_patient = resid_all(loc_site3_patient,:);
dependent_variables_site4_patient = resid_all(loc_site4_patient,:);
dependent_variables_site1_hc = resid_all(loc_site1_hc,:);
dependent_variables_site2_hc = resid_all(loc_site2_hc,:);
dependent_variables_site3_hc = resid_all(loc_site3_hc,:);
dependent_variables_site4_hc = resid_all(loc_site4_hc,:);

% Get beta
beta_value_site1_patient=independent_variables_site1_patient\dependent_variables_site1_patient;
beta_value_site2_patient=independent_variables_site2_patient\dependent_variables_site2_patient;
beta_value_site3_patient=independent_variables_site3_patient\dependent_variables_site3_patient;
beta_value_site4_patient=independent_variables_site4_patient\dependent_variables_site4_patient;
beta_value_site1_hc=independent_variables_site1_hc\dependent_variables_site1_hc;
beta_value_site2_hc=independent_variables_site2_hc\dependent_variables_site2_hc;
beta_value_site3_hc=independent_variables_site3_hc\dependent_variables_site3_hc;
beta_value_site4_hc=independent_variables_site4_hc\dependent_variables_site4_hc;

% Get residual error
resid_site1_patient=dependent_variables_site1_patient-independent_variables_site1_patient*beta_value_site1_patient;
resid_site2_patient=dependent_variables_site2_patient-independent_variables_site2_patient*beta_value_site2_patient;
resid_site3_patient=dependent_variables_site3_patient-independent_variables_site3_patient*beta_value_site3_patient;
resid_site4_patient=dependent_variables_site4_patient-independent_variables_site4_patient*beta_value_site4_patient;
resid_site1_hc=dependent_variables_site1_hc-independent_variables_site1_hc*beta_value_site1_hc;
resid_site2_hc=dependent_variables_site2_hc-independent_variables_site2_hc*beta_value_site2_hc;
resid_site3_hc=dependent_variables_site3_hc-independent_variables_site3_hc*beta_value_site3_hc;
resid_site4_hc=dependent_variables_site4_hc-independent_variables_site4_hc*beta_value_site4_hc;

% Check
corr(mean(resid_site2_patient)', mean(resid_site4_patient)')

% Concat
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

%% GLM all (Excluded subjects with greater head motion)

% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';
% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% Exclude subjects with greater head motion
loc_acceptable_headmotion = demographic(:,5)<=0.3;
demographic = demographic(loc_acceptable_headmotion,:);
data = data(loc_acceptable_headmotion,:);

% Regress 
site_design = zeros(size(demographic,1),4);
for i = 1:4
    site_design(:,i) = demographic(:,end) == i-1;
end

independent_variables_all = cat(2,site_design, demographic(:,[4 5]));
beta_value_all = independent_variables_all\data(:,3:end);
resid_all =  data(:,3:end) - independent_variables_all*beta_value_all;

% beta_value_all = demographic(:,[3 5])\resid_all;
% resid_all = resid_all -demographic(:,[3 5])*beta_value_all;

% Get residual error
% resid_all = cat(2,site_design, demographic(:,[3 4 5]));
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_excluded_greater_fd_and_regressed_out_site_sex_motion_all.mat', 'resid_all');

%% GLM of site, sex and headmotionon training data and applied to test data(Exclude subjects with greater head motion)
% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';
% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% Exclude subjects with greater head motion
loc_acceptable_headmotion = demographic(:,5)<=0.3;
demographic = demographic(loc_acceptable_headmotion,:);
data = data(loc_acceptable_headmotion,:);
site_design = zeros(size(demographic,1),4);
for i = 1:4
    site_design(:,i) = demographic(:,end) == i-1;
end
loc_train = demographic(:,end) ~=0;

% Fit site
indep_site = site_design(loc_train,:);
dep = data(loc_train,:);
dep = dep(:,3:end);
beta_value_site_train = indep_site\dep;
resid_train =  dep - indep_site*beta_value_site_train;

indep_sex_headmotion_train = demographic(:,[4 5]);
indep_sex_headmotion_train = indep_sex_headmotion_train(loc_train,:);

% Fit sex and headmotion
beta_value_sex_headmotion_train = indep_sex_headmotion_train\resid_train;

% Regress out for all subjects
resid_all = data(:,3:end) - site_design*beta_value_site_train;
resid_all = resid_all - demographic(:,[4 5])*beta_value_sex_headmotion_train;

% Concat
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_excluded_greater_fd_and_regressed_out_site_sex_motion_separately.mat', 'resid_all');

%% GLM of sex and headmotion on training data and applied to test data(Exclude subjects with greater head motion)
% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';
% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% Exclude subjects with greater head motion
loc_acceptable_headmotion = demographic(:,5)<=0.3;
demographic = demographic(loc_acceptable_headmotion,:);
data = data(loc_acceptable_headmotion,:);
loc_train = demographic(:,end) ~=0;

% Fit sex and headmotion
indep_age_sex_headmotion_train = demographic(:,[4 5]);
dep_age_sex_headmotion_train = data(:,3:end);
beta_value_age_sex_headmotion_train = indep_age_sex_headmotion_train\dep_age_sex_headmotion_train;

% Regress out for all subjects
resid_all = data(:,3:end) - indep_age_sex_headmotion_train*beta_value_age_sex_headmotion_train;

% Concat
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_excluded_greater_fd_and_regressed_out_sex_motion_separately.mat', 'resid_all');

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

%% Compare age, sex, headmotion between patients and hc
loc_p_site1 = (demographic(:,2)==1) & (demographic(:,6)==0);
loc_c_site1 = (demographic(:,2)==0) & (demographic(:,6)==0);
loc_p_site234 = (demographic(:,2)==1) & (demographic(:,6)~=0);
loc_c_site234 = (demographic(:,2)==0) & (demographic(:,6)~=0);

% Age
age = demographic(:,3);
age_p_site1 = age(loc_p_site1);
age_c_site1 = age(loc_c_site1);

age_p_site234 = age(loc_p_site234);
age_c_site234 = age(loc_c_site234);

[h,p,ci, tstat] = ttest2(age_p_site1, age_c_site1);
[h,p,ci, tstat] = ttest2(age_p_site234, age_c_site234);

save('D:\WorkStation_2018\SZ_classification\Scale\age_p_site1', 'age_p_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\age_c_site1', 'age_c_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\age_p_site234', 'age_p_site234');
save('D:\WorkStation_2018\SZ_classification\Scale\age_c_site234', 'age_c_site234');

% Sex
sex = demographic(:,4);
sex_site1 = sex(demographic(:,6)==0);
sex_p_site1 = sex(loc_p_site1);
sex_c_site1 = sex(loc_c_site1);

sex_site234 = sex(demographic(:,6)~=0);
sex_p_site234 = sex(loc_p_site234);
sex_c_site234 = sex(loc_c_site234);

sex_p_all = cat(1, sex_p_site1, sex_p_site234);
sex_c_all = cat(1, sex_c_site1, sex_c_site234);

[p, Q]= chi2test_LiuFeng([sum(sex_p_site1==1), sum(sex_p_site1==0);...
                        sum(sex_c_site1==1), sum(sex_c_site1==0)]);  % Site1

[p, Q]= chi2test_LiuFeng([sum(sex_p_site234==1), sum(sex_p_site234==0);...
                        sum(sex_c_site234==1), sum(sex_c_site234==0)]); % Site 2 3 4 
                    
[p, Q]= chi2test_LiuFeng([sum(sex_p_all==1), sum(sex_p_all==0);...
                        sum(sex_c_all==1), sum(sex_c_all==0)]); % Site 1 2 3 4 

[p, Q]= chi2test_LiuFeng([sum(sex_site1==1), sum(sex_site1==0);...
                    sum(sex_site234==1), sum(sex_site234==0)]); % Site 1 2 3 4 
                


save('D:\WorkStation_2018\SZ_classification\Scale\sex_p_site1', 'sex_p_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\sex_c_site1', 'sex_c_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\sex_p_site234', 'sex_p_site234');
save('D:\WorkStation_2018\SZ_classification\Scale\sex_c_site234', 'sex_c_site234');

% Proportion of patients and healthy controls of each site
loc_site1 = demographic(:,6)==0;
loc_site2 = demographic(:,6)==1;
loc_site3 = demographic(:,6)==2;
loc_site4 = demographic(:,6)==3;

diagnosis=demographic(:,2);

diagnosis_site1 = diagnosis(loc_site1);
diagnosis_site2 = diagnosis(loc_site2);
diagnosis_site3 = diagnosis(loc_site3);
diagnosis_site4 = diagnosis(loc_site4);

[p, Q]= chi2test_LiuFeng([sum(diagnosis_site1==1), sum(diagnosis_site1==0);...
                        sum(diagnosis_site2==1), sum(diagnosis_site2==0);....
                        sum(diagnosis_site3==1), sum(diagnosis_site3==0);....
                        sum(diagnosis_site4==1), sum(diagnosis_site4==0)]);


% Difference of head motion between patients and healthy controls of each site
loc_p_site1 = (demographic(:,2)==1) & (demographic(:,6)==0);
loc_p_site2 = (demographic(:,2)==1) & (demographic(:,6)==1);
loc_p_site3 = (demographic(:,2)==1) & (demographic(:,6)==2);
loc_p_site4 = (demographic(:,2)==1) & (demographic(:,6)==3);
loc_c_site1 = (demographic(:,2)==0) & (demographic(:,6)==0);
loc_c_site2 = (demographic(:,2)==0) & (demographic(:,6)==1);
loc_c_site3 = (demographic(:,2)==0) & (demographic(:,6)==2);
loc_c_site4 = (demographic(:,2)==0) & (demographic(:,6)==3);

headmotion=demographic(:,5);

headmotion_p_site1 = headmotion(loc_p_site1);
headmotion_p_site2 = headmotion(loc_p_site2);
headmotion_p_site3 = headmotion(loc_p_site3);
headmotion_p_site4 = headmotion(loc_p_site4);
headmotion_c_site1 = headmotion(loc_c_site1);
headmotion_c_site2 = headmotion(loc_c_site2);
headmotion_c_site3 = headmotion(loc_c_site3);
headmotion_c_site4 = headmotion(loc_c_site4);
headmotion_p_site234 = headmotion(loc_p_site234);
headmotion_c_site234 = headmotion(loc_c_site234);

[h,p,ci, tstat] = ttest2(headmotion_p_site1, headmotion_c_site1);
[h,p,ci, tstat] = ttest2(headmotion_p_site2, headmotion_c_site2);
[h,p,ci, tstat] = ttest2(headmotion_p_site3, headmotion_c_site3);
[h,p,ci, tstat] = ttest2(headmotion_p_site4, headmotion_c_site4);
[h,p,ci, tstat] = ttest2(headmotion_p_site234, headmotion_c_site234);

save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_p_site1', 'headmotion_p_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_c_site1', 'headmotion_c_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_p_site234', 'headmotion_p_site234');
save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_c_site234', 'headmotion_c_site234');
%% Take cov as features
cov = cat(2, data, demographic(:,[3,4,5]));