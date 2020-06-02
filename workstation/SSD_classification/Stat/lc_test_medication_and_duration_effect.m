function lc_test_medication_and_duration_effect()
% This function is used to compare the chronic SSD with first episode medicated SSD, 
% as well as first episode medicated SSD with first episode unmedicated SSD 
% Statistical method is NBS.
% Refer and thanks to NBS (NBSglm and NBSstats)

%% Inputs
if nargin < 1
    cov_chronic = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cov_chronic.mat';
    cov_firstepisode_medicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cov_firstepisode_medicated.mat';
    cov_firstepisode_unmedicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cov_firstepisode_unmedicated.mat';

    fc_chronic = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\fc_chronic.mat';
    fc_firstepisode_medicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\fc_firstepisode_medicated.mat';
    fc_firstepisode_unmedicated = 'D:\WorkStation_2018\SZ_classification\Data\Stat_results\fc_firstepisode_unmedicated.mat';
end

%% Prepare
load(cov_chronic);
load(cov_firstepisode_medicated);
load(cov_firstepisode_unmedicated);

load(fc_chronic); 
load(fc_firstepisode_medicated);
load(fc_firstepisode_unmedicated);


% NBS parameters
STATS.thresh = 3;
STATS.alpha = 0.025;  % Equal to two-tailed 0.05.
STATS.N = 246;
STATS.size = 'extent';
GLM.perms = 1000;
GLM.y = y;
GLM.test = 'ttest'; 

%% Duration
y = cat(1, fc_chronic, fc_firstepisode_medicated);
group = cat(1, ones(length(cov_chronic), 1), ones(length(cov_firstepisode_medicated), 1)+1);
group_design = zeros(length(group), 2);
n_g = size(group_design, 2);
for i =  1:n_g
    group_design(:,i) = group == i;
end
% Cov
cov_all = cat(1,  cov_chronic, cov_firstepisode_medicated);
cov_duration = cov_all(:,2:end-1);

design_mat = cat(2, group_design, cov_duration);

GLM.X = design_mat;
GLM.y = y;
GLM.perms = 1000;
GLM.contrast = [1 -1 0 0 0 0];
[test_stat_duration, pvalues_duration]=NBSglm(GLM);
STATS.test_stat = abs(test_stat_duration);  % two-tailed
[n_cnt_duration, cont_duration, pval_duration] = NBSstats(STATS);

%% Medication
y = cat(1, fc_firstepisode_medicated, fc_firstepisode_unmedicated);

group = cat(1, ones(length(cov_firstepisode_medicated), 1), ones(length(cov_firstepisode_unmedicated), 1)+1);
group_design = zeros(length(group), 2);
n_g = size(group_design, 2);
for i =  1:n_g
    group_design(:,i) = group == i;
end
% Cov
cov_all = cat(1,  cov_firstepisode_medicated, cov_firstepisode_unmedicated);
cov_medication = cov_all(:,1:end-1);

design_mat = cat(2, group_design, cov_medication);

GLM.X = design_mat;
GLM.y = y;
GLM.perms = 1000;
GLM.contrast = [1 -1 0 0 0 0 0];
[test_stat_medication, pvalues_medication]=NBSglm(GLM);
STATS.test_stat = abs(test_stat_medication);  % two-tailed
[n_cnt_medication, cont_medication, pval_medication] = NBSstats(STATS);

%% To 2D matrix
% cont_duration_full = full(cont_duration{1}) + full(cont_duration{1})';
cont_medication_full = full(cont_medication{1}) + full(cont_medication{1})';

tvalue_medication = zeros(246, 246);
tvalue_medication(tril(ones(246), -1) == 1) = test_stat_medication(1,:);
tvalue_medication = tvalue_medication + tvalue_medication';
tvalue_medication(~cont_medication_full)=0;

%% save
if is_save
    save('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalue_medication.mat', 'tvalue_medication');
end
fprintf('--------------------------All Done!--------------------------\n');
end