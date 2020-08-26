% Perform GLM + multiple correction for temporal properties of dynamic functional connectivity (mean dwell time, fractional windows and number of transitions).
% NOTE. Make sure the order of the dependent variables matches the order of the covariances
% ==============================================================================================

%% ================================Inputs===========================
subjects = 'F:\The_first_training\results\lcSelectedDataFolders.txt';
data_path = 'F:\The_first_training\results_dfc\lc_dfnc_cluster_stats.mat';
covariance = 'F:\The_first_training\cov\covariates.xlsx';
output_path = 'F:\The_first_training\results_dfc';
n_states = 4;
colnum_id = 1;
columns_group_label=2;
columns_covariates = [3,4,5];
contrast = [-1 1 0 0 0];
correction_threshold = 0.05;
correction_method = 'fdr';

%% ==============================Load=============================
% subject name
subjects_file = importdata(subjects);
n_sub = length(subjects_file);
subjects_name = cell(n_sub,1);
for i = 1:n_sub
    sn = strsplit(subjects_file{i}, filesep);
    subjects_name{i}= sn{end};
end

% Y
load(data_path);
mean_dwelltime = state_vector_stats.mean_dwell_time;
fractional_window = state_vector_stats.frac_time_state;
num_transitions = state_vector_stats.num_transitions;

% X
[cov, header, raw] = xlsread(covariance);

% design matrix
group_label = cov(:,columns_group_label);
uni_group_label = unique(group_label);
group_design = zeros(size(cov,1),numel(uni_group_label));
for i =  1:numel(uni_group_label)
    group_design(:,i) = ismember(group_label, uni_group_label(i));
end
design_matrix = cat(2, group_design, cov(:,columns_covariates));

% Sort design_matrix according with subject_name
id_in_cov = raw(2:end,colnum_id);
for i = 1:n_sub
    if  ~isa(id_in_cov{i}, 'char')
        id_in_cov{i} = num2str(id_in_cov{i});
    end
end
[Lia,Locb] = ismember(id_in_cov, subjects_name);
design_matrix_sorted = design_matrix(Locb,:);

% Make directory to save results
if ~exist(output_path,'dir')
    mkdir(output_path);
end

%% ==============================Stat=============================
GLM.perms = 0;
GLM.X = design_matrix_sorted;
GLM.y = [mean_dwelltime, fractional_window,num_transitions];
y_name = {'Mean dwell time', 'Fraction of time', 'Number of transitions'};
GLM.contrast = contrast;
GLM.test = 'ttest';
[test_stat,pvalues]=lc_NBSglm(GLM);

%% Multiple comparison correction
if strcmp(correction_method, 'fdr')
    results = multcomp_fdr_bh(pvalues, 'alpha', correction_threshold);
elseif strcmp(correction_method, 'fwe')
    results = multcomp_bonferroni(pvalues, 'alpha', correction_threshold);
else
    fprintf('Please indicate the correct correction method!\n');
end
h_corrected = results.corrected_h;

%% Save
dlmwrite(fullfile(output_path, 'results_metrics.txt'), [h_corrected; test_stat; pvalues], 'precision', '%5f', 'delimiter', '\t');

%% ==============================Plot=============================
loc_hc = group_design(:,1)==1;
loc_p = group_design(:,1)==0;
opt.colormap = 'gray';

% mean_dwelltime
mean_dwt_hc = mean(mean_dwelltime(loc_hc,:));
mean_dwt_p = mean(mean_dwelltime(loc_p,:));
std_dwt_hc = std(mean_dwelltime(loc_hc,:))./power(sum(loc_hc), 0.5);
std_dwt_p = std(mean_dwelltime(loc_p,:))./power(sum(loc_p), 0.5);
mean_dwt = [mean_dwt_hc',mean_dwt_p'];
std_dwt = [std_dwt_hc',std_dwt_p'];
opt.xticklabel = xticklabel;
opt.ylabel = 'Mean dwell time';
figure
set(gcf,'Position',[100 100 300 300]);
lc_plotbar(mean_dwt,std_dwt, opt);
le = legend({'HC', 'Patients'},'Location','NorthOutside','Orientation','horizon');
set(le, 'LineWidth',0.5)
saveas(gcf,fullfile(output_path, 'mean_dwell time.pdf'))

% Fraction of time spent
mean_fts_hc = mean(fractional_window(loc_hc,:));
mean_fts_p = mean(fractional_window(loc_p,:));
std_fts_hc = std(fractional_window(loc_hc,:))./power(sum(loc_hc), 0.5);
std_fts_p = std(fractional_window(loc_p,:))./power(sum(loc_p), 0.5);
mean_fts = [mean_fts_hc',mean_fts_p'];
std_fts = [std_fts_hc',std_fts_p'];
opt.ylabel = 'Fraction of time spent';
figure
set(gcf,'Position',[100 100 300 300]);
lc_plotbar(mean_dwt,std_dwt, opt);
le = legend({'HC', 'Patients'},'Location','NorthOutside','Orientation','horizon');
set(le, 'LineWidth',0.5)
saveas(gcf,fullfile(output_path, 'fraction_time.pdf'))

% Number of transitions
mean_nt_hc = mean(num_transitions(loc_hc,:));
mean_nt_p = mean(num_transitions(loc_p,:));
std_nt_hc = std(num_transitions(loc_hc,:))./power(sum(loc_hc), 0.5);
std_nt_p = std(num_transitions(loc_p,:))./power(sum(loc_p), 0.5);
mean_nt = [mean_nt_hc',mean_nt_p'];
std_nt = [std_nt_hc',std_nt_p'];
opt.xticklabel = {''};
opt.ylabel = 'Number of transitions';
figure
set(gcf,'Position',[100 100 150 300]);
lc_plotbar(mean_nt',std_nt', opt);
saveas(gcf,fullfile(output_path, 'number_transitions.pdf'))

