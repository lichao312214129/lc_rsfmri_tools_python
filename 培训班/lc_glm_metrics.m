%% ================================Inputs===========================
dfnc_workdir = 'F:\The_first_training\dfnc';
prefix = 'le';
xticklabel = {'State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6'};

covariance = 'F:\The_first_training\cov\covariates.xlsx';
colnum_id = 1;   % which colnum is the subject ID
columns_group_label=2;  % which colnum is the group label
columns_covariates = [3,4];  % which colnums are the covariates
contrast = [-1 1 0 0];
correction_method = 'fdr';
correction_threshold = 0.05;
only_display_sig = 0;

%% ==============================Load=============================
% subject name
load(fullfile(dfnc_workdir,[prefix,'_dfnc.mat']));
ica_path = fileparts(dfncInfo.userInput.ica_param_file);
subjects_file = importdata(fullfile(ica_path,[prefix, 'SelectedDataFolders.txt']));
n_sub = length(subjects_file);
subjects_name = cell(n_sub,1);
for i = 1:n_sub
    sn = strsplit(subjects_file{i}, filesep);
    subjects_name{i}= sn{end};
end

% Components
load(fullfile(dfnc_workdir,[prefix, '_dfnc.mat']));
comps = dfncInfo.userInput.comp;
% xticklabel = {comps.name};

% Y
load(fullfile(dfnc_workdir,[prefix, '_dfnc_cluster_stats.mat']));
mean_dwelltime = state_vector_stats.mean_dwell_time;
fractional_window = state_vector_stats.frac_time_state;
num_transitions = state_vector_stats.num_transitions;

% X
%%% ===  TODO
[~, ~, suffix] = fileparts(covariance);
if strcmp(suffix,  '.txt')
    cov = importdata(covariance);
    cov.data;
    cov.textdata;
elseif suffix == '.xlsx'
    [~, header, cov] = xlsread(covariance);
end
cov = cov(2:end,:);

% design matrix
group_label = cov(:,columns_group_label);
group_label = cell2mat(group_label);
uni_group_label = unique(group_label);
group_design = zeros(size(cov,1),numel(uni_group_label));
for i =  1:numel(uni_group_label)
    group_design(:,i) = ismember(group_label, uni_group_label(i));
end
design_matrix = cat(2, group_design, cell2mat(cov(:,columns_covariates)));

% Sort design_matrix according with subject_name
id_in_cov = cov(:,colnum_id);
for i = 1:n_sub
    if  ~isa(id_in_cov{i}, 'char')
        id_in_cov{i} = num2str(id_in_cov{i});
    end
end
[Lia,Locb] = ismember(id_in_cov, subjects_name);
design_matrix_sorted = design_matrix(Locb,:);

% Make directory to save results
if ~exist(dfnc_workdir,'dir')
    mkdir(dfnc_workdir);
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
dlmwrite(fullfile(dfnc_workdir, 'results_metrics.txt'), [h_corrected; test_stat; pvalues], 'precision', '%5f', 'delimiter', '\t');

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

saveas(gcf,fullfile(dfnc_workdir, 'mean_dwell time.pdf'))

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
saveas(gcf,fullfile(dfnc_workdir, 'fraction_time.pdf'))

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
saveas(gcf,fullfile(dfnc_workdir, 'number_transitions.pdf'))

