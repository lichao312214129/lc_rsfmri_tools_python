%% Calculate dfnc variance

%% ================================Inputs===========================
subjects_name = 'F:\The_first_training\results\lcSelectedDataFolders.txt';
dfnc_results_path = 'F:\The_first_training\results_dfnc_script';
prefix = 'lc';
covariance = 'F:\The_first_training\cov\covariates.xlsx';
output_path = 'F:\The_first_training\results_dfc';
colnum_id = 1;
columns_group_label = 2;
columns_covariates = [3,4,5];
contrast = [-1 1 0 0 0];
correction_threshold = 0.05;
correction_method = 'fdr';
xticklabel = {'State 1', 'State 2','State 3', 'State 4'};

%% ==============================Load=============================
% subject name
subjects_file = importdata(subjects_name);
n_subj = length(subjects_file);
subjects_name = cell(n_subj,1);
for i = 1:n_subj
    sn = strsplit(subjects_file{i}, filesep);
    subjects_name{i}= sn{end};
end

% Y
dfnc_struct = dir(fullfile(dfnc_results_path,[prefix,'_dfnc_sub','*','results.mat']));
dfnc_name = {dfnc_struct.name}';
dfnc_path = fullfile(dfnc_results_path, dfnc_name);
n_subj = length(dfnc_name);
for i = 1:n_subj
    fncdyn = importdata(dfnc_path{i});
    fncdyn = fncdyn.FNCdyn;
    if i == 1
        var_dfnc = zeros(n_subj, size(fncdyn,2));
    end
    var_dfnc(i,:) = var(fncdyn);
end

[~, n_fnc] = size(var_dfnc);

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
for i = 1:n_subj
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

%% ===============================Stat==============================
GLM.perms = 0;
GLM.X = design_matrix_sorted;
GLM.y = var_dfnc;
y_name = {'var_dfnc'};
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

%% ==============================Save=============================
save(fullfile(output_path, 'results_var_dfnc.mat'), 'h_corrected', 'test_stat', 'pvalues');

%% ==============================Pre plot=============================
% Components name
load(fullfile(dfnc_results_path,[prefix, '_dfnc.mat']))
comps = dfncInfo.userInput.comp;

% Number of nodes
syms x
eqn = x*(x-1)/2 == n_fnc;
n_node = solve(eqn,x);
n_node = double(n_node);
n_node(n_node<0) = [];

% vector to squre net
mask = tril(ones(n_node,n_node),-1) == 1;
loc_hc = group_design(:,1)==1;
loc_p = group_design(:,1)==0;

var_dfnc_mean_hc_square = zeros(n_node,n_node);
var_dfnc_mean_hc_square(mask) = mean(var_dfnc(loc_hc,:));
var_dfnc_mean_hc_square = var_dfnc_mean_hc_square + var_dfnc_mean_hc_square';

var_dfnc_mean_p_square = zeros(n_node,n_node);
var_dfnc_mean_p_square(mask) = mean(var_dfnc(loc_p,:));
var_dfnc_mean_p_square = var_dfnc_mean_p_square + var_dfnc_mean_p_square';

tvalues = zeros(n_node,n_node);
tvalues(mask) = test_stat;
tvalues = tvalues + tvalues';
pv = ones(n_node,n_node);
pv(mask) = pvalues;
pv = pv + pv';
log_p_sign_t = log10(pv)*sign(tvalues);
log_p_sign_t = log_p_sign_t-diag(diag(log_p_sign_t));

%% ==============================Plot=============================
max_caix = max([max(var_dfnc_mean_p_square(:)), max(var_dfnc_mean_hc_square(:))]);
map_var = brewermap(50,'Reds');
map_pt = brewermap(50,'*RdBu');
legends = {'Visual', 'SomMot', 'DorsAttn'};
netIndex = [1,1,2,3,2,3,2,3];
figure('Position',[100 100 800 400]);

% HC
subplot(1,3,1)
lc_netplot('-n', var_dfnc_mean_hc_square, '-ni',  netIndex,'-il',1, '-lg', legends);
axis square
h = gca;
colormap(h, map_var);
caxis([0,max_caix]);
title('HC');

% Patient
subplot(1,3,2)
lc_netplot('-n', var_dfnc_mean_p_square, '-ni',  netIndex,'-il',1, '-lg', legends);
axis square
h = gca;
colormap(h, map_var);
cb = colorbar('horiz','position',[0.3 0.1 0.15 0.02]);
caxis([0,max_caix]);
ylabel(cb,'Variance of DFNC', 'FontSize', 10);
title('Patient');

% Patient - HC
min_caix_pt = min(log_p_sign_t(:));
max_caix_pt = max(log_p_sign_t(:));

subplot(1,3,3)
lc_netplot('-n', log_p_sign_t, '-ni',  netIndex,'-il',1, '-lg', legends);
axis square
colormap(map_pt);
cb = colorbar('horiz','position',[0.73 0.1 0.15 0.02]);
caxis([min_caix_pt, max_caix_pt]);
ylabel(cb,'-log10(p) * sign(t)', 'FontSize', 10);
title('Patient -  HC');


%% ================================Save=============================
saveas(gcf,fullfile(output_path, ['mean_dfnc_in_state', num2str(i), '.pdf']));




