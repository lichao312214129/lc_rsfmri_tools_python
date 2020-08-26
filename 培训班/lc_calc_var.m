%% Calculate dfnc variance

%% ================================Inputs===========================
dfnc_workdir = 'F:\The_first_training\results_dfnc_script';
prefix = 'lc';

covariance = 'F:\The_first_training\cov\covariates.xlsx';
output_path = 'F:\The_first_training\results_dfnc_script';
colnum_id = 1;
columns_group_label = 2;
columns_covariates = [3,4];
contrast = [-1 1 0 0];
correction_threshold = 0.05;
correction_method = 'fdr';
only_display_sig = 0;

%% ==============================Load=============================
% subject name
% subject name
load(fullfile(dfnc_workdir,[prefix,'_dfnc.mat']));
ica_path = fileparts(dfncInfo.userInput.ica_param_file);
subjects_file = importdata(fullfile(ica_path,[prefix, 'SelectedDataFolders.txt']));

n_subj = length(subjects_file);
subjects_name = cell(n_subj,1);
for i = 1:n_subj
    sn = strsplit(subjects_file{i}, filesep);
    subjects_name{i}= sn{end};
end

% Y
dfnc_struct = dir(fullfile(dfnc_workdir,[prefix,'_dfnc_sub','*','results.mat']));
dfnc_name = {dfnc_struct.name}';
dfnc_path = fullfile(dfnc_workdir, dfnc_name);
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
[~, header, cov] = xlsread(covariance);
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
if only_display_sig
    test_stat(~h_corrected) = 0;
end

% Number of nodes
n_node = [(1 + power(1-8*1*(-n_fnc), 0.5))/2, (1 - power(1-8*1*(-n_fnc), 0.5))/2];
n_node = n_node(sign(n_node)==1);

% Components name
load(fullfile(dfnc_workdir,[prefix, '_dfnc.mat']))
comps = dfncInfo.userInput.comp;
legends = {comps.name};
net_num_cell = {comps.value};
n_net = length(net_num_cell);
net_num = cell2mat({comps.value});
netIndex = zeros(n_node,1)';
for i = 1:n_net
    netIndex(ismember(net_num, net_num_cell{i})) = i;
end

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

%% ==============================Plot=============================
max_caix = max([max(var_dfnc_mean_p_square(:)), max(var_dfnc_mean_hc_square(:))]);
map_var = brewermap(50,'Reds');
map_pt = brewermap(50,'*RdBu');
figure('Position',[100 100 800 400]);

% HC
subplot(1,3,1)
lc_netplot('-n', var_dfnc_mean_hc_square, '-ni',  netIndex,'-il',1, '-lg', legends,'-lgf',8);
axis square
h = gca;
colormap(h, map_var);
caxis([0,max_caix]);
title('HC');

% Patient
subplot(1,3,2)
lc_netplot('-n', var_dfnc_mean_p_square, '-ni',  netIndex,'-il',1, '-lg', legends,'-lgf',8);
axis square
h = gca;
colormap(h, map_var);
cb = colorbar('horiz','position',[0.3 0.1 0.15 0.02]);
caxis([0,max_caix]);
ylabel(cb,'Variance of dFNC', 'FontSize', 10);
title('Patient');

% Patient - HC
min_caix_pt = min(tvalues(:));
max_caix_pt = max(tvalues(:));

subplot(1,3,3)
lc_netplot('-n', tvalues, '-ni',  netIndex,'-il',1, '-lg', legends,'-lgf',8);
axis square
colormap(map_pt);
cb = colorbar('horiz','position',[0.73 0.1 0.15 0.02]);
caxis([min_caix_pt, max_caix_pt]);
ylabel(cb,'T-values', 'FontSize', 10);
title('Patient -  HC');

%% ================================Save=============================
saveas(gcf,fullfile(output_path, 'var_dfnc.pdf'));




