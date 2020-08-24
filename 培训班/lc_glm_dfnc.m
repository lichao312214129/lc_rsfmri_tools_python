% Perform GLM + multiple correction for dynamic functional connectivity.
% NOTE. Make sure the order of the dependent variables matches the order of the covariances
% ==============================================================================================

%% ================================Inputs===========================
subjects_name = 'F:\The_first_training\results\lcSelectedDataFolders.txt';
dfnc_results_path = 'F:\The_first_training\results_dfnc_script';
prefix = 'lc';
covariance = 'F:\The_first_training\cov\covariates.xlsx';
output_path = 'F:\The_first_training\results_dfnc_script';
colnum_id = 1;   % which colnum is the subject ID
columns_group_label=2;  % which colnum is the group label
columns_covariates = [2,3];  % which colnums are the covariates
contrast = [-1 1 0 0];
correction_method = 'fdr';
correction_threshold = 0.05;
xticklabel = {'State 1', 'State 2','State 3', 'State 4'};

%% ==============================Load=============================
% subject name
subjects_file = importdata(subjects_name);
n_sub = length(subjects_file);
subjects_name = cell(n_sub,1);
for i = 1:n_sub
    sn = strsplit(subjects_file{i}, filesep);
    subjects_name{i}= sn{end};
end

% Components
load(fullfile(dfnc_results_path,[prefix, '_dfnc.mat']));
comps = dfncInfo.userInput.comp;

% Y
load(fullfile(dfnc_results_path,[prefix, '_dfnc_cluster_stats.mat']));
dfnc = squeeze(dfnc_corrs);
[n_sub, n_fnc, n_states] = size(dfnc);

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

%% ===============================Stat==============================
GLM.perms = 0;
X = design_matrix_sorted;
y_name = {'dfnc'};
GLM.contrast = contrast;
GLM.test = 'ttest';

% ------Each State Loop--------
test_stat = zeros(n_states, n_fnc);
pvalues = ones(n_states, n_fnc);
h_corrected = zeros(n_states, n_fnc);
for i  = 1:n_states
    GLM.y = squeeze(dfnc(:,:,i));
    GLM.X = X;
    % De-NaN
    GLM.X(isnan(GLM.y(:,1)),:) = [];
    GLM.y(isnan(GLM.y(:,1)),:) = [];
    
%     [h,p] = ttest2(GLM.y(GLM.X(:,1)==1,:),GLM.y(GLM.X(:,1)==0,:));
    
    [test_stat(i,:),pvalues(i,:)]=lc_NBSglm(GLM);
    
    %% Multiple comparison correction
    if strcmp(correction_method, 'fdr')
        results = multcomp_fdr_bh(pvalues(i,:), 'alpha', correction_threshold);
    elseif strcmp(correction_method, 'fwe')
        results = multcomp_bonferroni(pvalues(i,:), 'alpha', correction_threshold);
    else
        fprintf('Please indicate the correct correction method!\n');
    end
    h_corrected(i,:) = results.corrected_h;
end
% ------Each State Loop End--------

%% ==============================Save=============================
save(fullfile(output_path, 'results_dfnc.mat'), 'h_corrected', 'test_stat', 'pvalues');

%% ==============================Plot=============================
% How many nodes
n_node = [(1 + power(1-8*1*(-n_fnc), 0.5))/2, (1 - power(1-8*1*(-n_fnc), 0.5))/2];
n_node = n_node(sign(n_node)==1);

% vector to square mat
mask = tril(ones(n_node,n_node),-1) == 1;
loc_hc = group_design(:,1)==1;
loc_p = group_design(:,1)==0;

legends = {comps.name};
net_num_cell = {comps.value};
n_net = length(net_num_cell);
net_num = cell2mat({comps.value});
netIndex = zeros(n_node,1)';
for i = 1:n_net
    netIndex(ismember(net_num, net_num_cell{i})) = i;
end

for i = 1: n_states
    dfnc_hc_state = squeeze(dfnc(loc_hc,:,i));
    dfnc_hc_state(isnan(dfnc_hc_state(:,1)),:) = [];
    dfnc_p_state = squeeze(dfnc(loc_p,:,i));
    dfnc_p_state(isnan(dfnc_p_state(:,1)),:) = [];
    dfnc_mean_hc_state = mean(dfnc_hc_state);
    dfnc_mean_p_state = mean(dfnc_p_state);
    
    dfnc_mean_hc_state_square = zeros(n_node,n_node);
    dfnc_mean_hc_state_square(mask) = dfnc_mean_hc_state;
    dfnc_mean_hc_state_square = dfnc_mean_hc_state_square + dfnc_mean_hc_state_square';

    dfnc_mean_p_state_square = zeros(n_node,n_node);
    dfnc_mean_p_state_square(mask) = dfnc_mean_p_state;
    dfnc_mean_p_state_square = dfnc_mean_p_state_square + dfnc_mean_p_state_square';

    tvalues = zeros(n_node,n_node);
    tvalues(mask) = test_stat(i,:);
    tvalues = tvalues + tvalues';
    
    % ----Plot square net-----
    [map,num,typ] = brewermap(50,'*RdBu');
    figure('Position',[100 100 800 400]);
    
    % HC
    subplot(1,3,1)
    lc_netplot('-n', dfnc_mean_hc_state_square, '-ni',  netIndex,'-il',1, '-lg', legends);
    axis square
    colormap(map);
    caxis([-1,1]);
    title('HC');
    
    % Patient
    subplot(1,3,2)
    lc_netplot('-n', dfnc_mean_p_state_square, '-ni',  netIndex,'-il',1, '-lg', legends);
    axis square
    colormap(map);
    cb = colorbar('horiz','position',[0.3 0.1 0.15 0.02]);
    caxis([-1,1]);
    ylabel(cb,'Functional connectivity (Z)', 'FontSize', 10);
    title('Patient');
     
    
    % Patient - HC
    subplot(1,3,3)
    lc_netplot('-n', tvalues, '-ni',  netIndex,'-il',1, '-lg', legends);
    axis square
    colormap(map);
    cb = colorbar('horiz','position',[0.73 0.1 0.15 0.02]);
    % caxis([-max(max(abs(log_p_sign_t))),max(max(abs(log_p_sign_t)))]);
    ylabel(cb,'T-values', 'FontSize', 10);
    title('Patient -  HC');
    
    % Save
    saveas(gcf,fullfile(output_path, ['mean_dfnc_in_state', num2str(i), '.pdf']));
end


