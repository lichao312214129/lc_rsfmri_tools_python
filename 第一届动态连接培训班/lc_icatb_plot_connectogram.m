%% ===========================Inputs================================
dfnc_workdir = 'F:\The_first_training\dfnc';
prefix = 'le';
only_display_significance = 0;

%% ===========================Load data=============================
% Statistical results
load(fullfile(dfnc_workdir,'results_dfnc.mat'));
[n_states, n_fnc] = size(test_stat);

% ICs
load(fullfile(dfnc_workdir,[prefix,'_dfnc.mat']));
ica_path = fileparts(dfncInfo.userInput.ica_param_file);
ic = fullfile(ica_path,[prefix, '_mean_component_ica_s_all_.nii']);

% Components
load(fullfile(dfnc_workdir, [prefix, '_dfnc.mat']));
comps = dfncInfo.userInput.comp;
comps_num = dfncInfo.comps;

%% ==================Get net_idx and network name===================
selected_comps = {comps.name}';
selected_comp_labels = {comps.value}';
n_selected_comps = length(selected_comps);
uni_selected_comps = unique(selected_comps);
n_uni_selected_comps = length(uni_selected_comps);
net_idx = cell(n_selected_comps,1);
for i = 1:n_uni_selected_comps
    net_idx{i} = ones(length(selected_comp_labels{i}),1)+i-1;
end
net_idx = cell2mat(net_idx);

% comp_labels = mat2cell(comps_num,[16,1]);
% for i = 1:n_selected_comps
%     comp_labels = cat(1,comp_labels, repmat(selected_comps(i), [length(selected_comp_labels{i}),1]));
% end

comp_network_names = cat(2,selected_comps,selected_comp_labels);

%% ======================Calc how many nodes========================
syms x
eqn = x*(x-1)/2 == n_fnc;
n_node = solve(eqn,x);
n_node = double(n_node);
n_node(n_node<0) = [];

%% ====================Vector to square mat=========================
mask = tril(ones(n_node,n_node),-1) == 1;
if only_display_significance
    test_stat(h_corrected==0) = 0;
end
    
for i = 1:n_states
    test_stat_mat = zeros(n_node,n_node);
    test_stat_mat(mask) = test_stat(i,:);
    test_stat_mat = test_stat_mat + test_stat_mat';

    %% ===========================Plot================================== 
    lc_icatb_plot_connectogram_base(comp_network_names, 'C', test_stat_mat, 'threshold', 0.7, 'image_file_names', ic, 'colorbar_label', 'Corr',  'line_width', 1, 'display_type', 'render','slice_plane', 'axial','radius_sm',1.8, 'conn_threshold',-Inf);

    %% ===========================Save==================================  
    set(gcf,'PaperType','a3');
    saveas(gcf,fullfile(dfnc_workdir, ['tvalues_circos_state',num2str(i),'.pdf']))
end