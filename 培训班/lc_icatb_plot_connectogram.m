%% ===========================Inputs================================
dfnc_param = 'F:\The_first_training\results_dfnc_script\lc_dfnc.mat';
ic = 'F:\The_first_training\results\lc_mean_component_ica_s_all_.nii';
ic_name = 'F:\The_first_training\results\lc_gica_results\ic_name.xlsx';
results_file = 'F:\The_first_training\results_dfnc_script\results_dfnc.mat';
output_path = 'F:\The_first_training\results_dfnc_script';

%% ===========================Load data=============================
load(results_file);
[n_states, n_fnc] = size(test_stat);

load(dfnc_param);
comps_num = dfncInfo.comps;

[~, ~, ic_name_raw] = xlsread(ic_name);

%% ==================Get net_idx and network name===================
selected_comps = ic_name_raw(comps_num);
n_selected_comps = length(selected_comps);
uni_selected_comps = unique(selected_comps);
n_uni_selected_comps = length(uni_selected_comps);
net_idx = zeros(n_selected_comps,1);
for i = 1:n_uni_selected_comps
    net_idx(ismember(selected_comps, uni_selected_comps(i))) = i;
end

comp_network_names = cell(n_uni_selected_comps,2);
comp_network_names(:,1) = uni_selected_comps;
for i = 1:n_uni_selected_comps
    comp_network_names{i,2} =  reshape(comps_num(ismember(selected_comps, uni_selected_comps(i))),1,[]);
end

comp_labels = cell(n_selected_comps,1);
for i = 1:n_selected_comps
    comp_labels{i} =num2str(comps_num(i));
end

%% ======================Calc how many nodes========================
syms x
eqn = x*(x-1)/2 == n_fnc;
n_node = solve(eqn,x);
n_node = double(n_node);
n_node(n_node<0) = [];

%% ====================Vector to square mat=========================
i = 1;
mask = tril(ones(n_node,n_node),-1) == 1;
test_stat_mat = zeros(n_node,n_node);
test_stat_mat(mask) = test_stat(i,:);
test_stat_mat = test_stat_mat + test_stat_mat';

%% ===========================Plot================================== 
C = icatb_vec2mat(clusterInfo.Call(1,:));
lc_icatb_plot_connectogram_base(comp_network_names, 'C', C, 'threshold', 0.7,'comp_labels',comp_labels, 'image_file_names', ic, 'colorbar_label', 'Corr',  'line_width', 1, 'display_type', 'render','slice_plane', 'axial','radius_sm',1.6);
figure
imagesc(C);
colormap(jet)
caxis([-1,1])

%% ===========================Save==================================  
saveas(gcf,fullfile(output_path, 'tvalues_circos.pdf'))