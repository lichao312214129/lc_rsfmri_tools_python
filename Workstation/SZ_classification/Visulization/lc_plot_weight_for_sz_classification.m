% This script is used to visualize the classification weight using network matrix and circle style.
%% -----------------------------------------------------------------
load D:\WorkStation_2018\SZ_classification\Figure\weights.mat;
load D:\WorkStation_2018\SZ_classification\Figure\differecens_all.mat
load D:\WorkStation_2018\SZ_classification\Figure\differecens_feu.mat
load D:\WorkStation_2018\SZ_classification\Figure\mycmap.mat
legends = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};

%  Filter weights
perc_filter = 0.0;

[sort_weight_pooling, id] = sort(abs(weight_pooling(:)));
weight_pooling(id(1:floor(length(id) * perc_filter))) = 0;
weight_pooling = abs(weight_pooling);  % ABS

[sort_weight_unmedicated, id] = sort(abs(weight_unmedicated(:)));
weight_unmedicated(id(1:floor(length(id) * perc_filter))) = 0;
weight_unmedicated = abs(weight_unmedicated);  % ABS
% Plot
if_add_mask = 0;
how_disp = 'all';
if_binary = 0;
which_group = 1;

net_index_path = 'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\ML\netIndex.mat';
load D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\ML\colormap_weight.mat;

% Weights
load D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\mycmap_weight_all 

figure;
ax = tight_subplot(1, 2, [0.05 0.1], [0.05 0.05],[0.05 0.05]);
axes(ax(1))
lc_netplot(weight_pooling, if_add_mask, weight_pooling ~= 0, how_disp, 0, which_group, net_index_path, 1, legends);
colormap(mycmap_weight_all );
caxis([0 5]);
axis square
title('All datasets', 'fontsize', 10, 'fontweight','bold');
% colorbar('Location','westoutside');

axes(ax(2))
lc_netplot(weight_unmedicated, if_add_mask, weight_unmedicated ~= 0, how_disp, 0, which_group, net_index_path, 1, legends);
colormap(mycmap_weight_all );
caxis([0 5]);
axis square
title('First episode unmedicated subgroup', 'fontsize', 10, 'fontweight','bold');
% colorbar('Location','westoutside');
saveas(gcf,  'D:\WorkStation_2018\SZ_classification\Figure\weight_all.pdf');

% Differences
figure;
ax = tight_subplot(1, 2, [0.05 0.1], [0.05 0.05],[0.05 0.05]);
axes(ax(1))
lc_netplot(differences_all + differences_all', if_add_mask, weight_pooling ~= 0, how_disp, 0, which_group, net_index_path, 1, legends);
colormap(mycmap)
caxis([-1 1]);
axis square
title('All datasets', 'fontsize', 10, 'fontweight','bold');
% colorbar('Location','westoutside');

axes(ax(2))
lc_netplot(differences_feu + differences_feu', if_add_mask, weight_unmedicated ~= 0, how_disp, 0, which_group, net_index_path, 1, legends);
colormap(mycmap)
caxis([-1 1]);
axis square
title('First episode unmedicated subgroup', 'fontsize', 10, 'fontweight','bold');
% colorbar('Location','westoutside');
saveas(gcf,  'D:\WorkStation_2018\SZ_classification\Figure\diff_all.pdf');