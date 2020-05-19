% This script is used to visualize the classification weight using network matrix and circle style.
%% -----------------------------------------------------------------
tvalues_medicated = importdata('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalue_medication.mat');
tvalues_duration = importdata('D:\WorkStation_2018\SZ_classification\Data\Stat_results\tvalue_duration.mat');
cmp = 'D:\My_Codes\lc_private_codes\workstation\SSD_classification\Visulization\cmp_tvalues_medication_effect.mat';
legends = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
xstr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
legend_fontsize = 7;
how_disp = 'all';
if_binary = 0;
which_group = 1;

% Load
load (cmp)
net_index = 'D:\My_Codes\lc_private_codes\workstation\SSD_classification\Visulization\netIndex.mat';
net_index = importdata(net_index);
% Tvalues

figure
set(gcf,'position',[150 150 1000 600])


if_add_mask = 0;
if_add_mask = 0;
cohen_tresh = 0;
h = subplot(121);
set(gca,'position',[0.1 0.5 0.3 0.3])  %注意此处0.1*宽高比
tv = tvalues_medicated;
tv(triu(ones(246, 246), 1)==1)=0;
lc_netplot('-n', tvalues_medicated, '-ni', net_index, '-il', 1, '-lg', legends, '-lgf', 6)
colormap(h, cmp_tvalues_medication_effect)
caxis([-4 4])
axis square

% Average
h=subplot(122);
set(gca,'position',[0.4 0.5 0.3 0.3])%注意此处0.1*宽高比
mean_tvalues = get_average_diff(tvalues_medicated, net_index);
mean_tvalues = mean_tvalues + mean_tvalues';
matrixplot(mean_tvalues, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(h, cmp_tvalues_medication_effect)
caxis([-8 8])
axis square

% Save
% saveas(gcf, 'D:\WorkStation_2018\SZ_classification\Figure\tvalue_medication_effect.pdf');


function meanFC = get_average_diff(diff, netidx)
%
uniid = unique(netidx);
n_uniid = numel(uniid);
unjid = uniid;
n_unjid = numel(unjid);
meanFC = zeros(numel(uniid));
for i = 1 : n_uniid
    id = find(netidx == uniid(i));
    for j =  1: n_unjid
        fc = diff(id, find(netidx == unjid(j)));
        % if within fc, extract upper triangle matrix
        if (all(diag(fc)) == 0) && (size(fc, 1) == size(fc, 2))
            fc = fc(triu(ones(length(fc)),1) == 1);
        end
        % Exclude zeros in fc
        % TODO: Consider the source data  have zeros.
        fc(fc == 0) = [];
        % Mean
        meanFC(i,j) = mean(fc(:));
    end
end
% Post-Process meanFC
meanFC(isnan(meanFC)) = 0;
meanFC(triu(ones(size(meanFC)), 1) == 1)=0;
end