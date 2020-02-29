function plot_average_diff()
%% Plot average tvalues within or between networks

% Inputs
load D:\WorkStation_2018\SZ_classification\Figure\differecens_all.mat
load D:\WorkStation_2018\SZ_classification\Figure\differecens_feu.mat
load  D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\netIndex.mat;
mycolormap = 'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\cmp_average_diff.mat';
net = 'D:\WorkStation_2018\SZ_classification\Figure\weights.mat';
name = 'avargT_sz';
load (net);

% Weights filter
perc_filter = 0;

[sort_weight_pooling, id] = sort(abs(weight_pooling(:)));
weight_pooling(id(1:floor(length(id) * perc_filter))) = 0;
weight_pooling = abs(weight_pooling);  % ABS

[sort_weight_unmedicated, id] = sort(abs(weight_unmedicated(:)));
weight_unmedicated(id(1:floor(length(id) * perc_filter))) = 0;
weight_unmedicated = abs(weight_unmedicated);  % ABS

% Get average
average_weight_pooling = get_average_diff(weight_pooling, 0, [], netidx);
average_weight_unmedicated = get_average_diff(weight_unmedicated, 0, [], netidx);

average_diff_pooling = get_average_diff(differences_all, 0, [], netidx);
average_diff_unmedicated = get_average_diff(differences_feu, 0, [], netidx);

% Plot weight
load D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\mycmap_average_weight
figure;
subplot(1, 2, 1)
xstr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
matrixplot(average_weight_pooling, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmap_average_weight);
caxis([0.5,5])
colorbar

subplot(1, 2, 2)
matrixplot(average_weight_unmedicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmap_average_weight);
caxis([0.5,5])
colorbar
saveas(gcf,  'D:\WorkStation_2018\SZ_classification\Figure\average_weight_med_and_unmedicated_099.pdf')

% Plot differencs
load D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\Visulization\mycmap_average_diff
figure;
subplot(1, 2, 1)
mycmp = importdata(mycolormap);
xstr = {'Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'};
matrixplot(average_diff_pooling, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmap_average_diff);
caxis([-5,5])
colorbar

subplot(1, 2, 2)
matrixplot(average_diff_unmedicated, xstr, xstr, 'FigShap','d','FigStyle','Tril');
colormap(mycmap_average_diff);
caxis([-5,5])
colorbar
saveas(gcf,  'D:\WorkStation_2018\SZ_classification\Figure\average_diff_med_and_unmedicated_099.pdf')
end

function meanFC = get_average_diff(diff, is_mask, mask, netidx)
% Differences filter
if is_mask
    diff(mask == 0) = 0;
end
diff_full = diff + diff';
%
uniid = unique(netidx);
n_uniid = numel(uniid);
unjid = uniid;
n_unjid = numel(unjid);
meanFC = zeros(numel(uniid));
for i = 1 : n_uniid
    id = find(netidx == uniid(i));
    for j =  1: n_unjid
        fc = diff_full(id, find(netidx == unjid(j)));
        % if within fc, extract upper triangle matrix
        if (all(diag(fc)) == 0) && (size(fc, 1) == size(fc, 2))
            fc = fc(triu(ones(length(fc)),1) == 1);
        end
        % Exclude zeros in fc
        % TODO: Consider the source data  have zeros.
%         fc(fc == 0) = [];
        % Mean
        meanFC(i,j) = mean(fc(:));
    end
end
% Post-Process meanFC
meanFC(isnan(meanFC)) = 0;
meanFC(triu(ones(size(meanFC)), 1) == 1)=0;
end