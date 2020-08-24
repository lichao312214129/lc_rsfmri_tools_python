% Display dfnc states
% ==============================================================================================

%% ================================Inputs===========================
dfnc_results_path = 'F:\The_first_training\results_dfnc_script';
prefix = 'lc';
output_path = 'F:\The_first_training\results_dfnc_script';

%% ==============================Load=============================
load(fullfile(dfnc_results_path,[prefix, '_dfnc_post_process']));
load(fullfile(dfnc_results_path,[prefix, '_dfnc.mat']));

states = clusterInfo.Call;
comps = dfncInfo.userInput.comp;

[n_state,n_fnc] = size(states);

%% ==============================Plot=============================
% How many nodes
n_node = [(1 + power(1-8*1*(-n_fnc), 0.5))/2, (1 - power(1-8*1*(-n_fnc), 0.5))/2];
n_node = n_node(sign(n_node)==1);

% network label and legends
legends = {comps.name};
net_num_cell = {comps.value};
n_net = length(net_num_cell);
net_num = cell2mat({comps.value});
netIndex = zeros(n_node,1)';
for i = 1:n_net
    netIndex(ismember(net_num, net_num_cell{i})) = i;
end

% vector to square mat
mask = tril(ones(n_node,n_node),-1) == 1;
state_square = cell(n_state,1);
for i = 1:n_state
    state_square_ = zeros(n_node,n_node);
    state_square_(mask) = states(i,:);
    state_square_ = state_square_ + state_square_';
    state_square{i} = state_square_;
end

% Plot
% ----Plot square net-----
[map,num,typ] = brewermap(50,'*RdBu');
figure('Position',[100 100 1200 400]);
for i = 1:n_state
    subplot(1,n_state,i)
    lc_netplot('-n', state_square{i}, '-ni',  netIndex,'-il',1, '-lg', legends);
    axis square
    colormap(map);
    caxis([-1,1]);
    freq = sprintf('%.0f',sum(clusterInfo.IDXall==i));
    freq_perc = sprintf('%.0f',100*sum(clusterInfo.IDXall==i)/length(clusterInfo.IDXall));
    title(['State', num2str(i), ': ',num2str(freq), ' (', num2str(freq_perc),'%)'],'FontWeight','bold');
end
cb = colorbar('horiz','position',[0.45 0.1 0.15 0.02]);
ylabel(cb,'Functional connectivity (Z)', 'FontSize', 10);
% Save
set(gcf,'PaperType','a3');
saveas(gcf,fullfile(output_path, ['States.pdf']));