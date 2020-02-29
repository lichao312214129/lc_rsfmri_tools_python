function lc_bar_region_of_each_network(location_of_sep, n_node, extend, is_legend, legends)
% TO plot bar with sevral regions, each region with a unique color
% representting a network.
n_net = length(location_of_sep);
randseed(1);
color = jet(n_net) / 1.1;
barwidth = abs((n_node + extend / 2) - (n_node+extend));
extend_of_legends = extend + 4 ;
h = zeros(n_net - 1, 1);
for i = 1 : n_net-1
    h(i) = fill([location_of_sep(i), location_of_sep(i+1), location_of_sep(i+1), location_of_sep(i)], [n_node + extend / 2, n_node + extend / 2, n_node+extend n_node + extend], color(i,:));
    fill([ n_node + barwidth, n_node + barwidth, n_node + extend, n_node + extend], [location_of_sep(i), location_of_sep(i+1), location_of_sep(i+1), location_of_sep(i)], color(i,:))
    if is_legend
        % Y axix
        text(n_node + extend_of_legends, (location_of_sep(i+1) - location_of_sep(i)) / 2 +  location_of_sep(i),...
            legends{i}, 'fontsize', 5, 'rotation', 0);
         % X axix
        text((location_of_sep(i+1) - location_of_sep(i)) / 2 +  location_of_sep(i), n_node + extend_of_legends,...
            legends{i}, 'fontsize', 5, 'rotation', -90);
    end
end
end