function lc_line(sepIndex, nNode, linewidth)
% nNode: node¸öÊý
n_net = length(sepIndex);
for i=1:n_net
    line([sepIndex(i),sepIndex(i)],[-10,nNode],'color','k','LineWidth',linewidth)
    line([0, nNode],[sepIndex(i),sepIndex(i)],'color','k','LineWidth',linewidth)
end
end