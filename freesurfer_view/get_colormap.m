% N is calculated by prep_colormap.py
% For example, N=144
% [map,num,typ] = brewermap(144,'*RdBu');

% Step 1: get colormap
[map,num,typ] = brewermap(144,'*RdBu');

% Step 2: save map to excel
xlswrite('./map.xlsx', map);