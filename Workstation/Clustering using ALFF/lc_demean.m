function [X_demean] = lc_demean(X,dim)
% 用于对数据进行除以均值处理，目的是统一数据的量纲
% input:
%   X:矩阵或者向量。n_subj*n_feature
%   dim=1,对每个特征除均值（组水平）;dim=2,对每个人所有特征除均值（个体水平）
%%
if nargin<2
    dim=2;% 默认个体水平
end

%
[n_subj,n_feature]=size(X);
X_demean = bsxfun(@rdivide, X, mean(X,dim));
end

