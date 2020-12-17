a  % 因为a没有赋值，所以会报错

% 用try catch来达到容错的目的，同时也获取错误信息
try
    a;
catch ME
    fprintf('报错的信息是：%s\n', ME.message)
    a = 1;
end

fprintf('a被赋值，其值为：%d\n', a)