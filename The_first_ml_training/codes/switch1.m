a = input('请输入一个数：', 's');
a = eval(a);
switch a  % 分支条件
    case 1  % 分支1
        disp('a == 1');
    case 0 % 分支2
        disp('a == 0');
    otherwise % 分支3
        disp('a 不是1也不是0');
end

