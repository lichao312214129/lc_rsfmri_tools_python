strings = {'i', 'love', 'matlab'};
nStr = length(strings);

for i = 1:nStr
    fprintf('%s\t', strings{i});
end

%%
strings = {'i', 'love', 'matlab'};
nStr = length(strings);
i = 1;
while i <= nStr
    fprintf('%s\t', strings{i});
    i = i + 1;
end