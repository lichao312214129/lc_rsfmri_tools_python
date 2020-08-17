% This script is used to extract components. Transform 4d to 3d.
% =================================================================

% Inputs
input_dir = 'F:\The_first_training\results\subject_4d_componets';
out_dir = 'F:\The_first_training\results\subject_3d_componets';

% Get all 4d files
file =  dir(input_dir);
file_names = {file.name}';
file_names = file_names(3:end);
file_path = fullfile(input_dir, file_names);

% Transform 4d to 3d subject by subject
n_sub = length(file_path);
for i = 1:n_sub
    fprintf('%d/%d\n',i, n_sub);
    [data, header] = y_Read(file_path{i});
    if i == 1
        n_componets = size(data,4);
    end
    save_name = strsplit(file_names{i}, '.');
    suffix = save_name{end};
    save_name = save_name{1};
    for j = 1:n_componets
        out_dir_components = fullfile(out_dir, ['component', num2str(j)]);
        if ~exist(out_dir_components, 'dir')
            mkdir(out_dir_components)
        end
        y_Write(squeeze(data(:,:,:,j)), header, fullfile(out_dir_components, [save_name, '.',suffix]));
    end
end

disp('Done!\n');

