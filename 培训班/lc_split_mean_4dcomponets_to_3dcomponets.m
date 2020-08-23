% This script is used to split group 4d components into multiple group 3d components.
% =================================================================

% Inputs
input_file = 'F:\The_first_training\results\lc_mean_component_ica_s_all_.nii';
out_dir = 'F:\The_first_training\results\group_3d_componets';

% Mkdir
if ~ exist(out_dir, 'dir')
    mkdir(out_dir)
end

% Get all 4d files
[fourD_componets, header] = y_Read(input_file);

% Transform 4d to 3d subject by subject
n_compnents = size(fourD_componets,4);
for i = 1:n_compnents
    fprintf('%d/%d\n',i, n_compnents);
    y_Write(squeeze(fourD_componets(:,:,:,i)), header, fullfile(out_dir, ['component',num2str(i), '.nii']));
end

disp('Done!\n');

