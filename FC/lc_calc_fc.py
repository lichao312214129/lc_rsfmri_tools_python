""" 
This script is used to extract signals on spheres from an atlas (Power264) and plot a connectome
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
from lc_read_write_Mat import read_mat
import nilearn
from nilearn import datasets
adhd = datasets.fetch_adhd(n_subjects=1)
fmri_filename = adhd.func[0]
confounds_filename = adhd.confounds[0]
print('Functional image is {0},\nconfounds are {1}.'.format(fmri_filename,
      confounds_filename))

power = datasets.fetch_coords_power_2011()
print('Power atlas comes with {0}.'.format(power.keys()))


# Compute within spheres averaged time-series
import numpy as np
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
print('Stacked power coordinates in array of shape {0}.'.format(coords.shape))


from nilearn import input_data

spheres_masker = input_data.NiftiSpheresMasker(
    seeds=coords, smoothing_fwhm=4, radius=5.,
    detrend=True, standardize=True, low_pass=0.1, high_pass=0.01, t_r=2.5)

timeseries = spheres_masker.fit_transform(fmri_filename,
                                          confounds=confounds_filename)

print('time series has {0} samples'.format(timeseries.shape[0]))

from sklearn.covariance import GraphLassoCV

covariance_estimator = GraphLassoCV(verbose=1,n_jobs=4)
covariance_estimator.fit(timeseries)
matrix = covariance_estimator.covariance_
print('Covariance matrix has shape {0}.'.format(matrix.shape))

# Plot
from nilearn import plotting
import matplotlib.pyplot as plt

plotting.plot_matrix(matrix, vmin=-1., vmax=1., colorbar=True,
                     title='Power correlation matrix')

# Tweak edge_threshold to keep only the strongest connections.
plotting.plot_connectome(matrix, coords, title='Power correlation graph',
                         edge_threshold='99.8%', node_size=20, colorbar=True)


file = r'D:\WorkStation_2018\WorkStation_dynamicFC_V3\Data\results\results_dfc\group_centroids_2.mat'
matrix = read_mat(file)

yeo_file = r'D:\My_Codes\Github_Related\Github_Code\Template_CBIG\stable_projects\brain_parcellation\Yeo2011_fcMRI_clustering\1000subjects_reference\Yeo_JNeurophysiol11_SplitLabels\MNI152\Yeo2011_17Networks_N1000.split_components.FSL_MNI152_2mm.nii.gz'
from nilearn import input_data
smoothed_img = input_data.smooth_img(yeo_file)  
yeo = datasets.fetch_atlas_yeo_2011()
print('Yeo atlas nifti image (3D) with 17 parcels and liberal mask is located '
      'at: %s' % yeo['thick_17'])
coords = plotting.find_parcellation_cut_coords(labels_img=yeo['thick_17'])
plotting.plot_connectome(matrix, coords, title='Power correlation graph',
                         edge_threshold='99.8%', node_size=20, colorbar=True)
