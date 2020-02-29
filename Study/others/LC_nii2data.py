import nibabel as nib
import os
import numpy as np
def nii2data():
    # input: image path
    path_img_p=os.path.normcase('D:\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\P_Weighted_selected')
    path_img_c=os.path.normcase('D:\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\C_Weighted_selected')
    dim_p=[61,73,61]  
    dim_c=[61,73,61]  
    
    # all image name
    imgName_p = os.listdir(path_img_p)
    imgName_c = os.listdir(path_img_c)
    #
    num_p=len(imgName_p)
    num_c=len(imgName_c)
    dim_p.append(num_p)
    dim_c.append(num_c)
    img_data_p=np.zeros(dim_p)
    img_data_c=np.zeros(dim_c)
    # all subjects' image path join name
    for i in range(num_p):
        imgName_p[i]=os.path.join(path_img_p,imgName_p[i])
        img = nib.load(imgName_p[i])
        img_data_p[:,:,:,i] = img.get_data()
        img_data_p_shape = img_data_p.shape
    
    for i in range(num_c):
        imgName_c[i]=os.path.join(path_img_c,imgName_c[i])
        img = nib.load(imgName_c[i])
        img_data_c[:,:,:,i] = img.get_data()
        img_data_c_shape = img_data_c.shape
    return img_data_p,img_data_c,img_data_p_shape,img_data_c_shape
if __name__ == '__main__':
    nii2data()