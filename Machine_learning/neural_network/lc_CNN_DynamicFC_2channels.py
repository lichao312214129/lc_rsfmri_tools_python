# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:31:09 2018
对我自己的动态FC训练卷积神经网络模型
@author: lenovo
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\neural_network')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\utils')
from lc_read_write_Mat import read_mat

from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np

class CNN_FC_2channels():
    
    def __init__(sel):
        # input
        sel.mat_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic'
        sel.feature_mean='mean18'
        sel.feature_std='std18'
        sel.label='folder_label18.xlsx'
        
        sel.NUM_CHANNELS=2
    
    
    def load_data_and_label(sel):
        data_name_mean=os.path.join(sel.mat_path,sel.feature_mean)
        data_name_std=os.path.join(sel.mat_path,sel.feature_std)
        
        label_name=os.path.join(sel.mat_path,sel.label)
        
        sel.data_mean=read_mat(data_name_mean,dataset_name=None)
        sel.data_std=read_mat(data_name_std,dataset_name=None)
        
        sel.label=pd.read_excel(label_name)['诊断']
        return sel
    
    def prepare_data(sel):
        
        # 筛选两个诊断
        label=pd.concat([sel.label[sel.label==1],sel.label[sel.label==3]]) 
        sel.data_mean=np.vstack([sel.data_mean[sel.label==1],sel.data_mean[sel.label==3]])
        sel.data_std=np.vstack([sel.data_std[sel.label==1],sel.data_std[sel.label==3]])
        
        sel.label=label
        
        # 2d to 4d
        n_node=114
        
        mask=np.ones([n_node,n_node])
        mask=np.triu(mask,1)==1 # 前期只提取了上三角（因为其他的为重复）
        sel.data_reshape=np.zeros([len(sel.label),n_node,n_node,sel.NUM_CHANNELS])
        for i in range(sel.data_reshape.shape[0]):
            
            # 将1维的上三角矩阵向量，嵌入到矩阵的上三角上
            sel.data_reshape[i,mask,0]=sel.data_mean[i,:]
            sel.data_reshape[i,mask,1]=sel.data_std[i,:]
            
            # 将上三角矩阵，对称地镜像到下三角，从而组成完整的矩阵
            sel.data_reshape[i,:,:,0]+=sel.data_reshape[i,:,:,0].T
            sel.data_reshape[i,:,:,1]+=sel.data_reshape[i,:,:,1].T
            
            # 对角线补上1
            sel.data_reshape[i,:,:,0]+=np.eye(n_node)
            sel.data_reshape[i,:,:,1]+=np.eye(n_node)
        
        
        # 平衡样本
        sel.label=sel.label[60:]
        sel.data_reshape=sel.data_reshape[60:,:,:,:]
        print(sum(sel.label==1),sum(sel.label==3))
        
        
        # split data
        x_train, x_test, y_train, y_test = \
            train_test_split(sel.data_reshape, sel.label, random_state=0)
        
        # np.fload32
        sel.x_train, sel.x_test, sel.y_train, sel.y_test =\
                                                        np.float32(x_train),\
                                                        np.float32(x_test),\
                                                        np.float32(y_train),\
                                                        np.float32(y_test )
         
        # dummy label
        sel.y_train=sel.gen_dummy_variable(sel.y_train)
        sel.y_test=sel.gen_dummy_variable(sel.y_test)
        
        return sel
    
    def gen_dummy_variable(sel,label):
        # encode label to one-hot
        n_class = np.int(len(np.unique(label)))#我的label为1-4，如果label为0-4时，要修改
        n_sample = np.int(label.shape[0])
        dummy_label = np.zeros((n_sample, n_class))
        
        label_unique=np.unique(label)

        label_int=[np.int(label_) for label_ in label]
        label_int=np.array(label_int)
        
        for i in range(len(dummy_label)):
            dummy_label[i,np.argwhere(label_unique==label_int[i])]=1
        
        return dummy_label
     
    
     
    def train(sel):
        import lc_CNN_backward_FC as backward
        bw=backward.CNN_backward()
        bw.main_train_and_test(sel.x_train,sel.y_train,sel.x_test, sel.y_test)

        
if __name__=='__main__':
    import lc_CNN_DynamicFC_2channels as CNN
    sel=CNN.CNN_FC_2channels()
    
    sel=sel.load_data_and_label()
    sel=sel.prepare_data()
    sel.train()
    
    
    
    