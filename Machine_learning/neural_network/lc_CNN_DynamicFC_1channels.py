# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:31:09 2018
对我自己的动态FC训练卷积神经网络模型
one channel
@author: lenovo
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\machine_learning_python\Machine_learning\neural_network')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\machine_learning_python\Utils')
sys.path.append(r'D:\LI_Chao_important_do_not_delete\Code\machine_learning_python\Machine_learning\classfication')
from lc_read_write_Mat import read_mat
import lc_scaler as scl

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CNN_FC_1channels():
    
    def __init__(sel,
                 mat_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\static',
                 label_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\folder_label_static.xlsx',
                 
                 which_group_to_classify=[1,3],
                 
                 scale_method='StandardScaler',
                 NUM_CHANNELS=1,
                 
                 model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic',
                 save_model_name='model_static'):
        
        # input
        sel.mat_path=mat_path
        sel.label_path=label_path
        
        sel.which_group_to_classify=which_group_to_classify #哪两个组需要分类
        
        sel.scale_method=scale_method
        sel.NUM_CHANNELS=NUM_CHANNELS
        
        sel.model_path=model_path
        sel.save_model_name=save_model_name

    
    def load_data_and_label(sel):
        
        sel.data=read_mat(sel.mat_path,dataset_name=None)
        
        sel.label=pd.read_excel(sel.label_path)['诊断']
        sel.folder=pd.read_excel(sel.label_path)['folder']
        return sel
    
    def load_data_and_label1(sel):
        
        sel.data=read_mat(sel.mat_path,dataset_name=None)
        
        sel.label=pd.read_excel(sel.label_path)['诊断']
        sel.folder=pd.read_excel(sel.label_path)['folder']
        return sel
    
    def prepare_data(sel):
        
        # to np.fload32 for tensorflow
        sel.data=np.float32(sel.data)
        
        # dropna
        # 1) data
        sel.data=pd.DataFrame(sel.data)
        sel.data=sel.data.dropna()
        ind=list(sel.data.index)
        sel.data=sel.data
        # 2) label
        sel.label=pd.DataFrame(sel.label)
        sel.label.index=np.arange(0,len(sel.label))
        sel.label=sel.label.loc[ind]
                                                                
        # select which group into classification
        label=pd.concat([sel.label[sel.label.iloc[:,0]==sel.which_group_to_classify[0]],
                         sel.label[sel.label.iloc[:,0]==sel.which_group_to_classify[1]]],
                         axis=0) 
        sel.data=sel.data.loc[label.index,:].values
        sel.label=label
        
        '''
        # split to training,validation and testing datasets
        sel.folder=sel.folder.loc[sel.label.index]
        folder_label=pd.concat([sel.folder,sel.label],axis=1)
        folder_label_all380=folder_label.iloc[135:,:]
        data_all380=sel.data[135:,:]
        
        folder_label_all380.to_excel('folder_label_all380.xlsx')
        np.save('data_all380.mat',data_all380)
        
        x_train, x_test, y_train, y_test = \
            train_test_split(data_all380, folder_label_all380, random_state=10,test_size=0.2)
            
        np.save('x_train304',x_train)
        np.save('y_train304',y_train)
        
        x_val, x_test, y_val, y_test = \
            train_test_split(x_test, y_test, random_state=10,test_size=0.5)
            
        np.save('x_val38',x_val)
        np.save('y_val38',y_val)
        
        np.save('x_test38',x_test)
        np.save('y_test38',y_test)
        '''
        
      # dummy label
        sel.label=sel.gen_dummy_variable(sel.label.iloc[:,0])
        
        # normalization on the subject level
        sel.norm_data=sel.normalization(sel.data)
        
        # balance the samples
        sel.label=sel.label[130:,:]
        sel.norm_data=sel.norm_data[130:,:]
        print(sum(sel.label[:,0]),sum(sel.label[:,1]))
        
        # split data to training and testing sub-data
        sel.x_train, sel.x_test, sel.y_train, sel.y_test = \
            train_test_split(sel.norm_data, sel.label, random_state=10,test_size=0.2)
        print('training/testing sample size={}/{}'.format(len(sel.y_train),len(sel.y_test)))
        
        # 2d to 4d for tensorflow
        sel.x_train=sel.trans_2d_to_4d(sel.x_train)
        sel.x_test=sel.trans_2d_to_4d(sel.x_test)
        
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
    
    def trans_2d_to_4d(sel,data_2d):
        # 2d to 4d for tensorflow
        n_node=114
        mask=np.ones([n_node,n_node])
        mask=np.triu(mask,1)==1 # 前期只提取了上三角（因为其他的为重复）
        data_4d=np.zeros([len(data_2d),n_node,n_node,sel.NUM_CHANNELS])
        for i in range(data_4d.shape[0]):
            # 1)将1维的上三角矩阵向量，嵌入到矩阵的上三角上
            data_4d[i,mask,0]=data_2d[i,:]
            # 2)将上三角矩阵，对称地镜像到下三角，从而组成完整的矩阵
            data_4d[i,:,:,0]+=data_4d[i,:,:,0].T
            # 3)对角线补上1
            data_4d[i,:,:,0]+=np.eye(n_node)
        return data_4d
        
        
    def normalization(sel,data):
        # because of our normalization level is on subject, 
        # we should transpose the data matrix
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data=scaler.transform(data.T) .T
        return z_data
    
    def normalization_sep(sel,train_X,test_X,scale_method):
        # normalize training and testing data separately
        # all normalization parameter derived from training data
        train_X,model=scl.scaler(train_X,scale_method)
        test_X=model.transform(test_X)
        return train_X,test_X
         
    def train(sel):
        print('start training...\n')
        import lc_CNN_backward_FC as backward
        bw=backward.CNN_backward(   MODEL_SAVE_PATH=os.path.join(sel.model_path,sel.save_model_name),
                                    BATCH_SIZE = 10,
                                    LEARNING_RATE_BASE =  0.0001,
                                    LEARNING_RATE_DECAY = 0.99,
                                    REGULARIZER = pow(10,-4) ,
                                    STEPS = 3000,
                                    MOVING_AVERAGE_DECAY=0.99,)
        Loss_train,Loss_test=bw.main_train_and_test(sel.x_train,sel.y_train,sel.x_test, sel.y_test)
        
        return Loss_train,Loss_test

        
if __name__=='__main__':
    import lc_CNN_DynamicFC_1channels as CNN
    
    sel=CNN.CNN_FC_1channels(
                 mat_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\static_add',
                 label_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\folder_label_static_add.xlsx',
                 which_group_to_classify=[1,3],
                 scale_method='StandardScaler',
                 NUM_CHANNELS=1,
                 model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic',
                 save_model_name='model_static_add')
    
    
    sel=sel.load_data_and_label()
    sel=sel.prepare_data()
    loss_train,loss_test=sel.train()
    
    # show loss
    fig,ax=plt.subplots()
    ax.plot(np.load(os.path.join(sel.model_path,sel.save_model_name,"Loss_train.npy")))
    ax.plot(np.load(os.path.join(sel.model_path,sel.save_model_name,"Loss_test.npy")))
    plt.legend(['training loss','testing loss'])
    plt.show()
    
    
    
    