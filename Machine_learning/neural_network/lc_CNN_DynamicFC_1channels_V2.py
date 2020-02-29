# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:10:50 2018
对我自己的动态FC训练卷积神经网络模型
已经把train，validation以及test数据集准备好
one channel
@author: lenovo
"""

import sys
import os
filepath = os.getcwd()
root = os.path.dirname(os.path.dirname(filepath))
sys.path.append(root)
from Utils import lc_scaler as scl

from sklearn import preprocessing
import os
import numpy as np
import matplotlib.pyplot as plt


class CNN_FC_1channels():

    def __init__(self,
                 x_file=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\machine_learning_data\x_train304.npy',
                 y_file=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\machine_learning_data\y_train304.npy',
                 label_col=1,
                 scale_method='StandardScaler',
                 NUM_CHANNELS=1,
                 model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic',
                 save_model_name='model_static'):

        # input
        self.x_file = x_file
        self.y_file = y_file
        self.label_col = label_col

        self.scale_method = scale_method
        self.NUM_CHANNELS = NUM_CHANNELS

        self.model_path = model_path
        self.save_model_name = save_model_name

    def load_data_and_label(self):
        self.x = np.load(self.x_file)
        self.y = np.load(self.y_file)[:, self.label_col]
        return self

    def prepare_data(self):
        # dummy label
        self.y = self.gen_dummy_variable(self.y)

        # normalization on the subject level
        self.norm_x = self.normalization(self.x)

        # 2d to 4d for tensorflow
        self.norm_x_4d = self.trans_2d_to_4d(self.norm_x)

        return self

    def gen_dummy_variable(self, label):
        # encode label to one-hot
        n_class = np.int(len(np.unique(label)))  # 我的label为1-4，如果label为0-4时，要修改
        n_sample = np.int(label.shape[0])
        dummy_label = np.zeros((n_sample, n_class))

        label_unique = np.unique(label)

        label_int = [np.int(label_) for label_ in label]
        label_int = np.array(label_int)

        for i in range(len(dummy_label)):
            dummy_label[i, np.argwhere(label_unique == label_int[i])] = 1

        return dummy_label
    
    def normalization(self, data):
        """ 
        Because of our normalization level is on subject,
        we should transpose the data matrix on python(but not on matlab)
        """
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data = scaler.transform(data.T) .T
        return z_data

    def normalization_grouplevel(self, train_X, test_X, scale_method):
        """
        Normalization at the group level
        Normalizing training and testing data separately
        All normalization parameter derived from training data
        """
        train_X, model = scl.scaler(train_X, scale_method)
        test_X = model.transform(test_X)
        return train_X, test_X

    def trans_2d_to_4d(self, data_2d):
        """ 
        Transfer the 2d data to 4d data so that the tensorflow can handle the data
        """
        n_node = 114
        mask = np.ones([n_node, n_node])
        mask = np.triu(mask, 1) == 1  # 前期只提取了上三角（因为其他的为重复）
        data_4d = np.zeros([len(data_2d), n_node, n_node, self.NUM_CHANNELS])
        for i in range(data_4d.shape[0]):
            # 1) 将1维的上三角矩阵向量，嵌入到矩阵的上三角上
            data_4d[i, mask, 0] = data_2d[i, :]
            # 2) 将上三角矩阵，对称地镜像到下三角，从而组成完整的矩阵
            data_4d[i, :, :, 0] += data_4d[i, :, :, 0].T
            # 3) 对角线补上1
            data_4d[i, :, :, 0] += np.eye(n_node)
        return data_4d

    def train(self):
        print('start training...\n')
        import lc_CNN_backward_FC_V2 as backward
        bw = backward.CNN_backward(MODEL_SAVE_PATH=os.path.join(self.model_path, self.save_model_name),
                                   BATCH_SIZE=30,
                                   LEARNING_RATE_BASE=0.001,
                                   LEARNING_RATE_DECAY=0.99,
                                   REGULARIZER=pow(10, -3),
                                   STEPS=500,
                                   MOVING_AVERAGE_DECAY=0.999)

        print('positive sample={}\nnegative sample={}\n'.format(
            np.sum(self.y[:, 0]), np.sum(self.y[:, 1])))
        bw.main_train_and_test(self.norm_x_4d, self.y)



if __name__ == '__main__':
    import lc_CNN_DynamicFC_1channels_V2 as CNN

    sel = CNN.CNN_FC_1channels(
        x_file=r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\machine_learning_model_and_data\data\x_train304.npy',
        y_file=r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\machine_learning_model_and_data\data\y_train304.npy',
        label_col=1,
        scale_method='StanardScaler',
        NUM_CHANNELS=1,
        model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC_V1\Data\machine_learning_model_and_data',
        save_model_name='model_static_add_23_backup_all'
    )

    sel = sel.load_data_and_label()
    sel = sel.prepare_data()
    sel.train()
