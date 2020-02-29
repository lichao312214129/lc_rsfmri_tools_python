# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:19:17 2018
"""

import sys
import os
filepath = os.getcwd()
root = os.path.dirname(os.path.dirname(filepath))
sys.path.append(root)

import pandas as pd
import tensorflow as tf
import Machine_learning.neural_network.lc_CNN_forward_FC
import numpy as np
import os
import Machine_learning.neural_network.lc_CNN_backward_FC as backward
bw=backward.CNN_backward()

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def test(x_test,y_test,which_model='last'):
   # 复现已有的计算图
   with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[
                                        None,
                                        lc_CNN_forward_FC.MAT_SIZE_ROW,
                                        lc_CNN_forward_FC.MAT_SIZE_COL,
                                        lc_CNN_forward_FC.NUM_CHANNELS]) 
        
        y_ = tf.placeholder(tf.float32, [None, lc_CNN_forward_FC.OUTPUT_NODE])
        y = lc_CNN_forward_FC.forward(x,False, None) 
        y_pred_tf = tf.nn.softmax(y)
        y_pred_tf =tf.argmax(y_pred_tf, 1)

        # loss
        ce_test =tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
        loss = tf.reduce_mean(ce_test) 
        
        # 实例化带滑动平均的保存器
        ema = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        # 开始会话
        with tf.Session() as sess:
            # fetch trained model if exist
            ckpt = tf.train.get_checkpoint_state(os.path.join(sel.model_path,sel.save_model_name))
            if ckpt and ckpt.model_checkpoint_path:
                print('loading ckpt!\n')
                # test use all model
                # generally, this step is used used to monitor the training process
                if which_model=='all':
                    loss_test=np.array([])
                    accuracy_test=np.array([])
                    sensitivity_test=np.array([])
                    specificity_test=np.array([])
                    auc_test=np.array([])
                    
                    # 提取model文件夹下所有model名
                    all_model_name=fetch_all_model_under_one_folder()
                    # 每个模型的测试结果
                    for model_name in all_model_name.iloc[:,0]:
                        model_name=os.path.join(sel.model_path,sel.save_model_name,model_name)
                        saver.restore(sess,model_name)               
                        
                        ls=sess.run(loss, feed_dict={x: x_test,y_:y_test})
                        y_pred= sess.run(y_pred_tf,feed_dict={x: x_test})
                        y_pred_prob=sess.run(y,feed_dict={x: x_test})

                        # save test performances of each model
                        loss_test=np.hstack([loss_test,ls]) 
                        acc,sens,spes,auc_score=eval_prformance(y_test,y_pred,y_pred_prob)
                        accuracy_test=np.hstack([accuracy_test,acc])
                        sensitivity_test=np.hstack([sensitivity_test,sens])
                        specificity_test=np.hstack([specificity_test,spes])
                        auc_test=np.hstack([auc_test,auc_score])
                        
                # test use the  last model    
                elif which_model=='last':
                    
                    # reload model
                    # ckpt.model_checkpoint_path could be replaced with specified model name
                    # such as the model that the test dataset has the minimum loss
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('reloaded the last model\n')
                    
                    loss_test=sess.run(loss, feed_dict={x: x_test,y_:y_test})
                    y_pred= sess.run(y_pred_tf,feed_dict={x: x_test})
                    y_pred_prob=sess.run(y,feed_dict={x: x_test})
                    
                    (accuracy_test,
                    sensitivity_test,
                    specificity_test,
                    auc_test)=eval_prformance(y_test,y_pred,y_pred_prob)
                    
                    # print test performances
                    print('Testing:loss={:4f}\naccuracy={}\nsensitivity={}\nspecificity={}\nauc={}\n'.format(
                                                                                          loss_test,
                                                                                          accuracy_test,
                                                                                          sensitivity_test,
                                                                                          specificity_test,
                                                                                          auc_test))
                    
                else:
                    print('to use specified model\n')
                    # reload model
                    # ckpt.model_checkpoint_path could be replaced with specified model name
                    # such as the model that the test dataset has the minimum loss
                    model_name=os.path.join(sel.model_path,sel.save_model_name,which_model)
                    saver.restore(sess,model_name)    

                    loss_test=sess.run(loss, feed_dict={x: x_test,y_:y_test})
                    y_pred= sess.run(y_pred_tf,feed_dict={x: x_test})
                    y_pred_prob=sess.run(y,feed_dict={x: x_test})
                    
                    (accuracy_test,
                    sensitivity_test,
                    specificity_test,
                    auc_test)=eval_prformance(y_test,y_pred,y_pred_prob)
                    
                    # print test performances
                    print('Testing:loss={:4f}\naccuracy={}\nsensitivity={}\nspecificity={}\nauc={}\n'.format(
                                                                                          loss_test,
                                                                                          accuracy_test,
                                                                                          sensitivity_test,
                                                                                          specificity_test,
                                                                                          auc_test))
                
                return (
                        y_pred,
                        y_pred_prob,
                        loss_test,
                        accuracy_test,
                        sensitivity_test,
                        specificity_test,
                        auc_test
                        )
                
            else:
                print('No checkpoint file found!\n')
                return None,None
                
            

def gen_dummy_variable(label):
    # encode label to one-hot
    n_class = np.int(label.max() + 1)
    n_sample = np.int(label.shape[0])
    dummy_label = np.zeros((n_sample, n_class))
    label_int=[np.int(label_) for label_ in label]
    label_int=np.array(label_int)
    
    for i in range(len(dummy_label)):
        dummy_label[i,label_int[i]]=1
    
    return dummy_label

def normalization(data):
    # because of our normalization level is on subject, 
    # we should transpose the data matrix on python(but not on matlab)
    scaler = preprocessing.StandardScaler().fit(data.T)
    z_data=scaler.transform(data.T) .T
    return z_data
    
def eval_prformance(y_test_one_hot,y_pred,y_pred_prob):  
    y_test_1d=np.argmax(y_test_one_hot,1)
    accuracy=float('%.4f' %(accuracy_score(y_test_1d,y_pred)))
    
    report=classification_report(y_test_1d,y_pred)
    report=report.split('\n')
    specificity=report[2].strip().split(' ')
    sensitivity=report[3].strip().split(' ')
    specificity=float([spe for spe in specificity if spe!=''][2])
    sensitivity=float([sen for sen in sensitivity if sen!=''][2])
    auc_score=float('%.4f' % roc_auc_score(y_test_one_hot,y_pred_prob))
    
    return accuracy,sensitivity,specificity,auc_score

def fetch_all_model_under_one_folder():
    """ 获得model文件夹下面所有model的名称，
    用于评估训练过程中training过程以及test过程的变化
    """
    model_folder=os.path.join(sel.model_path,sel.save_model_name)
    all_model=os.listdir(model_folder)
    all_model=pd.Series(all_model)
    #正则表达筛选model name
    all_model_ind=all_model.str.contains('model.ckpt-\d+')
    all_model_name=all_model[all_model_ind]
    all_model_name=all_model_name.str.findall('model.ckpt-\d+')
    #将Series中的列表元素，转换为str(默认提取每个匹配列表的第0个元素)
    ith_in_list=0
    all_model_name=[name[ith_in_list] for name in all_model_name]
    all_model_name=pd.DataFrame(all_model_name)
    # 去掉重复的模型名
    all_model_name=all_model_name.drop_duplicates()
    # 提取global_step
    global_step=all_model_name.iloc[:,0].str.findall('\d+')
    ith_in_list=0
    global_step=[np.int(step[ith_in_list]) for step in global_step]
    global_step=pd.DataFrame(global_step)
    global_step=global_step.sort_values(by=0)
    # 根据global_step 从小到大，更新all_model_name的顺序（由于global step的index已经变化，
    # 所以把all model name的index也改变）
    all_model_name.index=np.arange(0,len(all_model_name),1)
    all_model_name=all_model_name.loc[global_step.index]
    return all_model_name



if __name__ == '__main__':
    import lc_CNN_DynamicFC_1channels_V2 as CNN
    sel=CNN.CNN_FC_1channels(
                 x=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\x_test206.npy',
                 y=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zStatic\y_test206.npy',
                 label_col=0,
                 scale_method='StandardScaler',
                 NUM_CHANNELS=1,
                 model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic',
                 save_model_name='model_static_add_26')
    
    sel=sel.load_data_and_label()
    sel=sel.prepare_data()
    x_test,y_test=sel.norm_x_4d,sel.y
    
    # testing
    which_model='all'
    (y_pred_test,
    y_pred_prob_test,
    loss_test,
    accuracy_test,
    sensitivity_test,
    specificity_test,
    auc_test)=test(sel.norm_x_4d,sel.y,which_model=which_model)
    
    # plot
    if which_model=='all':
        balanced_accuracy_test=(sensitivity_test+specificity_test)/2
        plt.figure()
        plt.title('test')
    #    plt.plot(loss_test,'-')
        plt.plot(balanced_accuracy_test,'-*')
        plt.plot(auc_test,'-*')
    #    plt.plot(sensitivity_test,'-')
    #    plt.plot(specificity_test,'-')
        plt.legend(['balanced_accuracy','auc'])
        plt.show()
        
        # test loss最小时，test性能
        ind=np.argwhere(loss_train==np.min(loss_train))
        min_loss_test=np.min(loss_test)
        best_balanced_accuracy_test=balanced_accuracy_test[ind]
        best_sensitivity_test=sensitivity_test[ind]
        best_specificity_test=specificity_test[ind]
        best_auc_test=auc_test[ind]
        print('best test performances={}/{}/{}/{}'.format(best_balanced_accuracy_test,
                                                      best_sensitivity_test,
                                                      best_specificity_test,
                                                      best_auc_test))
    
    

