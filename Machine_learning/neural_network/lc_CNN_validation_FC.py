# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:40:49 2018
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Machine_learning\neural_network')
import time
import tensorflow as tf
import lc_CNN_forward_FC
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 导入backfoward
import lc_CNN_backward_FC as backward
bw=backward.CNN_backward()
validation_INTERVAL_SECS = 5

def validation(x_validation,y_validation,which_model='last',just_fetch=True):
    """复现已有的计算图（后续程序会自动读取已经训练过的模型，即使没有也没有关系，程序会从头训练）"""
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
        ce_validation =tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
        loss = tf.reduce_mean(ce_validation) 

        # 实例化带滑动平均的保存器
        ema = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        if_continue=True
        while if_continue==True:
            with tf.Session() as sess:
                # fetch trained model if exist
                ckpt = tf.train.get_checkpoint_state(os.path.join(sel.model_path,sel.save_model_name))
                if ckpt and ckpt.model_checkpoint_path:
                    # validation use all model
                    # generally, this step is used used to monitor the training process
                    if which_model=='all':
                        print('loading ckpt\n')
                        all_step=np.array([])
                        loss_validation=np.array([])
                        accuracy_validation=np.array([])
                        sensitivity_validation=np.array([])
                        specificity_validation=np.array([])
                        auc_validation=np.array([])
                        
                        # 提取model文件夹下所有model名
                        all_model_name=fetch_all_model_under_one_folder()
                        for model_name in all_model_name.iloc[:,0]:
                            model_name=os.path.join(sel.model_path,sel.save_model_name,model_name)
                            saver.restore(sess,model_name)                
                            global_step=model_name.split('-')[-1]
                            
                            ls=sess.run(loss, feed_dict={x: x_validation,y_:y_validation})
                            y_pred= sess.run(y_pred_tf,feed_dict={x: x_validation})
                            y_pred_prob=sess.run(y,feed_dict={x: x_validation})

                            # save validation performances of each model
                            all_step=np.hstack([all_step,global_step])
                            loss_validation=np.hstack([loss_validation,ls]) 
                            acc,sens,spes,auc_score=eval_prformance(y_validation,y_pred,y_pred_prob)
                            accuracy_validation=np.hstack([accuracy_validation,acc])
                            sensitivity_validation=np.hstack([sensitivity_validation,sens])
                            specificity_validation=np.hstack([specificity_validation,spes])
                            auc_validation=np.hstack([auc_validation,auc_score])
                            
                            # print validation performances
#                            print('\t\t\t### Global step={} ###'.format(global_step))
#                            print('validationing:loss={:4f}\naccuracy={}\nsensitivity={}\nspecificity={}\nauc={}\n'.format(ls,acc,sens,spes,auc_score))
                    
                    # validation use the  last model    
                    elif which_model=='last':
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step=ckpt.model_checkpoint_path.split('-')[-1]
                        all_step=global_step
                        print('reloaded the last model\n')
                        
                        loss_validation=sess.run(loss, feed_dict={x: x_validation,y_:y_validation})
                        y_pred= sess.run(y_pred_tf,feed_dict={x: x_validation})
                        y_pred_prob=sess.run(y,feed_dict={x: x_validation})
                        
                        (accuracy_validation,
                        sensitivity_validation,
                        specificity_validation,
                        auc_validation)=\
                                    eval_prformance(y_validation,y_pred,y_pred_prob)
                        
                        # print validation performances
                        print('Validation:loss={:4f}\naccuracy={}\nsensitivity={}\nspecificity={}\nauc={}\n'.format(
                              loss_validation,
                              accuracy_validation,
                              sensitivity_validation,
                              specificity_validation,
                              auc_validation))        
                    else:
                        print('###specify which model to load!###\n')
                        return
                        
                else:
                    print('No checkpoint file found')
                    return
                
            # if just fetch performance, then stop validation loop
            if just_fetch:
                if_continue=False
            else:
                time.sleep(validation_INTERVAL_SECS) 
                
    return (y_pred,
           y_pred_prob,
           all_step,
           loss_validation,
           accuracy_validation,
           sensitivity_validation,
           specificity_validation,
           auc_validation)
                
        
            

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


def eval_prformance(y_validation_one_hot,y_pred,y_pred_prob):  
    y_validation_1d=np.argmax(y_validation_one_hot,1)
    accuracy=float('%.4f' %(accuracy_score(y_validation_1d,y_pred)))
    
    report=classification_report(y_validation_1d,y_pred)
    report=report.split('\n')
    specificity=report[2].strip().split(' ')
    sensitivity=report[3].strip().split(' ')
    specificity=float([spe for spe in specificity if spe!=''][2])
    sensitivity=float([sen for sen in sensitivity if sen!=''][2])
    auc_score=float('%.4f' % roc_auc_score(y_validation_one_hot,y_pred_prob))
    
    return accuracy,sensitivity,specificity,auc_score

def fetch_all_model_under_one_folder():
    """ 获得model文件夹下面所有model的名称，
    用于评估训练过程中training过程以及validation过程的变化
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

def fetch_train_loss(pattern='loss_train'):
    """ 加载model文件夹下所有loss_train文件"""
    train_loss_path=os.path.join(sel.model_path,sel.save_model_name)
    all_file_name=os.listdir(train_loss_path)
    all_file_name=pd.Series(all_file_name)
    train_loss_ind=all_file_name.str.contains(pattern)
    all_train_loss_name=all_file_name[train_loss_ind]
    all_train_loss_path=[os.path.join(train_loss_path,name) for name in all_train_loss_name]
    all_train_loss=[np.load(path) for path in all_train_loss_path]
    all_train_loss=np.hstack(all_train_loss)
    return all_train_loss
   
if __name__ == '__main__':
    import lc_CNN_DynamicFC_1channels_V2 as CNN
    sel=CNN.CNN_FC_1channels(
                 x=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\machine_learning_model_and_data\data\x_val76.npy',
                 y=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\machine_learning_model_and_data\data\y_val76.npy',
                 label_col=1,
                 scale_method='StandardScaler',
                 NUM_CHANNELS=1,
                 model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\machine_learning_model_and_data',
                 save_model_name='model_static_add_26')
    
    sel=sel.load_data_and_label()
    sel=sel.prepare_data()
    x_validation,y_validation=sel.norm_x_4d,sel.y
    
    """ validation """
    which_model='all'
    ( y_pred_val,
      y_pred_prob_val,
       all_step,
       loss_validation,
       accuracy_validation,
       sensitivity_validation,
       specificity_validation,
       auc_validation
     )=validation(x_validation,y_validation,which_model=which_model,just_fetch=True)
    

    """plot"""
    if which_model=='all':
        loss_train=fetch_train_loss(pattern='loss_train')
        # plot loss
        plt.figure()
        plt.title('training and validation loss')
        plt.plot(loss_train,'r-')
        plt.plot(loss_validation,'b-')
        plt.legend(['loss_train','loss_validation'])
        
        """plot acc and auc"""
        balanced_accuracy_validation=(sensitivity_validation+specificity_validation)/2
        plt.figure()
        plt.title('train')
    #    plt.plot(ref_line,'k--')
        plt.plot(balanced_accuracy_validation,'-*')
        plt.plot(auc_validation,'-*')
    #    plt.plot(sensitivity_validation,'-')
    #    plt.plot(specificity_validation,'-')
        plt.legend(['balanced_accuracy_validation','auc'])

        """loss最小的性能"""
        min_loss_validation=np.min(loss_validation)
        ind=np.argwhere(loss_validation==np.min(loss_validation))
        
        best_balanced_accuracy_validation=balanced_accuracy_validation[ind]
        best_sensitivity_validation=sensitivity_validation[ind]
        best_specificity_validation=specificity_validation[ind]
        best_auc_validation=auc_validation[ind]
        print('best validation performances={}/{}/{}/{}'.format(best_balanced_accuracy_validation,
                                                      best_sensitivity_validation,
                                                      best_specificity_validation,
                                                      best_auc_validation))

    
    
    
    