# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:16:42 2018

@author: lenovo
"""

#coding:utf-8
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\neural_network')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\neural_network\connectome_conv_net')

import tensorflow as tf
import lc_CNN_forward_FC
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#=============================================================================

class CNN_backward():
    
    def __init__(sel,
                BATCH_SIZE = 50,
                LEARNING_RATE_BASE =  0.001,
                LEARNING_RATE_DECAY = 0.99,
                REGULARIZER = pow(10,-4) ,
                STEPS = 2000,
                MOVING_AVERAGE_DECAY = 0.99 ,
                MODEL_SAVE_PATH="./model/" ,
                MODEL_NAME="model.ckpt"):
        
        
        sel.BATCH_SIZE = BATCH_SIZE
        sel.LEARNING_RATE_BASE =  LEARNING_RATE_BASE
        sel.LEARNING_RATE_DECAY = LEARNING_RATE_DECAY
        sel.REGULARIZER = REGULARIZER
        sel.STEPS = STEPS
        sel.MOVING_AVERAGE_DECAY =MOVING_AVERAGE_DECAY
        sel.MODEL_SAVE_PATH=MODEL_SAVE_PATH
        sel.MODEL_NAME=MODEL_NAME
    
    # =============================================================================
    
    def backward(sel,x_train,y_train,x_test, y_test):
        # 复现已有的计算图（后续程序会自动读取已经训练过的模型，即使没有也没有关系，程序会从头训练）
        with tf.Graph().as_default() as g:
            # for train
            x = tf.placeholder(tf.float32,[
                                            sel.BATCH_SIZE,
                                            lc_CNN_forward_FC.MAT_SIZE_ROW,
                                            lc_CNN_forward_FC.MAT_SIZE_COL,
                                            lc_CNN_forward_FC.NUM_CHANNELS]) 
            
            y_ = tf.placeholder(tf.float32, [sel.BATCH_SIZE, lc_CNN_forward_FC.OUTPUT_NODE])
            y = lc_CNN_forward_FC.forward(x,True, sel.REGULARIZER) 
            
            
            # for test
            x_test_tf=tf.placeholder(tf.float32,[
                                            len(x_test),
                                            lc_CNN_forward_FC.MAT_SIZE_ROW,
                                            lc_CNN_forward_FC.MAT_SIZE_COL,
                                            lc_CNN_forward_FC.NUM_CHANNELS]) 
        
            y_test_tf=tf.placeholder(tf.float32, [len(x_test), lc_CNN_forward_FC.OUTPUT_NODE])

            
            
            global_step = tf.Variable(0, trainable=False) 
            
            # 定义损失函数(交叉熵)
            ce =tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
            cem = tf.reduce_mean(ce) 
            with tf.name_scope('loss') as scope:
                loss_train_tf = cem + tf.add_n(tf.get_collection('losses')) 
            
            # test data 的交叉熵以及将label变为一列
            y_pred_tf=lc_CNN_forward_FC.forward(x_test_tf,False, None) 
            y_pred_prob_tf=lc_CNN_forward_FC.forward(x_test_tf,False, None) 

            ce_test =tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_tf, labels=y_test_tf)
            loss_test_tf = tf.reduce_mean(ce_test) 
            
            y_pred_tf = tf.nn.softmax(y_pred_tf)
            y_pred_tf =tf.argmax(y_pred_tf, 1)
            
            #指数衰减学习率
            learning_rate = tf.train.exponential_decay( 
                sel.LEARNING_RATE_BASE,
                global_step,
                len(y_train)/ sel.BATCH_SIZE, 
        		sel.LEARNING_RATE_DECAY,
                staircase=True) 
            
            # 优化器
            train_step =  tf.train.AdamOptimizer(learning_rate).minimize(loss_train_tf, global_step=global_step)
            
            #滑动平均参数
            ema = tf.train.ExponentialMovingAverage(sel.MOVING_AVERAGE_DECAY, global_step)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([train_step, ema_op]): 
                train_op = tf.no_op(name='train')
                
            # 实例化保存器，并开始会话
            saver = tf.train.Saver() 
            
            # 保存loss
            if not os.path.exists(sel.MODEL_SAVE_PATH):
                os.makedirs(sel.MODEL_SAVE_PATH)
            Loss_train=np.array([])
            Loss_test=np.array([])
            
            with tf.Session() as sess: 
                init_op = tf.global_variables_initializer() 
                sess.run(init_op) 
                
                # if there is trained model, then continue training it... 
                ckpt = tf.train.get_checkpoint_state(sel.MODEL_SAVE_PATH) 
                if ckpt and ckpt.model_checkpoint_path:
                	saver.restore(sess, ckpt.model_checkpoint_path) 
                
                for i in range(sel.STEPS):
                    # x_train of current batch
                    xs, ys= sel.lc_next_batch(x_train,y_train,i)
                    
                    _,loss_train,step = sess.run([train_op, loss_train_tf,global_step], \
                                                   feed_dict={x: xs, y_: ys}) 
                    
                    
                    # current test error 
                    y_pred= sess.run(y_pred_tf,feed_dict={x_test_tf: x_test})
                    y_pred_prob=sess.run(y_pred_prob_tf,feed_dict={x_test_tf: x_test})
                    loss_test=sess.run(loss_test_tf,feed_dict={x_test_tf: x_test,y_test_tf:y_test})
                    
                    # 保存Loss
                    Loss_train=np.hstack([Loss_train,loss_train])
                    Loss_test=np.hstack([Loss_test,loss_test])
                    
                    if i % 300 == 0: 
                        # save loss
                        np.save(os.path.join(sel.MODEL_SAVE_PATH,"Loss_train.npy"),Loss_train)
                        np.save(os.path.join(sel.MODEL_SAVE_PATH,"Loss_test.npy"),Loss_test)
                        
                        # performance of training batch data and all testing data
                        print(("-"*10+"Training step(s)={}"+"-"*10+"\n").format(step))
                        
                        print("Training loss_train={:4f}\nTesting loss_train={:4f}\n".format(loss_train,loss_test))
                        
                        acc,sens,spes,auc_score=sel.eval_prformance(y_test,y_pred,y_pred_prob)
                        print('Testing:accuracy={}\nsensitivity={}\nspecificity={}\nauc={}\n'.format(acc,sens,spes,auc_score))
                        
                        saver.save(sess, os.path.join(sel.MODEL_SAVE_PATH, sel.MODEL_NAME), global_step=global_step)
    
        return Loss_train,Loss_test
    
    def lc_next_batch(sel,x_train,y_train,i):
         #
         i=i%np.floor((x_train.shape[0]/sel.BATCH_SIZE))
     
         start=np.int(np.min([i*sel.BATCH_SIZE,x_train.shape[0]]))
         end=np.int(np.min([i*sel.BATCH_SIZE+sel.BATCH_SIZE,x_train.shape[0]]))
         return x_train[start:end,:,:,:],y_train[start:end]
     
    def gen_dummy_variable(sel,label):
        # encode label to one-hot
        n_class = np.int(label.max() + 1)
        n_sample = np.int(label.shape[0])
        dummy_label = np.zeros((n_sample, n_class))
        label_int=[np.int(label_) for label_ in label]
        label_int=np.array(label_int)
        
        for i in range(len(dummy_label)):
            dummy_label[i,label_int[i]]=1
        
        return dummy_label
    
    def eval_prformance(sel,y_test_one_hot,y_pred,y_pred_prob):  
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
         
    def main_train_and_test(sel,x_train,y_train,x_test, y_test):
        import time
        s=time.time()
        Loss_train,Loss_test=sel.backward(x_train,y_train,x_test, y_test)
        e=time.time()
        print('Running time:{}\n'.format(e-s))
        
        return Loss_train,Loss_test
if __name__ == '__main__':

     import lc_CNN_backward_FC as backward
     sel=backward.CNN_backward()

     



