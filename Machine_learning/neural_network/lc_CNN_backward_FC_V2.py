# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:56:05 2018
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\neural_network')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\LC_Machine_learning-(Python)\Machine_learning\neural_network\connectome_conv_net')
import lc_CNN_forward_FC
import next_batch as nb

import tensorflow as tf
import os
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
#=============================================================================

class CNN_backward():
    
    def __init__(sel,
                BATCH_SIZE = 50,
                LEARNING_RATE_BASE =  0.005,
                LEARNING_RATE_DECAY = 0.99,
                REGULARIZER = pow(10,-4),
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
    
    def backward(sel,x_train,y_train):
        ds = nb.DataSet(x_train,y_train,len(y_train))# next batch data
        
        # 复现已有的计算图（后续程序会自动读取已经训练过的模型，即使没有也没有关系，程序会从头训练）
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32,[
                                            None,
                                            lc_CNN_forward_FC.MAT_SIZE_ROW,
                                            lc_CNN_forward_FC.MAT_SIZE_COL,
                                            lc_CNN_forward_FC.NUM_CHANNELS]) 

            y_ = tf.placeholder(tf.float32, [None, lc_CNN_forward_FC.OUTPUT_NODE])
            y = lc_CNN_forward_FC.forward(x,True, sel.REGULARIZER) 
            
            global_step = tf.Variable(0, trainable=False) 
            
            # 定义损失函数(交叉熵)
            ce =tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
            cem = tf.reduce_mean(ce) 
            loss_tf = cem + tf.add_n(tf.get_collection('losses')) 
            
            #指数衰减学习率
            learning_rate = tf.train.exponential_decay( 
                sel.LEARNING_RATE_BASE,
                global_step,
                len(y_train)/ sel.BATCH_SIZE, 
        		sel.LEARNING_RATE_DECAY,
                staircase=True) 
            
            # optimizer( batch normalization)
#            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#            with tf.control_dependencies(update_ops):
            train_step =  tf.train.AdamOptimizer(learning_rate).minimize(loss_tf, global_step=global_step)
            
            #滑动平均参数
            ema = tf.train.ExponentialMovingAverage(sel.MOVING_AVERAGE_DECAY, global_step)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([train_step, ema_op]): 
                train_op = tf.no_op(name='train')
                
            # 实例化保存器，并开始会话
            saver = tf.train.Saver(max_to_keep=500) 
            
            # 保存loss
            if not os.path.exists(sel.MODEL_SAVE_PATH):
                os.makedirs(sel.MODEL_SAVE_PATH)
            loss_train=np.array([])
            
            with tf.Session() as sess: 
                init_op = tf.global_variables_initializer() 
                sess.run(init_op) 
                
                # if there is trained model, then continue training it... 
                ckpt = tf.train.get_checkpoint_state(sel.MODEL_SAVE_PATH) 
                if ckpt and ckpt.model_checkpoint_path:
                	saver.restore(sess, ckpt.model_checkpoint_path) 
                
                # current time, used to create non-duplicate filenames
                current_time=time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
                loss_train_path=os.path.join(sel.MODEL_SAVE_PATH,"loss_train_"+current_time+".npy")
                
                for i in range(sel.STEPS):
                    # batch data
                    xs, ys=ds.next_batch(sel.BATCH_SIZE)
#                    xs, ys= sel.lc_next_batch(x_train,y_train,i)
                    
                    _,loss,step = sess.run([train_op, loss_tf,global_step], \
                                                   feed_dict={x: xs, y_: ys}) 
                                    
                    # save current training loss and model per epoch(approximately)
                    if i%(len(y_train)// sel.BATCH_SIZE)==0:
                        loss_train=np.hstack([loss_train,loss])
                        np.save(loss_train_path,loss_train)
                        saver.save(sess, os.path.join(sel.MODEL_SAVE_PATH, sel.MODEL_NAME), global_step=global_step)
                        
                        # print performances
                        print(("-"*10+"Training step(s)={}"+"-"*10+"\n").format(step))
                        print("Training loss={:4f}\n".format(loss))


                        
    
        return loss_train
    
    def lc_next_batch(sel,x_train,y_train,i):
         """
         Current method: only the first integer multiple batch size samples are included, 
         so whenever I restores to the initial value, I randomly scramble the index of the sample once.

         """
         i=i%np.floor((len(y_train)/sel.BATCH_SIZE))
         rand_ind=np.arange(0,len(y_train),1)
         if i==0:
             rand_ind=np.random.permutation(len(y_train))
     
         start=np.int(np.min([i*sel.BATCH_SIZE,x_train.shape[0]]))
         
         end=np.int(np.min([i*sel.BATCH_SIZE+sel.BATCH_SIZE,x_train.shape[0]]))
         
         return x_train[rand_ind[start:end],:,:,:],y_train[rand_ind[start:end],:]
     
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
         
    def main_train_and_test(sel,x_train,y_train):
        import time
        s=time.time()
        loss_train=sel.backward(x_train,y_train)
        e=time.time()
        print('Running time:{}\n'.format(e-s))
        
        return loss_train
if __name__ == '__main__':

     import lc_CNN_backward_FC as backward
     sel=backward.CNN_backward()

     



