# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:13:43 2018
以功能连接为特征的卷积神经网络:前向传播
<Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift>
@author: Li Chao
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
#==============================================================================
# input
MAT_SIZE_ROW=114
MAT_SIZE_COL=114
NUM_CHANNELS = 1

# conv1
CONV1_SIZE_ROW = 1
CONV1_SIZE_COL = 114
CONV1_KERNEL_NUM = 64 # 64

# conv2
CONV2_SIZE_ROW = 114
CONV2_SIZE_COL = 1
CONV2_KERNEL_NUM = 256 # 128

# fc1
FC_SIZE = 100

#fc2
OUTPUT_NODE = 2

KEEP_PROB=0.6
#==============================================================================


def get_weight(name,shape, regularizer):
    
#    w=tf.get_variable(shape=shape,initializer=tf.contrib.layers.xavier_initializer(),name='filter') # for sigmoid and tanh
#    w=tf.get_variable(shape=shape,initializer=tf.contrib.layers.variance_scaling_initializer(),name='filter') # for relu
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 
    return w

def get_bias(shape): 
    b = tf.Variable(tf.zeros(shape)+0.001)  
    return b

def conv2d(x,w,how):  
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=how)

def forward(x, train, regularizer):
    # The first conv
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE): 
        conv1_w = get_weight('conv1',[CONV1_SIZE_ROW, CONV1_SIZE_COL, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer) 
    
    conv1_b = get_bias([CONV1_KERNEL_NUM]) 
    conv1 = conv2d(x, conv1_w,'VALID') 
    
    # batch normaliztion
    conv1=batch_norm(conv1,decay=0.999,is_training=train,updates_collections=None)
    
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) 

    
    # The second conv
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE): 
        conv2_w = get_weight('conv2',[CONV2_SIZE_ROW, CONV2_SIZE_COL, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer)
        
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(relu1, conv2_w,'VALID') 
    
    # batch normaliztion
    conv2=batch_norm(conv2,decay=0.999,is_training=train,updates_collections=None)
    
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    
    # the first FC
    pool_shape = relu2.get_shape().as_list() 
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
    try:
        reshaped = tf.reshape(relu2, [pool_shape[0], nodes]) 
    except TypeError:
        reshaped = tf.reshape(relu2, [-1, nodes]) 
    
    with tf.variable_scope('fc1',reuse=tf.AUTO_REUSE): 
        fc1_w = get_weight('fc1',[nodes, FC_SIZE], regularizer) 
        
    fc1_b = get_bias([FC_SIZE])
    fc1=tf.matmul(reshaped, fc1_w) + fc1_b
    
    # batch normaliztion
    fc1=batch_norm(fc1,decay=0.999,is_training=train,updates_collections=None)
    
    fc1 = tf.nn.relu(fc1) 
    if train: fc1=tf.nn.dropout(fc1 ,KEEP_PROB) # dropout
    
    # the second FC
    with tf.variable_scope('fc2',reuse=tf.AUTO_REUSE): 
        fc2_w = get_weight('fc2',[FC_SIZE, OUTPUT_NODE], regularizer)
        
    fc2_b = get_bias([OUTPUT_NODE]) 
    y = tf.matmul(fc1, fc2_w) + fc2_b # do not go through the softmax or others now!
    return y 