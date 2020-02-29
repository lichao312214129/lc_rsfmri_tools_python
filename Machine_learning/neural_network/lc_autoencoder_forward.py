# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:20:22 2018

@author: lenovo
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
#==============================================================================
NUM_HIDDEN_1 = 256 
NUM_HIDDEN_2 = 128 
NUM_INPUT = 784

#==============================================================================
def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b
	
def forward(x, regularizer):
    
    # encoder
    w1 = get_weight([NUM_INPUT, NUM_HIDDEN_1], regularizer)
    b1 = get_bias([NUM_HIDDEN_1])
    y1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1),b1))

    w2 = get_weight([NUM_HIDDEN_1, NUM_HIDDEN_2], regularizer)
    b2 = get_bias([NUM_HIDDEN_2])
    y_encoder = tf.nn.sigmoid(tf.add(tf.matmul(y1, w2),b2))
    
    # decoder
    w3 = get_weight([NUM_HIDDEN_2, NUM_HIDDEN_1], regularizer)
    b3 = get_bias([NUM_HIDDEN_1])
    y3 = tf.nn.sigmoid(tf.add(tf.matmul(y_encoder, w3),b3))
    
    w4 = get_weight([NUM_HIDDEN_1,NUM_INPUT], regularizer)
    b4 = get_bias([NUM_INPUT])
    y_decoder = tf.nn.sigmoid(tf.add(tf.matmul(y3, w4),b4))
    
    return y_encoder,y_decoder