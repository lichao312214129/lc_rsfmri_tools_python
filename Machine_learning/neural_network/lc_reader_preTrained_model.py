# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:55:41 2018

@author: lenovo
"""
import tensorflow as tf
#import numpy as np
#from tensorflow.python import pywrap_tensorflow
import os
#
#
#checkpoint_path =r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\model_mean18'
#reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
#
#for key in var_to_shape_map:
#    print("tensor_name: ", key)
#    
#    w=reader.get_tensor(key)
#    w=np.squeeze(w)




model_path=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\model_static'

saver =tf.train.import_meta_graph(os.path.join(model_path,"model.ckpt-24609.meta")) 

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init) 

    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path) 
#    saver.restore(sess, os.path.join(model_path,"model.ckpt"))
    # 通过张量的名称来获取张量
    fc1=sess.run(tf.get_default_graph().get_tensor_by_name("fc1/filter:0"))[0,0]
    a=[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    print(fc1)
