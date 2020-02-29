# -*- coding: utf-8 -*-
"""
"""
#==============================================================================
from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
import lc_autoencoder_forward
import tensorflow as tf
import time
import os

import numpy as np
import matplotlib.pyplot as plt
#==============================================================================
BATCH_SIZE = 200
STEPS = 3000

LEARNING_RATE_BASE = 0.1#指数衰减学习率基数
LEARNING_RATE_DECAY = 0.99#指数衰减学习率延迟系数
REGULARIZER = 0.0001#正则系数
MOVING_AVERAGE_DECAY = 0.99#滑动平均

# 及时保存模型（断点再续）
MODEL_SAVE_PATH=r"D:\My_Codes\LC_Machine_Learning\Machine_learning (Python)\Machine_learning\neural_network\Model"
MODEL_NAME="mnist_model"
display_step=1000

#==============================================================================    
def backward(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, lc_autoencoder_forward.NUM_INPUT])
        y_ =x
        _,y = lc_autoencoder_forward.forward(x, None)
        
        # global计数器
        global_step = tf.Variable(0, trainable=False)
        
    #    #交叉熵损失
#        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#        cem = tf.reduce_mean(ce)
#        loss = cem + tf.add_n(tf.get_collection('losses'))#正则
        
        # minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_ - y, 2))
        
        # 指数衰减学习率
        learning_rate = tf.train.exponential_decay(
                                                    LEARNING_RATE_BASE,
                                                    global_step,
                                                    mnist.train.num_examples / BATCH_SIZE, 
                                                    LEARNING_RATE_DECAY,
                                                    staircase=True)
        
        # 定义训练目标/优化函数
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        # 所有训练参数的滑动平均/优化函数的滑动平均转化
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')
            
        # 实例化保存器
        saver = tf.train.Saver()
        
        # 会话
        with tf.Session() as sess:
                                
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            # if there is trained model, then continue training it... 
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
    
            for i in range(STEPS):
                # data of current batch
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: xs})
                #每一千轮打印，并保存一次模型
                if i % display_step == 0:
                    print('Step %i: Minibatch Loss: %f' % (step, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    
            plt.figure()
            plt.imshow(xs[0,:].reshape(28,28), origin="upper", cmap="gray")
            plt.title('原始')
            
            x_pred=sess.run(y,feed_dict={x:xs})
            plt.figure()
            plt.imshow(x_pred[0,:].reshape(28,28), origin="upper", cmap="gray")
            plt.title('预测')

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    s=time.time()
    backward(mnist)
    e=time.time()
    print('Running time:{}\n'.format(e-s))

if __name__ == '__main__':
    main()