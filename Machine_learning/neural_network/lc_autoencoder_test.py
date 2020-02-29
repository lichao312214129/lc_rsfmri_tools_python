# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:16:23 2018

@author: lenovo
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lc_autoencoder_forward
import lc_autoencoder_backward

import numpy as np
import matplotlib.pyplot as plt

REGULARIZER = 0.0001#正则系数
TEST_INTERVAL_SECS = 5

def test(mnist):
    
   # 复现已有的模型
    graph_app = tf.Graph()
    with graph_app.as_default():  
        x = tf.placeholder(tf.float32, [None, lc_autoencoder_forward.NUM_INPUT])
        y_true = x
        y_low_dimension,y_pred = lc_autoencoder_forward.forward(x, None)
        
        # 滑动平均
        ema = tf.train.ExponentialMovingAverage(lc_autoencoder_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
		
        # 测试集效果
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#        while True:
        with tf.Session(graph=graph_app) as sess:
            ckpt = tf.train.get_checkpoint_state(lc_autoencoder_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               
                test_loss  = sess.run(loss, feed_dict={x: mnist.test.images, y_true: mnist.test.images})
                print("After %s training step(s), test accuracy = %g" % (global_step, test_loss))
              
            else:
                print('No checkpoint file found')
                return
            
            
            # low dimension
            y_low_dim=sess.run(y_low_dimension, feed_dict={x: mnist.test.images})
            
            
            # show image
            batch_x,_=mnist.test.next_batch(20)
            
            fig, axes = plt.subplots(5, 4) 
            fig.suptitle('orignal')
            for i,ax  in enumerate(axes.ravel()):
                ax.imshow(batch_x[i,:].reshape(28,28), origin="upper", cmap="gray")
                ax.axis('off')
                plt.show()
            
            y_predict = sess.run(y_pred, feed_dict={x: batch_x})
            fig, axes = plt.subplots(5, 4) 
            fig.suptitle('predict')
            for i,ax  in enumerate(axes.ravel()):
                ax.imshow(y_predict[i,:].reshape(28,28), origin="upper", cmap="gray")
                ax.axis('off')
                plt.show()
                
#        time.sleep(TEST_INTERVAL_SECS)
            
            return y_low_dim

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    y_low_dim=test(mnist)
    return y_low_dim

if __name__ == '__main__':
    y_low_dim=main()

