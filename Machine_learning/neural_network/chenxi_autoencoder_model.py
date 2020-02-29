#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:49:48 2018

@author: cici
"""

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import optimizers

model = Sequential()    
#model.add(Convolution3D(nb_filters[0],nb_depth=nb_conv[0], nb_row=nb_conv[0], 
#nb_col=nb_conv[0], input_shape=(1, img_rows, img_cols, patch_size), activation='relu'))
model.add(Convolution3D(64,(8,8,8), input_shape=(61,73,61,1), activation='relu', padding="same",name="Conv_1"))
model.add(MaxPooling3D(pool_size=(2, 2, 2),name="Pooling_1"))

model.add(Convolution3D(128,(7,7,7), activation='relu', padding="same",name="Conv_2"))
model.add(MaxPooling3D(pool_size=(2, 2, 2),name="Pooling_2"))


model.add(Convolution3D(256,(5,5,5), activation='relu', padding="same",name="Conv_3"))
model.add(MaxPooling3D(pool_size=(2, 2, 2),name="Pooling_3"))


model.add(Flatten())
model.add(Dense(1024,kernel_initializer="normal", activation='relu',name="Dense_1"))
model.add(Dropout(0.2))
model.add(Dense(256,kernel_initializer="normal", activation='relu',name="Dense_2"))
model.add(Dropout(0.2))
model.add(Dense(32,kernel_initializer="normal", activation='relu',name="Dense_3"))
model.add(Dropout(0.2))
model.add(Dense(2, kernel_initializer="normal", activation='softmax',name="Output"))
model.summary()


#adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sgd = optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
#model.compile(loss='mse', optimizer=adam,metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_train,y_train,batch_size=10,
                          epochs=1200,verbose=2,validation_split=0.05)

'''
plot picture
'''
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
    
    
#def show_train_history(train_history,train):
#    plt.plot(train_history.history[train])
#    #plt.plot(train_history.history[validation])
#    plt.title('Train History')
#    plt.ylabel(train)
#    plt.xlabel('Epoch')
#    plt.legend(['train'],loc='upper left')
#    plt.show()
    
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
#show_train_history(train_history,'acc')
#show_train_history(train_history,'loss')
'''
test scores
'''
scores = model.evaluate(x_test,y_test)
print('accuracy=',scores[1])

prediction = model.predict_classes(x_test)

predict_value = model.predict(x_test)

import pandas as pd
pd.crosstab(test_true_label,prediction,rownames=['label'],colnames=['predict'])