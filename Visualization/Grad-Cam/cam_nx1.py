from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.utils import np_utils
from keras.layers import Dense, Activation,Dropout,Convolution1D,MaxPooling1D,Flatten
from keras.layers.normalization import BatchNormalization
import numpy as np

x_train = np.load('datas/X.npy')
y_train = np.load('datas/y.npy')
x_test = np.load('datas/Xs.npy')
y_test = np.load('datas/ys.npy')

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

#################modeling#######################
# 建立序贯模型
model = Sequential()                                           #256*16

model.add(Convolution1D(                                       #256*32
    filters=64,
    kernel_size=2,
    padding='same',
    strides=1,
    input_shape=(256,32))) 

model.add(MaxPooling1D(                                       #128*32
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(BatchNormalization())

model.add(Convolution1D(                                       #128*64
    filters=128,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #64*64
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(BatchNormalization())

model.add(Convolution1D(                                      #64*128
    filters=256,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #32*128
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(BatchNormalization())

'''
model.add(Convolution1D(                                      #32*256
    filters=512,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #16*256
    pool_size=2,
    strides=2,
    padding='same')) 

model.add(Convolution1D(                                      #16*512
    filters=1024,
    kernel_size=2,
    padding='same',
    strides=1)) 

model.add(MaxPooling1D(                                       #8*512
    pool_size=2,
    strides=2,
    padding='same')) 
'''

model.add(Flatten())                                           #4096

model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=2))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()

# 配置模型的学习过程
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#######################training###################
model.fit(x_train,y_train,batch_size=32,epochs=1000)

model.save('model.h5')
#######################evaluate###################
score=model.evaluate(x_test,y_test)
print('Test accuracy:', score[1])
