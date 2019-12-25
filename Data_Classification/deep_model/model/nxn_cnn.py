from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.utils import np_utils
from keras.layers import Dense, Activation,Dropout,Convolution2D,MaxPooling2D,Flatten
import numpy as np

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_vld = np.load("x_vld.npy")
y_vld = np.load("y_vld.npy")

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# trans_data
# pass

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

# 建立序贯模型
model = Sequential()                                          

model.add(Convolution2D(                                       #64*64*25
    filters=25,
    kernel_size=(3,3),
    padding='same',
    input_shape=(64,64,1)))

model.add(MaxPooling2D(                                       #32*32*25
    pool_size=(2,2),
    strides=2)) 

model.add(Convolution2D(                                       #32*32*50
    filters=50,
    kernel_size=(3,3),
    padding='same')) 

model.add(MaxPooling2D(                                       #16*16*50
    pool_size=(2,2),
    strides=2))

model.add(Convolution2D(                                      #16*16*100
    filters=100,
    kernel_size=(3,3),
    padding='same'))

model.add(MaxPooling2D(                                       #8*8*100
    pool_size=(2,2),
    strides=2)) 

model.add(Convolution2D(                                      #8*8*200
    filters=100,
    kernel_size=(3,3),
    padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))                      #4*4*400


model.add(Flatten())                                           #6400

model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=2))
model.add(Activation('softmax'))

model.summary()


adam = optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#######################training###################
model.fit(x_train,y_train,batch_size=32,epochs=1000)

#######################evaluate###################
score=model.evaluate(x_test,y_test)
print('Test accuracy:', score[1])
