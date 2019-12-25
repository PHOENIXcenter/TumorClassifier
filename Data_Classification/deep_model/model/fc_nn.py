from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam,SGD
import numpy as np
import os

X_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
X_vld = np.load("x_vld.npy")
y_vld = np.load("y_vld.npy")

X_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

model = Sequential()
model.add(Dense(input_dim=4096, units=1024, activation='relu')) 
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=1, activation='softmax'))  #建立输出层
 
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)  # 设定学习效率等参数
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1)

result_test = model.evaluate(x_test, y_test, batch_size=164)
print('\nTest Acc:', result_test[1])
