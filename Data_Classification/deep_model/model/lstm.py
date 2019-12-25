from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras
import numpy as np


X_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
X_vld = np.load("x_vld.npy")
y_vld = np.load("y_vld.npy")

X_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

maxword = 4000
X_train = sequence.pad_sequences(X_train, maxlen = maxword)
X_test = sequence.pad_sequences(X_test, maxlen = maxword)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1


model = Sequential()
model.add(Embedding(vocab_size, 64, input_length = maxword))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics =['accuracy'])
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5,batch_size = 100)
scores = model.evaluate(X_test, y_test)
print(scores)
