# -*- coding: utf-8 -*-
"""exam12_classification_fashion_mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kq4Kq-ZwtpWRZJP7BgHaacqyQQ0g1rJf
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras import datasets

(X_train, Y_train), (X_test, Y_test) = datasets.fashion_mnist.load_data()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

label = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat',
         'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

my_sample = np.random.randint(60000)
plt.imshow(X_train[my_sample], cmap='gray')
plt.show()
print(Y_train[my_sample])
print(X_train[my_sample])

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)
print(Y_train[5000])
print(y_train[5000])

x_train = X_train.reshape(-1, 28 * 28)
x_test = X_test.reshape(-1, 28 * 28)
x_train = x_train / 255
x_test = x_test / 255
print(x_train.shape)

model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

opt = Adam(learning_rate=0.01)
model.compile(opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
fit_hist = model.fit(x_train, y_train, batch_size=256,
          epochs=15, validation_split=0.2, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('accuracy', score[1])

plt.plot(fit_hist.history['accuracy'])
plt.plot(fit_hist.history['val_accuracy'])
plt.show()

my_sample = np.random.randint(10000)
plt.imshow(X_test[my_sample], cmap='gray')
print(label[Y_test[my_sample]])
pred = model.predict(x_test[my_sample].reshape(-1, 784))
print(pred)
print(label[np.argmax(pred)])

