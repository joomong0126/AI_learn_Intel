import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = np.load('C:/pythoncharm/pythonProject_Ai/datasets/binary_image_data1.npy',allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Conv2D(32,input_shape=(64,64,3),kernel_size=(3,3),padding='same',activation='relu'))
# Conv2D 이미지처리하는 기법
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
early_stopping = EarlyStopping(monitor='val_binary_accuracy',patience=7)
# EarlyStopping: Keras에서 제공하는 콜백(callback) 중 하나로, 모델 훈련 중에 지정된 조건이 충족되면 훈련을 조기에 중지시키는 역할
# monitor: 어떤 지표를 모니터링할지 지정합니다. 주로 검증 데이터에 대한 지표를 선택합니다.
# patience: 지정된 지표가 개선되지 않은 상태로 몇 번의 에폭(epoch)을 지나면 훈련을 중지할지를 결정합니다.
#  예를 들어, patience=7은 7번의 에폭 동안 지정된 지표가 향상되지 않으면 훈련을 중지합니다.

fit_hist = model.fit(X_train, Y_train, batch_size=128,epochs=100,validation_split=0.15,callbacks=[early_stopping])
score = model.evaluate(X_test,Y_test)
print('Evaluation loss:',score[0])
print('Evaluation accuracy:',score[1])
model.save('./cat_and_dog_{}.h5'.format(str(np.around(score[1],3))))

plt.plot(fit_hist.history['binary_accuracy'],label='binary_accuracy')
plt.plot(fit_hist.history['val_binary_accuracy'],label='val_binary_accuracy')
plt.legend()
plt.show()
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

