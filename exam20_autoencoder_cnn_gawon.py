# 이 코드는 MNIST 데이터셋을 사용한 간단한 오토인코더 모델을 만들고, 학습 및 이미지 재구성을 시각화하는 예제입니다. 오토인코더를 사용하는 경우, 모델은 입력 데이터의 특징을 추출하고 재구성함으로써 데이터의 압축된 표현을 학습합니다.

# 모델 만드는 코드
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

#간단한 CNN(Convolutional Neural Network) 기반의 오토인코더 모델을 정의합니다. 인코더와 디코더 부분으로 나누어져 있습니다.
input_img = Input(shape=(28, 28, 1,))
# 인코더 부분
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)    # 28 x 28
x = MaxPool2D((2, 2), padding='same')(x)                                # 14 x 14
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # 14 x 14
x = MaxPool2D((2, 2), padding='same')(x)                                # 7 x 7
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # 7 x 7
encoded = MaxPool2D((2, 2), padding='same')(x)                          # 4 x 4   # 4X4는 아임쇼를 가지고 그려볼 수 있어.
# 디코더 부분
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)       # 4 x 4
x = UpSampling2D((2, 2))(x)                                             # 8 x 8 # upsampling과 Maxpool2D의 차이점 기억하기.
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             # 8 x 8
x = UpSampling2D((2, 2))(x)                                             # 16 x 16
x = Conv2D(16, (3, 3), activation='relu')(x)                            # 14 x 14
x = UpSampling2D((2, 2))(x)                                             # 28 x 28
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)       # 28 x 28


# 오토인코더 모델을 컴파일합니다. 손실 함수로는 이진 교차 엔트로피(binary crossentropy)를 사용하고, 옵티마이저로는 Adam을 선택합니다.
autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#데이터 전처리:
# MNIST 데이터를 로드하고, 0-255 범위의 픽셀 값을 0-1 범위로 정규화하고, Convolutional Neural Network에 입력으로 사용할 수 있도록 데이터를 형태를 변환합니다.
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

# 모델 학습:
fit_hist = autoencoder.fit(conv_x_train, conv_x_train,
               epochs=100, batch_size=256,
               validation_data=(conv_x_test, conv_x_test))

# 모델을 사용하여 이미지 재구성 및 시각화:
# 학습된 모델을 사용하여 테스트 데이터의 이미지를 재구성하고, 원본 이미지와 재구성된 이미지를 시각화합니다.
decoded_img = autoencoder.predict(conv_x_test[:10])
n = 10
plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 재구성된 이미지
    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 학습 과정 시각화:
#학습 과정의 손실을 시각화합니다.
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

#모델 저장:
autoencoder.save('./models/autoencoder.h5')




