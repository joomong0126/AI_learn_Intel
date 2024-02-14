import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

# 입력 이미지의 크기를 정의
input_img = Input(shape=(784,))
# 인코더 부분 정의
encoded = Dense(32, activation='relu')
encoded = encoded(input_img)
# 디코더 부분 정의
decoded = Dense(784, activation='sigmoid')
decoded = decoded(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()
# 오토인코더 모델 컴파일
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# MNIST 데이터셋 로드 및 전처리
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255
# 2D 이미지를 1D로 펼치기
flatted_x_train = x_train.reshape(-1, 28 * 28)
flatted_x_test = x_test.reshape(-1, 28 * 28)
print(flatted_x_train.shape)
print(flatted_x_test.shape)
# 모델 학습
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train,
               epochs=50, batch_size=256,
               validation_data=(flatted_x_test, flatted_x_test))
# 테스트 데이터에 대한 이미지 압축 및 재구성
encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)
# 결과 시각화
n = 10
plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# 학습 과정 시각화
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

# 모델 구조 정의:
#
# 입력 이미지는 784개의 픽셀로 구성된 1D 벡터로 펼쳐진 이미지입니다.
# 인코더는 32개의 뉴런으로 구성된 은닉층을 가집니다.
# 디코더는 원래의 784개의 픽셀을 복원하는 역할을 합니다.
# 모델 컴파일:
#
# 이진 교차 엔트로피 손실 함수를 사용하고 Adam 옵티마이저를 사용하여 오토인코더 모델을 컴파일합니다.
# 데이터 전처리:
#
# MNIST 데이터를 로드하고, 픽셀 값을 0에서 1 사이로 정규화하며 2D 이미지를 1D 벡터로 변환합니다.
# 모델 학습:
#
# 오토인코더를 플래튼한 데이터에 대해 50 에포크 동안 학습시킵니다.
# 인코더 및 디코더 모델 정의:
#
# 인코더 모델은 입력 이미지를 압축하여 특징을 추출하는 역할을 합니다.
# 디코더 모델은 압축된 이미지를 받아 다시 원래 차원으로 복원하는 역할을 합니다.
# 테스트 데이터에 대한 이미지 압축 및 재구성:
#
# 테스트 데이터 중 일부를 사용하여 인코더로 이미지를 압축하고, 디코더로 압축된 이미지를 재구성합니다.
# 결과 시각화:
#
# 원본 이미지와 재구성된 이미지를 비교하여 시각화합니다.
# 학습 과정 시각화:
#
# 학습 중 손실 값의 변화를 시각화하여 모델의 학습 과정을 확인합니다.
# 실생활 적용:
# 이 코드는 이미지의 차원을 압축하고 복원하는 간단한 오토인코더를 보여줍니다. 이러한 오토인코더를 사용하면 데이터의 중요한 특징을 추출하고, 이를 통해 차원 축소나 특징 추출과 같은 작업을 수행할 수 있습니다. 예를 들어, 고객의 구매 이력 데이터에서 중요한 특징을 추출하거나, 센서 데이터에서 이상 징후를 탐지하는 데 활용될 수 있습니다.





