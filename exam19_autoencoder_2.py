import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

# 입력 이미지의 크기를 정의
input_img = Input(shape=(784,))

# 인코더 부분 정의
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded) # 입력일 때는 relu

# 디코더 부분 정의
decoded = Dense(64, activation='sigmoid')(encoded) # 출력일 때는 sigmoid
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# 오토인코더 모델 정의
autoencoder = Model(input_img, decoded)
autoencoder.summary()

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

# 테스트 데이터에 대한 이미지 재구성
decoded_img = autoencoder.predict(flatted_x_test[:10])

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
# 오토인코더는 3개의 은닉층을 가진 간단한 구조로, 입력을 점차적으로 압축하고 다시 풀어내는 역할을 합니다.
# 오토인코더 모델 컴파일:
#
# 이진 교차 엔트로피 손실 함수를 사용하고 Adam 옵티마이저를 사용하여 모델을 컴파일합니다.
# 데이터 전처리:
#
# MNIST 데이터를 로드하고, 픽셀 값을 0에서 1 사이로 정규화하며 2D 이미지를 1D 벡터로 변환합니다.
# 모델 학습:
#
# 오토인코더를 플래튼한 데이터에 대해 50 에포크 동안 학습시킵니다.
# 결과 시각화:
#
# 학습된 오토인코더를 사용하여 테스트 데이터의 이미지를 재구성하고, 원본 이미지와 재구성된 이미지를 시각화합니다.
# 학습 과정 시각화:
#
# 학습 중 손실 값의 변화를 시각화하여 모델의 학습 과정을 확인합니다.
# 실생활 적용:
# 이 코드는 이미지 압축 및 재구성에 사용되는 간단한 오토인코더를 보여줍니다. 실생활에서는 이러한 오토인코더를 이용하여 데이터의 특징 추출, 노이즈 제거, 차원 축소 등 다양한 응용이 가능합니다. 예를 들어, 의료 영상에서 병변을 감지하거나, 고객 리뷰 데이터에서 유용한 정보를 추출하는 등의 응용이 가능합니다.




