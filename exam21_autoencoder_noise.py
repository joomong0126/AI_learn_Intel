# 노이즈를 제거하기 위해 프로그램을 돌려서 노이즈를 제거한 후, white noise를 섞어서 다시 복원했을 때도 노이즈가 없어질지 확인하기 위한 코드

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

autoencoder = load_model('./models/autoencoder.h5')
(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train / 255
x_test = x_test/ 255
conv_x_train = x_train.reshape(-1,28,28,1)
conv_x_test = x_test.reshape(-1,28,28,1)
print(conv_x_train.shape,conv_x_test.shape)

noise_factor = 0.5
conv_x_test_noisy = conv_x_test + np.random.normal(
    loc = 0.0,scale = 1.0,size=conv_x_test.shape) * noise_factor
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0,1.0)

decoded_img = autoencoder.predict(conv_x_test_noisy[:10])

n= 10
plt.gray()
for i in range(n):
    ax = plt.subplot(3,10,i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, 10, i + 1+n)
    plt.imshow(conv_x_test_noisy[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, 10, i + 2*n)
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 잡음을 섞었는 데도 안돼. 잡음은 특성이 없는 데도 안돼. 그런데 잘 안됨.

