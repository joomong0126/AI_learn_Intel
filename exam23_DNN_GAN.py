# 입력-은닉-출력

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

OUT_DIR = './DNN_out'
img_shape = (28,28,1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

MY_NUMBER = 3

# 오토엩코더의 디코더부분만 만들거에요
# 이진분류기를 이용해서 글씨를 판별하게끔. ex) 조금이라도 비슷하면 손글씨라고 판단하게끔 만들어야해. 안그러면 조금이라도 틀리면 이거는 잡음으로 판단을 해. 그래서 너무 정확하게 판단하게끔 만들면 안돼
# 생성모델이 점점 성능이 좋아지면서 이진분류기의 성능도 점점 좋아져야해.
# 값모델이 2개가 있어야해. 이 2개의 모델이 서로 경쟁하면서 성능이 좋아짐.

(x_train, y_train),(_, _) = mnist.load_data()
x_train = x_train[y_train == MY_NUMBER]
print(x_train.shape)

x_train = x_train / 127.5 - 1   ## 이런경우 음수값이 있을 때 leakyRelu를 사용해
x_train = np.expand_dims(x_train, axis=3) # axis=3 이말은 3차원짜리 데이터 6만개 !
print(x_train.shape)

generator = Sequential()
generator.add(Dense(128, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01))
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape(img_shape))
generator.summary()

# LeakyReLU(Leaky Rectified Linear Unit)는 일반적인 ReLU(Rectified Linear Unit) 함수의 변형 중 하나입니다. LeakyReLU는 ReLU의 변종 중 하나로, 주로 생성 모델에서 사용됩니다. LeakyReLU는 다음과 같은 특징을 가지고 있습니다.
#
# 음수 값에 대한 처리:
#
# LeakyReLU는 일반적인 ReLU와는 달리 음수 값에 대해 완전히 0이 아닌 작은 음수 값을 반환합니다. 즉, 입력이 음수일 때도 작은 기울기를 가지므로, 항상 양수의 출력을 생성합니다.
# 그래디언트 소실 문제 완화:
#
# 일반적인 ReLU는 입력이 음수인 경우 그래디언트가 0이 되어 그 이후의 역전파에서 가중치 업데이트가 이루어지지 않는 "그래디언트 소실" 문제를 가질 수 있습니다. LeakyReLU는 이를 완화하여 일부 음수 값에 대해서도 그래디언트를 전달합니다.
# 생성 모델에서의 활용:
#
# 생성 모델에서는 LeakyReLU가 학습의 안정성을 향상시킬 수 있습니다. 특히, 생성자(generator) 부분에서 사용되는 경우, LeakyReLU를 통해 모델이 다양한 특징을 학습하고 모드 붕괴(mode collapse)를 피할 수 있습니다.
# 따라서, 코드에서 생성자 모델의 첫 번째 레이어에 LeakyReLU를 사용하는 이유는 모델이 학습하는 동안 다양한 특징을 포착하고 그래디언트 소실 문제를 완화하기 위해서입니다. LeakyReLU의 alpha 매개변수는 음수 영역의 기울기를 조절하는 값으로, 작은 음수 값으로 설정하여 음수 영역에서도 정보를 유지하도록 합니다.


# 이진분류기
lrelu = LeakyReLU(alpha=0.01)
discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(128, activation=lrelu))
discriminator.add(Dense(1,activation='sigmoid')) # 이진분류기 sigmoid, 다중분류기 softmax
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()

gan_model.compile(loss='binary_crossentropy',optimizer='adam')
# ganmodel 컴파일을 할 때 학습을 안하는 걸 할려면
discriminator.trainable=False

real = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

# print(real)
# print(fake)


# 만약에 이미지가 쉬우면 discriminator이 앞지르게 됨. 그럴 경우 generator를 두번 학습시키고  discriminator을 한 번만 학습시키게끔 해야해.
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0],batch_size)
    real_img = x_train[idx]

    z= np.random.normal(0,1,(batch_size,noise))
    fake_img = generator.predict(z)

    # 가짜 이미지(잡음이미지) -> fake_img
    d_hist_real = discriminator.train_on_batch(real_img,real)
    d_hist_fake = discriminator.train_on_batch(fake_img, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    # generator를 한 번 해야해.
    z = np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z,real)

    # 학습이 오래걸릴테니까 print를 중간중간에 해보는 용
    if epoch % sample_interval == 0:
        print('%d [ D loss: %f, acc: %.2f%%] [G loss:%f]'%(
            epoch, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z =np.random.normal(0,1,(row*col,noise))
        fake_imgs= generator.predict(z)
        fake_imgs = 0.5 * fake_imgs
        _, axs = plt.subplots(row,col,figsize= (row,col),sharey=True,sharex=True)
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[count, :, :,0],cmap='gray')
                axs[i,j].axis('off')
                count += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1))
        plt.savefig(path)
        plt.close()
        generator.save('./models/generator_3_LGW.h5')




















