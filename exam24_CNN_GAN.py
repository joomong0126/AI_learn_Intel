#경쟁력으로 학습을 하면서
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

OUT_DIR = './CNN_out'
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
generator.add(Dense(256*7*7, input_dim=noise))
generator.add(Reshape((7,7,256)))
generator.add(Conv2DTranspose(128,kernel_size=3,strides=2,padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.01))
generator.add(Conv2DTranspose(64,kernel_size=3,strides=1,padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.01))
generator.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))
generator.add(Activation('tanh'))
generator.summary()

# 중간중간마다 BatchNormalization()를 넣어줌.
# Batch Normalization은 신경망의 안정성과 학습 속도를 향상시키는 데 도움을 주는 regularization 기법 중 하나입니다. 이것은 특히 생성 모델에서 더욱 중요합니다. Batch Normalization을 사용하는 주요 이유는 다음과 같습니다:
#
# 안정성 향상:
#
# Batch Normalization은 각 미니배치의 입력을 정규화하여 안정성을 향상시킵니다. 이는 모델이 안정적으로 수렴하고 더 적은 학습 에폭으로 좋은 성능을 얻을 수 있도록 도와줍니다.
# 학습 속도 향상:
#
# 미니배치마다 정규화를 수행하므로, 각 레이어에 대한 입력 분포가 일정하게 유지됩니다. 이는 학습 속도를 향상시키고, 적은 학습률로도 안정적인 학습이 가능하도록 만듭니다.
# 초기화에 덜 민감:
#
# Batch Normalization은 가중치 초기화에 상대적으로 덜 민감합니다. 이는 학습 초기 단계에서 가중치 초기화에 대한 주의를 줄여줍니다.
# Regularization 효과:
#
# Batch Normalization은 일종의 내부적인 정규화 효과를 제공하여 과적합을 줄이는 역할을 합니다.
# 좀 더 자유로운 학습:
#
# 모델이 Batch Normalization을 통해 더 유연하게 학습할 수 있게 됩니다. Learning Rate를 크게 설정해도 안정적으로 학습이 가능하며, 다양한 크기의 미니배치에 대응할 수 있습니다.
# 상기 이유들로 인해 Batch Normalization은 생성 모델에서 특히 중요하며, 안정적인 학습과 높은 품질의 이미지 생성을 돕기 위해 Generator 모델의 각 Conv2DTranspose 레이어 뒤에 배치 정규화가 사용되는 것입니다.

# 이진분류기
lrelu = LeakyReLU(alpha=0.01)
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3,strides=2,padding='same',input_shape=img_shape))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Conv2D(64, kernel_size=3,strides=2,padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Conv2D(128, kernel_size=3,strides=2,padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()


#discriminator.add(Dense(1,activation='sigmoid')) # 이진분류기 sigmoid, 다중분류기 softmax
discriminator.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
discriminator.trainable=False

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()

gan_model.compile(loss='binary_crossentropy',optimizer='adam')
# ganmodel 컴파일을 할 때 학습을 안하는 걸 할려면


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



















