from PIL import Image
import numpy as np  # 전처리해야해
from tensorflow.keras.models import load_model


img_path = '../datasets/cat_dog/test_img/test02.jpg'
model_path = './cat_and_dog_0.836.h5'

model = load_model(model_path)

img = Image.open(img_path)
img = img.convert('RGB')
img = img.resize((64,64))
data = np.asarray(img)
data = data /255
data = data.reshape(1,64,64,3)
#pred= self.model.predict(data)
print(pred)

# job01에서 categories = ['cat', 'dog'] 를 보면,  0에 가까우면 cat, 1에 가까우면 dog
# anaconda를 설치하면 가상환경을 관리함.
# 큐티: GI 개발 플랫폼이다.