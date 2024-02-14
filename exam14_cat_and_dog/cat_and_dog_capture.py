import sys
from PIL import Image
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np
from tensorflow.keras.models import load_model
import cv2 # open cv
import time

form_window = uic.loadUiType('./cat_and_dog.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = load_model('.\cat_and_dog_0.836.h5')

        self.btn_open.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        capture = cv2.VideoCapture(0)  # capture 가 카메라인거야. 웹캠이 있으면 웹캠이 우선순위야. 그래서 (0) 번을 주면 내장 웹캠을 우선순위로 잡음.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
        flag = True
        while flag:
            v, frame = capture.read() # read를 하면 한장의 이미지가 들어옴. read()를 하면 동영상을 읽는 게 아니라, 사진 한장을 읽는 거야.
            print(type(frame))

            if (v):
                cv2.imshow('VideoFrame', frame) # image show를 안해도 상관없음. 이미 이미지를 가져온거라서.
                cv2.imwrite('./capture.png', frame)

            # time.sleep(0.01) # 30분의 1초마다 사진을 찍고 저장하려면 time.sleep(0.03)  # 0.5 하면 1초에 2장씩 받을 수 있어. 그러면 동영상이 끊어질 수 있음.
            key = cv2.waitKey(30) # 1초에 30장 정도 되도록
            if key == 27: # 아스키코드값으로 하면 ESC키 값이야.
                flag = False  # 30번부터 36번까지 파일 경로 찾아 오는 것

            pixmap = QPixmap('./capture.png')
            self.lbl_image.setPixmap(pixmap)

            try:
                img = Image.open('./capture.png')
                img = img.convert('RGB')
                img = img.resize((64, 64))
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1, 64, 64, 3)

                pred = self.model.predict(data)
                print(pred)
                if pred < 0.5:
                    self.lbl_result.setText('고양이입니다.')
                else:
                    self.lbl_result.setText('강아지입니다.')
            except:
                print('error ')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())
