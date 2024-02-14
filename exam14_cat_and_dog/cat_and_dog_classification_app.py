import sys
from PyQt5.QtWidgets import *
# from PyQt5.QtWidgets import *: PyQt5의 위젯 관련 클래스들을 모두 가져옵니다.
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# PyQt5에서 제공하는 uic 모듈을 가져옵니다.
# uic 모듈은 Qt Designer에서 디자인한 UI 파일을 로드하는 데 사용됩니다.
form_window = uic.loadUiType('./cat_and_dog.ui')[0]
#  loadUiType 함수의 반환값은 튜플이므로 [0]을 사용하여 첫 번째 요소만 가져옵니다.
# 튜플 : 여러 값을 모아놓은 컨테이너로, 각 값은 쉼표(,)로 구분되며 괄호 ()로 둘러싸여 있습니다.
class Exam(QWidget, form_window): #  QWidget, form_window 이렇게 두 개의 클래스를 상속해.
    def __init__(self):
        super().__init__()
        self.setupUi(self) # 클래스만 만들어짐.
        self.path = ('C:/pythoncharm/pythonProject_Ai/datasets/cat_dog/test_img')
        self.model =load_model('./cat_and_dog_0.836.h5')
        self.btn_open.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(self,'Open file',
                                                '../datasets/cat_dog',
                                                'Image Files(*.jpg;*.png);;text Files(*.txt);;All Files(*.*)')
        print(self.path)
        if self.path[0] =='':
            self.path == old_path
        try:
            pixmap = QPixmap(self.path[0])
            self.lbl_image.setPixmap(pixmap)

            print(self.path)
            img = Image.open(self.path[0])
            img = img.convert('RGB')
            img = img.resize((64, 64))
            data = np.asarray(img)
            data = data/255
            data=data.reshape(1,64,64,3)
            pred = self.model.predict(data)
            print(pred)

            if pred < 0.5:
                self.lbl_result.setText('고양이 입니다.')
            else:
                self.lbl_result.setText('강아지 입니다.')
        except:
            self.lbl_result.setText('오류')


if __name__ == '__main__': #이름이 main 이면 실행하라.
    app = QApplication(sys.argv)
    #  PyQt5 애플리케이션을 생성합니다.
    mainWindow = Exam()
    # Exam 클래스의 인스턴스를 생성합니다.
    mainWindow.show()
    # GUI를 표시합니다.
    sys.exit(app.exec_()) # app 이 시그널을 받아.
    #애플리케이션 이벤트 루프를 시작하고, 애플리케이션이 종료될 때까지 대기합니다.
    # app.exec_(): PyQt5 애플리케이션의 이벤트 루프를 실행하는 메서드입니다.
    # sys.exit()를 호출하여 시스템을 종료합니다.


