import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./poster.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_elly.clicked.connect(self.btn_click_slot)
        self.btn_winter.clicked.connect(self.btn_click_slot)
        self.btn_rosy.clicked.connect(self.btn_click_slot)
        self.btn_out.clicked.connect(self.btn_click_slot)

    def btn_click_slot(self):
        btn = self.sender()
        self.lbl_elly.hide()
        self.lbl_winter.hide()
        self.lbl_rosy.hide()
        self.lbl_out.hide()
        if btn.objectName() == 'btn_elly':self.lbl_elly.show()
        elif btn.objectName() == 'btn_winter':self.lbl_winter.show()
        elif btn.objectName() == 'btn_rosy':self.lbl_rosy.show()
        elif btn.objectName() == 'btn_out':self.lbl_out.show()


    # def btn_winter_slot(self):
    #     self.lbl_elly.hide()
    #     self.lbl_winter.hide()
    #     self.lbl_rosy.hide()
    #     self.lbl_out.hide()
    #     self.lbl_winter.show() 이런식으로 일일이 나열해야하는 것을 16~25와 같은 과정을 통해서 한 번에 처리함.



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())



