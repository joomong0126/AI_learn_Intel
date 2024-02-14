import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./Qnotepad.ui')[0]

## 노트 패드를 이용해서 메모장을 구현하고 이를 저장하는 방법을 배움.

class Exam(QMainWindow, form_window): # QMainWindow 로 시작했으니.
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path=('제목없음','')
        self.edited_flag = False # flag을 쓰면 시스템 수정할 때 편리함.
        self.setWindowTitle('*'+self.path[0].split('/')[-1] + '- QT Note Pad') # 문자열이니까 더하기

        ##### file menu 구현
        self.actionSave_as.triggered.connect(self.action_save_as_slot)
        #  "Save As" 메뉴 옵션을 나타내는 객체(액션)입니다.
        # triggered 시그널은 사용자가 해당 액션을 실행했을 때 발생합니다.
        # connect 메서드를 사용하여 triggered 시그널이 발생할 때 self.action_save_as_slot 메서드를 호출하도록 연결합니다.
        self.actionSave.triggered.connect(self.action_save_slot)
        self.actionSave.triggered.connect(self.action_exit_slot)
        self.plain_te.textChanged.connect(self.text_changed_slot)
        self.actionOpen.triggered.connect(self.action_open_slot) # action을 취할 때는 triggered 를 사용해
        self.actionNew.triggered.connect(self.action_new_slot)

        ##### edit menu 구현
        self.actionUn_do.triggered.connect(self.plain_te.undo)
        self.actionCut.triggered.connect(self.plain_te.cut)
        self.actionCopy.triggered.connect(self.plain_te.copy)
        self.actionPaste.triggered.connect(self.plain_te.paste)
        self.actionDelete.triggered.connect(self.plain_te.cut) # delete 기능이 없어서 cut()을 사용함.
        self.actionSelect_all.triggered.connect(self.plain_te.selectAll)
        self.actionFont_2.triggered.connect(self.action_font_slot)
        self.actionAbout.triggered.connect(self.action_about_slot)


    def action_about_slot(self):
        QMessageBox.about(self,'Qt Note Pad','만든이:abc label\n\r 버전 정보:1.0.0')



    def action_font_slot(self):
        font = QFontDialog.getFont()
        print(font)
        if font[1]:
            self.plain_te.setFont(font[0])  # 0번이 폰트 객체를 의미


    def save_edited(self):
        if self.edited_flag:
            ans = QMessageBox.question(self, '저장하기', '저장할까요?', QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
                                       QMessageBox.Yes)
            #  보통 메모장을 닫을 때, '저장' 버튼에 포커싱이 가있기 때문에 그렇게 설정함.
            if ans == QMessageBox.Yes:  # 파일 저장
                if self.action_save_slot():
                    #  print('debug01') 문자 사이 사이에 디버그를 넣어서 코드가 어디까지 죽었는 지 알아보기 위함.
                    return
            elif ans == QMessageBox.Cancel:
                return 1

    def action_new_slot(self): # 제목없음이랑 같은 상태입니다.
        if self.save_edited():
            return
        self.plain_te.setPlainText('')
        self.edited_flag =False
        self.path = ('제목 없음','')
        self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad')


    def action_open_slot(self):
        if self.save_edited():
            return
        # if self.edited_flag:
        #     return
        # if self.edited_flag:
        #     ans = QMessageBox.question(self, '저장하기', '저장할까요?', QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
        #                                QMessageBox.Yes)
        #     #  보통 메모장을 닫을 때, '저장' 버튼에 포커싱이 가있기 때문에 그렇게 설정함.
        #     if ans == QMessageBox.Yes:  # 파일 저장
        #         if self.action_save_slot():
        #             #  print('debug01') 문자 사이사이에 디버그를 넣어서 코드가 어디까지 죽었는 지 알아보기 위함.
        #             return
        #     elif ans == QMessageBox.Cancel:
        #         return
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(self,'Open FIle','','Text Files(*.txt);;Python Files(*.py);;All Files(*.*)')
        if self.path[0]:
            with open(self.path[0],'r')as f:
                str_read = f.read()
            self.plain_te.setPlainText(str_read)
            self.edited_flag = False
            self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad')
        else:
            self.path = old_path


    def text_changed_slot(self):
        print('change') # text를 입력할 때마다 출력됨.
        self.edited_flag = True
        self.setWindowTitle('*' + self.path[0].split('/')[-1] + '- QT Note Pad')
        self.plain_te.textChanged.disconnect(self.text_changed_slot)


    def action_exit_slot(self):
        if self.save_edited():
            return

        if self.edited_flag:
            return

        # # 메모장 닫을 때, 저장하고 닫아야해.
        # # flag을 쓰면 시스템 수정할 때 편리함.
        # if self.edited_flag:
        #     ans = QMessageBox.question(self,'저장하기','저장할까요?',QMessageBox.No | QMessageBox.Cancel|QMessageBox.Yes,QMessageBox.Yes)
        #     #  보통 메모장을 닫을 때, '저장' 버튼에 포커싱이 가있기 때문에 그렇게 설정함.
        #     if ans == QMessageBox.Yes: # 파일 저장
        #         if self.action_save_slot():
        #            #  print('debug01') 문자 사이사이에 디버그를 넣어서 코드가 어디까지 죽었는 지 알아보기 위함.
        #             return
        #     elif ans == QMessageBox.Cancel: return
        #print('debug02')
        self.close()

    def action_save_slot(self):
        if self.path[0] != '제목 없음':
            with open(self.path[0],'w') as f:
                f.write(self.plain_te.toPlainText())
                self.edited_flag = False
                self.plain_te.textChanged_slot.connect(self.text_changed_slot)
                self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad')

        else: return self.action_save_as_slot() # '제목 없음' 일 경우 저장됨.

    def action_save_as_slot(self): # action_save_as_slot 메서드는 "Save As" 메뉴 옵션을 선택했을 때 호출되는 메서드입니다. ( 새로운 함수 지정, 후 동작 설정)
        old_path = self.path # 현재 파일의 경로를 old_path에 저장합니다. (  self.path는 현재 파일의 경로)
        self.path = QFileDialog.getSaveFileName(self,'Save file','','Text Files(*.txt);;Python Files(*.py);;All File(*.*)')
        print(self.path)
        if self.path[0]:
            with open(self.path[0], 'w') as f: # : 선택된 파일 경로를 이용하여 파일을 쓰기 모드('w')로 엽니다. 이 부분은 파일을 열고 해당 파일에 대한 파일 객체(f)를 생성합니다.
                f.write(self.plain_te.toPlainText()) # 파일 객체(f)를 이용하여 텍스트 에디터(self.plain_te)의 플레인 텍스트를 읽어와서 파일에 씁니다. 이로써 사용자가 선택한 경로에 현재 에디터의 내용이 저장됩니다.
                self.edited_flag = False
                self.plain_te.textChanged_slot.connect(self.text_changed_slot)
                self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad') # 문자열이니까 더하기  )
            return 0
        else :
            self.path = old_path
            return 1
        # 사용자가 파일을 선택하지 않거나 저장을 취소했을 경우, 이전의 파일 경로(old_path)를 다시 self.path로 복원합니다.


# 파일을 열고 저장읋 하려다가 수정 할 게 생각나서 저장하려다가 취소를 눌렀어. 그때 종료되지 않고, 창이 그대로 떠 있게 하기 위해서 return을 사용함.

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())


