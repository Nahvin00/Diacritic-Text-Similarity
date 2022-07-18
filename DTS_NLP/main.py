from PyQt5.QtWidgets import QMainWindow, QApplication, QPlainTextEdit, QPushButton, QLabel, QProgressBar
from PyQt5 import uic
from PyQt5 import QtGui
import sys
import time
from nlp_proc import main_proc


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi("NLP_UI.ui", self)

        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.textedit = self.findChild(QPlainTextEdit, "plainTextEdit")
        self.button = self.findChild(QPushButton, "pushButton")
        self.output = self.findChild(QLabel, "label_3")
        self.output.setVisible(0)
        self.progress = self.findChild(QProgressBar, "progressBar")
        self.progress.setVisible(0)
        self.button.clicked.connect(self.clickedBtn)

        self.show()

    def clickedBtn(self):
        self.output.setVisible(0)
        self.progress.setVisible(1)
        for i in range(4):
            self.progress.setValue(int(i / 3 * 100))
            time.sleep(0.5)
        self.progress.setVisible(0)
        inp = self.textedit.toPlainText()
        a, b, c, d = main_proc(inp)
        oup = "Results:\nMatched: {} with similarity of {}\nAccent: {} with confidence of {}".format(a, b, c, d)
        self.output.setVisible(1)
        self.output.setText(oup)


app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
