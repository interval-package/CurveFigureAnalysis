from PyQt5.QtWidgets import QStackedWidget

from GUI.Pages.MainWindow.MainWindow import *


class GUIMain(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(GUIMain, self).__init__()
        self.setupUi(self)

        # 设置页面
        self.stackWidget = QStackedWidget()

