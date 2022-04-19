from PyQt5.QtWidgets import QStackedWidget

from GUI.Pages.MainWindow.MainWindow import *

from GUI.Pages.StartPage.StartPageWrapper import StartPage


class GUIMain(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(GUIMain, self).__init__()
        self.setupUi(self)

        # 设置页面
        self.stackWidget = QStackedWidget()

        # 0
        self.StartPage = StartPage()
        self.stackWidget.addWidget(self.StartPage)

        # 将页面置放于窗体中间
        self.setCentralWidget(self.stackWidget)
        self.switchPage(0)

        self.setIcon()

    def switchPage(self, index):
        self.stackWidget.setCurrentIndex(index)

    def setIcon(self):
        # 由于路径问题需要重新设置一下icon的路径
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("GUI/Assets/icon_music.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
