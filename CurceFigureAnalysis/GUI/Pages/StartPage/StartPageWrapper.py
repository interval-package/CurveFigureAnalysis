from CurceFigureAnalysis.GUI.Pages.StartPage.StartPage import *
import os


class StartPage(QtWidgets.QWidget, Ui_StartPage):
    def __init__(self):
        super(StartPage, self).__init__()
        self.setupUi(self)

        self.SelectButton.clicked.connect(lambda: self.msg(os.path.curdir))

    def msg(self, Filepath=None):
        print(Filepath)
        # 当窗口非继承QtWidgets.QDialog时，self需替换成 None
        file = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", Filepath, "JPG pic (*.jpg);PNG Pic Files (*.png)")
        print(file)
