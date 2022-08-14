import sys

from PyQt5.QtWidgets import QApplication
from CurceFigureAnalysis.GUI.GUIWrapper import *


def Gui_main():
    # 初始化软件，这是必要操作
    app = QApplication(sys.argv)

    # 创建窗体对象
    win = GUIMain()
    win.show()

    # 结束所有逻辑，之前所有循环的逻辑结束
    sys.exit(app.exec_())


if __name__ == '__main__':
    Gui_main()
    pass
