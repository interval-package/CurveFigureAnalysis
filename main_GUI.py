import sys

from PyQt5.QtWidgets import QApplication
from GUI.GUIWrapper import GUIMain
from GUI.GUIWrapper import *


if __name__ == '__main__':
    # 初始化软件，这是必要操作
    app = QApplication(sys.argv)

    # 创建窗体对象
    win = GUIMain()
    win.show()

    # 结束所有逻辑，之前所有循环的逻辑结束
    sys.exit(app.exec_())
    pass
