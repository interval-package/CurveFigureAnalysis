# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'StartPage.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_StartPage(object):
    def setupUi(self, StartPage):
        StartPage.setObjectName("StartPage")
        StartPage.resize(400, 300)
        self.gridLayoutWidget = QtWidgets.QWidget(StartPage)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, -1, 401, 301))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(30, 30, 30, 30)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 2, 0, 1, 1)
        self.SelectButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.SelectButton.setObjectName("SelectButton")
        self.gridLayout.addWidget(self.SelectButton, 1, 0, 1, 2)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 2, 1, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 0, 1, 1, 1)

        self.retranslateUi(StartPage)
        QtCore.QMetaObject.connectSlotsByName(StartPage)

    def retranslateUi(self, StartPage):
        _translate = QtCore.QCoreApplication.translate
        StartPage.setWindowTitle(_translate("StartPage", "Form"))
        self.label.setText(_translate("StartPage", "TextLabel"))
        self.pushButton_3.setText(_translate("StartPage", "PushButton"))
        self.SelectButton.setText(_translate("StartPage", "Select Pic"))
        self.pushButton_2.setText(_translate("StartPage", "PushButton"))
