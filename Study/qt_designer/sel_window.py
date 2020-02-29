# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sel.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets
 

class Ui_Dialog(object):
        
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 256)
        self.sel = QtWidgets.QPushButton(Dialog)
        self.sel.setGeometry(QtCore.QRect(160, 90, 93, 28))
        self.sel.setObjectName("sel")
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.sel.setText(_translate("Dialog", "PushButton"))

