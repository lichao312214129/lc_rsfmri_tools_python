# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
#pyuic5 test.ui –o test.py
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Ui_lichao(object):
    def setupUi(self, lichao):
        lichao.setObjectName("lichao")
        lichao.resize(385, 394)
        self.Running = QtWidgets.QPushButton(lichao)
        self.Running.setGeometry(QtCore.QRect(90, 140, 141, 31))
        self.Running.setObjectName("Running")
        self.Running.clicked.connect(self.running)#
        self.textEdit = QtWidgets.QTextEdit(lichao)
        self.textEdit.setGeometry(QtCore.QRect(110, 40, 201, 41))
        self.textEdit.setObjectName("textEdit")
        self.word = QtWidgets.QLabel(lichao)
        self.word.setGeometry(QtCore.QRect(110, 20, 72, 15))
        self.word.setLineWidth(5)
        self.word.setMidLineWidth(1)
        self.word.setTextFormat(QtCore.Qt.AutoText)
        self.word.setObjectName("word")
        self.limit = QtWidgets.QSpinBox(lichao)
        self.limit.setGeometry(QtCore.QRect(140, 260, 46, 22))
        self.limit.setToolTipDuration(-7)
        self.limit.setProperty("value", 20)
        self.limit.setObjectName("limit")
        self.retranslateUi(lichao)
        QtCore.QMetaObject.connectSlotsByName(lichao)

    def retranslateUi(self, lichao):
        _translate = QtCore.QCoreApplication.translate
        lichao.setWindowTitle(_translate("lichao", "Dialog"))
        self.Running.setText(_translate("lichao", "Running"))
        self.textEdit.setHtml(_translate("lichao", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">董梦实是个小屁孩</span></p></body></html>"))
        self.word.setText(_translate("lichao", "word"))
        
#=====================================================        
#    def running1(self):
##        word='facai'
##        print(word)
#        x= self.textEdit.toPlainText()
#        x1=list(x)
#        print(x1)
#        plt.plot(x1)
#
##        self.textEdit.setText(word)
        
        
 
    
#===========================================================
if __name__=='__main__':

    app=QtWidgets.QApplication(sys.argv)
#    MainWindow = QtWidgets.QMainWindow()
#    ui = Ui_lichao()
#    ui.setupUi(MainWindow)
#    MainWindow.show()
#    sys.exit(app.exec_())
    #
    myshow=QtWidgets.QWidget()
    ui=Ui_lichao()
    ui.setupUi(myshow)
    myshow.show()
    sys.exit(app.exec_())
#    
#    myshow=MyWindow()
#    myshow.show()
#    sys.exit(app.exec_())  
