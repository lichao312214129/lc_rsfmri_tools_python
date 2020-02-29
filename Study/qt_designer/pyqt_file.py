# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:11:40 2018

@author: lenovo
"""

import sys
import os
from PyQt5.QtWidgets import QApplication,QWidget,QFileDialog
from t import Ui_Form

class myform(QWidget,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.btn1.clicked.connect(self.btn_clear)
        self.btn2.clicked.connect(self.btn_open)
        self.btn3.clicked.connect(self.btn_save)
        self.show()
    def btn_clear(self):
        self.textEdit.clear()
    def btn_open(self):
        filename=QFileDialog.getOpenFileName(self,'open file','/home/jm/study')
        with open(filename[0],'r') as f:
            my_txt=f.read()
            self.textEdit.setPlainText(my_txt)
    def btn_save(self):
        filename=QFileDialog.getSaveFileName(self,'save file','/home/jm/study')
        with open(filename[0],'w') as f:
            my_text=self.textEdit.toPlainText()
            f.write(my_text)
          
if __name__=='__main__':
    app=QApplication(sys.argv)
    w=myform()
    app.exec_()