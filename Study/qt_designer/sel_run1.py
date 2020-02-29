# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:30:46 2018

@author: lenovo
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from sel_window import Ui_Dialog

class mwindow(QWidget, Ui_Dialog):
    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)
        
             
    def msg(self):
        directory1 = QFileDialog.getExistingDirectory(self,
                                    "选取文件夹",
                                    "D:/")                                 #起始路径
        print(directory1)
 
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                    "D:/",
                                    "All Files (*);;Text Files (*.txt)")   #设置文件扩展名过滤,注意用双分号间隔
        print(fileName1,filetype)
 
        files, ok1 = QFileDialog.getOpenFileNames(self,
                                    "多文件选择",
                                    "D:/",
                                    "All Files (*);;Text Files (*.txt)")
        print(files,ok1)
 
        fileName2, ok2 = QFileDialog.getSaveFileName(self,
                                    "文件保存",
                                    "D:/",
                                    "All Files (*);;Text Files (*.txt)")
#=========================================================== 
            
#===========================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = mwindow()
    w.sel.clicked.connect(w.msg)
    w.show()
    sys.exit(app.exec_())