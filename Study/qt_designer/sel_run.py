# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 09:43:55 2018

@author: lenovo
"""
# import modules
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5 import QtWidgets
from sel_window import Ui_Dialog
#==============define class and initialization===================
class Dialog(QWidget, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # DIY

#=================Running Functinon=============================
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
# =====================close window=======================
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               '本程序',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

# =================main function=======================
def main():
    app = QApplication(sys.argv)
    w = Dialog()
#    w.sel.clicked.connect(w.msg)
    w.show()
    sys.exit(app.exec_())

# ===================executing==========================
if __name__ == '__main__':
    main()