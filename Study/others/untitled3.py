# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 11:04:18 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 11:03:08 2018

@author: lenovo
"""


import sys
from PyQt4 import QtCore, QtGui, uic
 
qtCreatorFile = "D:/myCodes/test.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
 
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
