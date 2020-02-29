import sys
import os
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon
from easyLearn_gui1 import Ui_MainWindow

import sys
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
from PyQt5.QtCore import QStringListModel

class MainCode_easylearn(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # initiating listView
        self.slm = QStringListModel()

        # connections
        self.browser_workingdir.clicked.connect(self.select_workingdir)

        self.pushButton_cleargroups.clicked.connect(self.clear_all_selection)
        self.pushButton_removegroups.clicked.connect(self.remove_selected_file_by_pushbutton)
        self.listView_groups.doubleClicked.connect(self.remove_selected_file_by_doubleclicked)
        self.listView_groups.clicked.connect(self.selecting_file)
        self.pushButton_addgroups.clicked.connect(self.select_workingdir)
        # self.listView.clicked.connect(self.remove_selected_file_by_doubleclicked)
        self.setWindowTitle('easylearn')
        self.setWindowIcon(QIcon('D:/My_Codes/LC_Machine_Learning/lc_rsfmri_tools/lc_rsfmri_tools_python/gui_test/bitbug_favicon.ico')) 
        
    def select_workingdir(self):
        try:
            self.directory
        except:
            self.directory = 0

        if not self.directory:
            self.directory = QFileDialog.getExistingDirectory(self, "Select a directory", os.getcwd()) 
        else:
            self.directory = QFileDialog.getExistingDirectory(self, "Select a directory", self.directory) 

        self.lineEdit_workingdir.setText(self.directory)

        self.qList = os.listdir(self.directory)
        self.slm.setStringList(self.qList)  
        self.listView_groups.setModel(self.slm) 
        self.current_list = 0  # Every time selection, the current list will be initiated once.

    def selecting_file(self, QModelIndex):
        if self.current_list == 0:
            self.current_list = self.qList

        self.selected_file = self.current_list[QModelIndex.row()]
        print(self.selected_file)

    def remove_selected_file_by_doubleclicked(self,QModelIndex):
        """
        Remove the selected file
        """
        if self.current_list == 0:
            self.current_list = self.qList

        selected_file = self.current_list[QModelIndex.row()]
        QMessageBox.information(self, "WARNING", "Remove selection: "+ selected_file)
        self.current_list = list(set(self.current_list) - set([self.current_list[QModelIndex.row()]]))
        self.slm.setStringList(self.current_list)  
        self.listView_groups.setModel(self.slm) 

    def remove_selected_file_by_pushbutton(self):
        """
        TODO
        """
        print(f'Remove {self.selected_file}') 

        if self.current_list == 0:
            self.current_list = self.qList

        QMessageBox.information(self, "QListView", "Remove this file: "+ self.selected_file)
        self.current_list = list(set(self.current_list) - set([self.selected_file]))  # Note. the second item in set is list
        self.slm.setStringList(self.current_list)  
        self.listView_groups.setModel(self.slm)

    def clear_all_selection(self):
        """
        Remove all selections
        """
        self.slm.setStringList([])  # 设置模型列表视图，加载数据列表
        self.listView_groups.setModel(self.slm)  #设置列表视图的模型
        self.current_list == 0  # re-initiate the current_list

    def main(self):
        app=QApplication(sys.argv)
        md=MainCode_easylearn()
        md.show()
        sys.exit(app.exec_())

if __name__=='__main__':
    app=QApplication(sys.argv)
    md=MainCode_easylearn()
    md.show()
    sys.exit(app.exec_())
