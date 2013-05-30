# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created: Thu May 30 16:04:58 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1105, 918)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(-1, -1, 1111, 841))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.tabWidget = QtGui.QTabWidget(self.horizontalLayoutWidget)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.oscilatorParamsGroupBox = QtGui.QGroupBox(self.tab)
        self.oscilatorParamsGroupBox.setGeometry(QtCore.QRect(0, 0, 291, 401))
        self.oscilatorParamsGroupBox.setStyleSheet(_fromUtf8("QGroupBox { \n"
"     border: 2px solid gray; \n"
"     border-radius: 3px; \n"
" } "))
        self.oscilatorParamsGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.oscilatorParamsGroupBox.setFlat(False)
        self.oscilatorParamsGroupBox.setObjectName(_fromUtf8("oscilatorParamsGroupBox"))
        self.modelHelpLabel = QtGui.QLabel(self.oscilatorParamsGroupBox)
        self.modelHelpLabel.setGeometry(QtCore.QRect(30, 20, 61, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.modelHelpLabel.setFont(font)
        self.modelHelpLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.modelHelpLabel.setObjectName(_fromUtf8("modelHelpLabel"))
        self.modelComboBox = QtGui.QComboBox(self.oscilatorParamsGroupBox)
        self.modelComboBox.setGeometry(QtCore.QRect(40, 50, 221, 27))
        self.modelComboBox.setObjectName(_fromUtf8("modelComboBox"))
        self.verticalLayoutWidget = QtGui.QWidget(self.oscilatorParamsGroupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 150, 271, 241))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.oscillatorParametersLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.oscillatorParametersLayout.setMargin(0)
        self.oscillatorParametersLayout.setObjectName(_fromUtf8("oscillatorParametersLayout"))
        self.initialXHelpLabel = QtGui.QLabel(self.oscilatorParamsGroupBox)
        self.initialXHelpLabel.setGeometry(QtCore.QRect(30, 80, 66, 17))
        self.initialXHelpLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.initialXHelpLabel.setObjectName(_fromUtf8("initialXHelpLabel"))
        self.initialVHelpLabel = QtGui.QLabel(self.oscilatorParamsGroupBox)
        self.initialVHelpLabel.setGeometry(QtCore.QRect(30, 110, 66, 17))
        self.initialVHelpLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.initialVHelpLabel.setObjectName(_fromUtf8("initialVHelpLabel"))
        self.initialXEdit = QtGui.QLineEdit(self.oscilatorParamsGroupBox)
        self.initialXEdit.setGeometry(QtCore.QRect(100, 80, 113, 27))
        self.initialXEdit.setObjectName(_fromUtf8("initialXEdit"))
        self.initialVEdit = QtGui.QLineEdit(self.oscilatorParamsGroupBox)
        self.initialVEdit.setGeometry(QtCore.QRect(100, 110, 113, 27))
        self.initialVEdit.setObjectName(_fromUtf8("initialVEdit"))
        self.simulatorGroupBox = QtGui.QGroupBox(self.tab)
        self.simulatorGroupBox.setGeometry(QtCore.QRect(0, 410, 291, 121))
        self.simulatorGroupBox.setStyleSheet(_fromUtf8("QGroupBox { \n"
"     border: 2px solid gray; \n"
"     border-radius: 3px; \n"
" } "))
        self.simulatorGroupBox.setObjectName(_fromUtf8("simulatorGroupBox"))
        self.stepSizeHelpLabel = QtGui.QLabel(self.simulatorGroupBox)
        self.stepSizeHelpLabel.setGeometry(QtCore.QRect(10, 30, 81, 21))
        self.stepSizeHelpLabel.setObjectName(_fromUtf8("stepSizeHelpLabel"))
        self.stepSizeEdit = QtGui.QLineEdit(self.simulatorGroupBox)
        self.stepSizeEdit.setGeometry(QtCore.QRect(90, 30, 191, 27))
        self.stepSizeEdit.setObjectName(_fromUtf8("stepSizeEdit"))
        self.durationHelpLabel_2 = QtGui.QLabel(self.simulatorGroupBox)
        self.durationHelpLabel_2.setGeometry(QtCore.QRect(10, 60, 81, 21))
        self.durationHelpLabel_2.setObjectName(_fromUtf8("durationHelpLabel_2"))
        self.durationEdit = QtGui.QLineEdit(self.simulatorGroupBox)
        self.durationEdit.setGeometry(QtCore.QRect(90, 60, 191, 27))
        self.durationEdit.setObjectName(_fromUtf8("durationEdit"))
        self.simulateButton = QtGui.QPushButton(self.simulatorGroupBox)
        self.simulateButton.setGeometry(QtCore.QRect(100, 90, 98, 27))
        self.simulateButton.setObjectName(_fromUtf8("simulateButton"))
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.tab)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(300, 0, 801, 791))
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.simulationPlotLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.simulationPlotLayout.setMargin(0)
        self.simulationPlotLayout.setObjectName(_fromUtf8("simulationPlotLayout"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1105, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.oscilatorParamsGroupBox.setTitle(QtGui.QApplication.translate("MainWindow", "Oscillator Parameters", None, QtGui.QApplication.UnicodeUTF8))
        self.modelHelpLabel.setText(QtGui.QApplication.translate("MainWindow", "Model:", None, QtGui.QApplication.UnicodeUTF8))
        self.initialXHelpLabel.setText(QtGui.QApplication.translate("MainWindow", "x(0):", None, QtGui.QApplication.UnicodeUTF8))
        self.initialVHelpLabel.setText(QtGui.QApplication.translate("MainWindow", "v(0):", None, QtGui.QApplication.UnicodeUTF8))
        self.initialXEdit.setText(QtGui.QApplication.translate("MainWindow", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.initialVEdit.setText(QtGui.QApplication.translate("MainWindow", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.simulatorGroupBox.setTitle(QtGui.QApplication.translate("MainWindow", "Simulation", None, QtGui.QApplication.UnicodeUTF8))
        self.stepSizeHelpLabel.setText(QtGui.QApplication.translate("MainWindow", "Step Size:", None, QtGui.QApplication.UnicodeUTF8))
        self.stepSizeEdit.setText(QtGui.QApplication.translate("MainWindow", "0.000001", None, QtGui.QApplication.UnicodeUTF8))
        self.durationHelpLabel_2.setText(QtGui.QApplication.translate("MainWindow", "Duration:", None, QtGui.QApplication.UnicodeUTF8))
        self.durationEdit.setText(QtGui.QApplication.translate("MainWindow", "0.020", None, QtGui.QApplication.UnicodeUTF8))
        self.simulateButton.setText(QtGui.QApplication.translate("MainWindow", "Simulate", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("MainWindow", "Model", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("MainWindow", "Tab 2", None, QtGui.QApplication.UnicodeUTF8))

