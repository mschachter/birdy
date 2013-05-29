# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mpl_popup_window.ui'
#
# Created: Wed Mar 28 14:47:06 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MplPopupWindow(object):
    def setupUi(self, MplPopupWindow):
        MplPopupWindow.setObjectName(_fromUtf8("MplPopupWindow"))
        MplPopupWindow.resize(703, 541)
        self.verticalLayout = QtGui.QVBoxLayout(MplPopupWindow)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))

        self.retranslateUi(MplPopupWindow)
        QtCore.QMetaObject.connectSlotsByName(MplPopupWindow)

    def retranslateUi(self, MplPopupWindow):
        MplPopupWindow.setWindowTitle(QtGui.QApplication.translate("MplPopupWindow", "Dialog", None, QtGui.QApplication.UnicodeUTF8))

