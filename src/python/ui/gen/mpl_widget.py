# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mpl_widget.ui'
#
# Created: Wed Mar 28 14:30:15 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MplWidget(object):
    def setupUi(self, MplWidget):
        MplWidget.setObjectName(_fromUtf8("MplWidget"))
        MplWidget.resize(638, 479)
        self.verticalLayout = QtGui.QVBoxLayout(MplWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))

        self.retranslateUi(MplWidget)
        QtCore.QMetaObject.connectSlotsByName(MplWidget)

    def retranslateUi(self, MplWidget):
        MplWidget.setWindowTitle(QtGui.QApplication.translate("MplWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))

