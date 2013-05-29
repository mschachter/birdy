import copy

from PyQt4.QtCore import QSize, QRect
from PyQt4.QtGui import QWidget, QIcon, QPixmap, QToolButton, QDialog, QHBoxLayout, QLayout

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from ui.gen.mpl_popup_window import Ui_MplPopupWindow

from ui.gen.mpl_widget import Ui_MplWidget
from ui.config import get_image_path

class MplFigure(Ui_MplWidget, QWidget):

    def __init__(self, parent=None, special_actions=[]):
        QWidget.__init__(self, parent=parent)
        self.setupUi(self)

        self.popup = None

        self.figure = Figure()
        self.figure.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(parent)
        self.special_actions = []

        tw = QWidget()
        self.button_layout = QHBoxLayout(tw)
        self.button_layout.setSizeConstraint(QLayout.SetFixedSize)

        icon = QIcon()
        icon.addPixmap(QPixmap(get_image_path('popup_icon.png')), QIcon.Normal, QIcon.Off)
        self.toolButton = QToolButton(parent=None)
        self.toolButton.setIcon(icon)
        self.toolButton.setIconSize(QSize(16, 16))
        self.toolButton.clicked.connect(self.popup_clicked)
        self.toolButton.setStatusTip('Expand into new window')
        self.button_layout.addWidget(self.toolButton)

        for sa in special_actions:
            icon = QIcon()
            icon.addPixmap(QPixmap(get_image_path(sa.icon_path)), QIcon.Normal, QIcon.Off)
            tb = QToolButton(parent=tw)
            tb.setIcon(icon)
            tb.setIconSize(QSize(16, 16))
            tb.clicked.connect(sa.execute)
            self.special_actions.append(sa)
            self.button_layout.addWidget(tb)

        self.verticalLayout.addWidget(tw)
        self.verticalLayout.addWidget(self.canvas)

    def popup_clicked(self):
        if self.popup is None:
            self.popup = MplFigurePopup(self.figure, parent=self)
        self.popup.show()


class MplFigurePopup(Ui_MplPopupWindow, QDialog):

    def __init__(self, figure, parent=None):
        QDialog.__init__(self, parent=parent)
        self.setupUi(self)

        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.verticalLayout.addWidget(self.canvas)
        self.verticalLayout.addWidget(self.toolbar)


class MplFigureSpecialAction(object):

    def __init__(self, icon_path):
        self.icon_path = icon_path

    def execute(self):
        pass

