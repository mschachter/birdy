import sys

from PyQt4.QtGui import QApplication, QMainWindow
from ui.controllers import MainController

from ui.gen.main_window import *

from ui.models import ALL_MODELS


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, controller, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.controller = controller

        self.populate_models()
        self.select_model('Normal')

    def populate_models(self):
        for model_name,model in ALL_MODELS.iteritems():
            self.modelComboBox.addItem(model_name)

    def select_model(self, model_name):
        model = ALL_MODELS[model_name]

        for pname,cparam in model.get_control_params().iteritems():
            clayout = cparam.create_widget()
            self.oscillatorParametersLayout.addLayout(clayout)


def main():
    print 'Starting birdy app...'
    app = QApplication([])

    controller = MainController()
    main_window = MainWindow(controller)
    main_window.show()
    sys.exit(app.exec_())
