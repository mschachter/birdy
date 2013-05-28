import sys

from PyQt4.QtGui import QApplication, QMainWindow

from ui.gen.main_window import *

from ui.models import ALL_MODELS, ControlParameter


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, controller, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.controller = controller

        self.populate_models()
        self.select_model('Normal')

    def populate_models(self):

        self.control_params = dict()

        for model_name,model in ALL_MODELS.iteritems():
            for pname in model.get_control_params():
                dval = model.get_control_default(pname)
                min_val,max_val = model.get_control_bounds(pname)
                cparam = ControlParameter(pname, (min_val, max_val), dval)
                self.control_params[pname] = cparam

            self.modelComboBox.addItem(model_name)

    def select_model(self, model_name):
        model = ALL_MODELS[model_name]

        for pname in model.get_control_params():
            cparam = self.control_params[pname]
            clayout = cparam.create_layout()
            self.oscillatorParametersLayout.addLayout(clayout)


def main():
    print 'Starting birdy app...'
    app = QApplication([])

    main_window = MainWindow(None)
    main_window.show()
    sys.exit(app.exec_())
