import sys

from PyQt4.QtGui import QApplication, QMainWindow
from ui.controllers import MainController

from ui.gen.main_window import *

from ui.models import ALL_MODELS


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, controller, parent=None):
        self.initialized = False
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.controller = controller

        self.simulateButton.clicked.connect(self.simulate_clicked)
        self.modelComboBox.currentIndexChanged.connect(self.oscillator_model_changed)

        self.all_model_names = ALL_MODELS.keys()
        self.all_model_names.sort()

        self.populate_models()
        self.initialized = True
        self.select_model('Normal')

    def populate_models(self):
        for model_name in self.all_model_names:
            self.modelComboBox.addItem(model_name)

    def select_model(self, model_name):
        if not self.initialized:
            return

        clear_layout(self.oscillatorParametersLayout)

        model = ALL_MODELS[model_name]

        for pname in model.get_control_param_names():
            cparam = model.get_control_params()[pname]
            clayout = cparam.create_widget()
            self.oscillatorParametersLayout.addLayout(clayout)

        self.controller.oscillator_model = model

    def oscillator_model_changed(self, index):
        mname = self.all_model_names[index]
        self.select_model(mname)

    def simulate_clicked(self, checked):

        duration = float(self.durationEdit.text())
        dt = float(self.stepSizeEdit.text())
        print self.controller.oscillator_model

        initial_x = float(self.initialXEdit.text())
        initial_v = float(self.initialVEdit.text())

        print 'Simulating: dt=%0.6f, duration=%0.6f, initial_x=%0.2f, initial_v=%0.2f...' % (dt, duration, initial_x, initial_v)
        #sim_output = self.controller.run_simulation(self.controller.oscillator_model, duration, dt)
        sim_output = self.controller.run_simulation(self.controller.oscillator_model, duration, dt, initial_x=initial_x, initial_v=initial_v)
        self.controller.simulation_output = sim_output
        print 'Simulation done!'

        clear_layout(self.simulationPlotLayout)

        #add new plot
        sim_widget = sim_output.create_widget()
        self.simulationPlotLayout.addWidget(sim_widget)


def clear_layout(layout):
    for i in reversed(range(layout.count())):
        thing = layout.itemAt(i)
        cname = thing.__class__.__name__
        if cname.endswith('Layout'):
            clear_layout(thing)
        else:
            w = thing.widget()
            if w is not None:
                w.setParent(None)
        del thing


def main():
    print 'Starting birdy app...'
    app = QApplication([])

    controller = MainController()
    main_window = MainWindow(controller)
    main_window.show()
    sys.exit(app.exec_())
