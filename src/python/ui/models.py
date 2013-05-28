
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider

from oscillators import NormalOscillator


class OscillatorModel(object):

    def __init__(self):
        pass

    def get_control_params(self):
        pass

    def get_control_bounds(self, control_param):
        pass

    def get_control_default(self, control_param):
        pass


class NormalOscillatorModel(OscillatorModel):

    def __init__(self):
        OscillatorModel.__init__(self)
        self.oscillator = NormalOscillator
        self.bounds = {'alpha': [-0.70, 0.05], 'beta': [-0.40, 0.40]}
        self.defaults = {'alpha':-0.41769, 'beta':-0.346251775}

    def get_control_params(self):
        return ['alpha', 'beta']

    def get_control_bounds(self, control_param):
        return self.bounds[control_param]

    def get_control_default(self, control_param):
        return self.defaults[control_param]


class ControlParameter(object):

    def __init__(self, name, bounds, default_val):
        self.name = name
        self.min_val,self.max_val = bounds
        self.default_val = default_val
        self.current_val = default_val

    def get_slider_index(self, val):
        return int( 100*(val - self.min_val) / (self.max_val - self.min_val) )

    def get_slider_value(self, index):
        frac = index / 100.0
        return (self.max_val - self.min_val)*frac + self.min_val

    def set_slider_value(self, val):
        index = self.get_slider_index(val)
        self.slider.setSliderPosition(index)

    def slider_changed(self, index):
        self.current_val = self.get_slider_value(index)
        #print 'slider changed %s: index=%d, val=%0.6f' % (self.name, index, self.current_val)
        self.update()

    def update(self):
        self.pedit.setText('%0.6f' % self.current_val)
        self.set_slider_value(self.current_val)

    def create_layout(self):
        mlayout = QVBoxLayout()

        hlayout = QHBoxLayout()
        plabel = QLabel()
        plabel.setText('%s: ' % self.name)
        hlayout.addWidget(plabel)

        self.pedit = QLineEdit()
        self.pedit.setText('%0.6f' % self.default_val)
        hlayout.addWidget(self.pedit)
        mlayout.addLayout(hlayout)

        spos = self.get_slider_index(self.default_val)
        self.slider = QSlider()
        self.slider.setRange(0, 100)
        self.slider.setSliderPosition(spos)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        mlayout.addWidget(self.slider)

        return mlayout

ALL_MODELS = {'Normal': NormalOscillatorModel()}