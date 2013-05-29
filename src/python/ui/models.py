
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider

from oscillators import NormalOscillator
from sound import WavFile
from ui.mpl_figure import MplFigure


class OscillatorModel(object):

    def __init__(self):
        pass

    def get_control_params(self):
        pass


class NormalOscillatorModel(OscillatorModel):

    def __init__(self):
        OscillatorModel.__init__(self)
        self.oscillator = NormalOscillator()

        self.control_params = dict()
        self.control_params['alpha'] = ControlParameter('alpha', [-0.70, 0.05], -0.41769)
        self.control_params['beta'] = ControlParameter('beta', [-0.40, 0.40], -0.346251775)

    def get_control_param_names(self):
        return ['alpha', 'beta']

    def get_control_params(self):
        return self.control_params




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

    def create_widget(self, parent=None):
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


class SimulationOutput(object):

    def __init__(self, output, duration, dt):
        self.output = output
        self.duration = duration
        self.dt = dt

        wf = WavFile()
        wf.sample_rate = 1.0 / dt
        wf.data = self.output[:, 0]

        self.wav_file = wf

    def create_widget(self, parent=None):
        mplfig = MplFigure(parent=parent)
        self.wav_file.plot(mplfig.figure, min_freq=100.0, max_freq=10000.0, spec_sample_rate=1000.0, freq_spacing=125.0, rms_thresh=0.0)
        return mplfig
