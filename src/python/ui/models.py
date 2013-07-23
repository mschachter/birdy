
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QWidget

import numpy as np

import matplotlib.pyplot as plt

from oscillators import NormalOscillator, PhysicalOscillator
from tools.sound import WavFile
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
        self.control_params['alpha'] = ControlParameter('alpha', [-1.25, 0.20], -0.41769)
        self.control_params['beta'] = ControlParameter('beta', [-0.75, 0.75], -0.346251775)

    def get_control_param_names(self):
        return ['alpha', 'beta']

    def get_control_params(self):
        return self.control_params


class PhysicalOscillatorModel(OscillatorModel):

    def __init__(self):
        OscillatorModel.__init__(self)
        self.oscillator = PhysicalOscillator()

        self.control_params = dict()
        self.control_params['psub'] = ControlParameter('psub', [1800.0, 2100.0], 1900.0)
        self.control_params['k1'] = ControlParameter('k1', [0.005, 0.75], 0.016)
        self.control_params['f0'] = ControlParameter('f0', [-0.05, 0.05], 0.0399)

    def get_control_param_names(self):
        return ['psub', 'k1', 'f0']

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
        mlayout = QVBoxLayout(parent)

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

ALL_MODELS = {'Normal': NormalOscillatorModel(), 'Physical': PhysicalOscillatorModel()}


class SimulationOutput(object):

    def __init__(self, output, duration, dt):
        self.output = output
        self.duration = duration
        self.dt = dt

        wf = WavFile()
        wf.sample_rate = 1.0 / dt
        wf.data = self.output[:, 0]
        wf.analyze()

        self.wav_file = wf

    def create_widget(self, parent=None):
        mplfig = MplFigure(parent=parent)

        fig = mplfig.figure
        ax = fig.add_subplot()

        gs = plt.GridSpec(100, 1)
        ax = fig.add_subplot(gs[:14])
        ax.plot(self.wav_file.data_t, self.output[:, 0], 'k-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('x(t)')

        ax = fig.add_subplot(gs[20:34])
        ax.plot(self.wav_file.data_t, self.output[:, 1], 'g-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('v(t)')

        ax = fig.add_subplot(gs[40:54])
        ff = self.wav_file.fundamental_freq
        ff_index = (np.where(self.wav_file.power_spectrum_f == ff)[0]).min()
        ax.plot(self.wav_file.power_spectrum_f, self.wav_file.power_spectrum, 'k-')
        ax.plot(ff, self.wav_file.power_spectrum[ff_index], 'ro')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('log10(power)')

        ax = fig.add_subplot(gs[60:94])
        self.wav_file.plot_spectrogram(ax)

        return mplfig
