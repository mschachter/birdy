import subprocess
import wave
import struct

import numpy as np
from scipy.io.wavfile import read as read_wavfile


class WavFile():
    """ Class for representing a sound and writing it to a .wav file """

    def __init__(self, file_name=None):

        if file_name is None:
            self.sample_depth = 2  # in bytes
            self.sample_rate = 44100.0  # in Hz
            self.data = None
            self.num_channels = 1
        else:
            wr = wave.open(file_name, 'r')
            self.num_channels = wr.getnchannels()
            self.sample_depth = wr.getsampwidth()
            wr.close()

            self.sample_rate,self.data = read_wavfile(file_name)

    def to_wav(self, output_file, normalize=False, max_amplitude=32767.0):
        wf = wave.open(output_file, 'w')

        wf.setparams( (self.num_channels, self.sample_depth, self.sample_rate, len(self.data), 'NONE', 'not compressed') )
        #normalize the sample
        if normalize:
            nsound = ((self.data / np.abs(self.data).max())*max_amplitude).astype('int')
        else:
            nsound = self.data
        print 'nsound.min=%d, max=%d' % (nsound.min(), nsound.max())
        hex_sound = [struct.pack('h', x) for x in nsound]
        wf.writeframes(''.join(hex_sound))
        wf.close()


def play_sound(file_name):
    """ Install sox to get this to work: http://sox.sourceforge.net/ """
    subprocess.call(['play', file_name])


