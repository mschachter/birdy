import subprocess
import wave
import struct

import numpy as np
from scipy.io.wavfile import read as read_wavfile
from scipy.fftpack import fft,fftfreq

import matplotlib.pyplot as plt


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
        #print 'nsound.min=%d, max=%d' % (nsound.min(), nsound.max())
        hex_sound = [struct.pack('h', x) for x in nsound]
        wf.writeframes(''.join(hex_sound))
        wf.close()

    def analyze(self, min_freq=0, max_freq=10000.0, spec_sample_rate=1000.0, freq_spacing=125.0):

        self.data_t = np.arange(0.0, len(self.data), 1.0) / self.sample_rate

        #compute log power spectrum
        fftx = fft(self.data)
        ps_f = fftfreq(len(self.data), d=(1.0 / self.sample_rate))
        findx = (ps_f > min_freq) & (ps_f < max_freq)
        self.power_spectrum = np.log10(np.abs(fftx[findx]))
        self.power_spectrum_f = ps_f[findx]

        #estimate fundamental frequency from log power spectrum in the simplest way possible
        ps = np.abs(fftx[findx])
        peak_index = ps.argmax()
        try:
            self.fundamental_freq = self.power_spectrum_f[peak_index]
        except IndexError:
            print 'Could not identify fundamental frequency!'
            self.fundamental_freq = 0.0

        #compute spectrogram
        t,f,spec,spec_rms = log_spectrogram(self.data, self.sample_rate, spec_sample_rate=spec_sample_rate, freq_spacing=freq_spacing, min_freq=min_freq, max_freq=max_freq)
        self.spectrogram_t = t
        self.spectrogram_f = f
        self.spectrogram = spec
        self.spectrogram_rms = spec_rms

    def plot(self, fig=None, show_rms=True):

        self.analyze()

        if show_rms:
            spw_size = 15
            spec_size = 35
        else:
            spw_size = 25
            spec_size = 75

        if fig is None:
            fig = plt.figure()
        gs = plt.GridSpec(100, 1)
        ax = fig.add_subplot(gs[:spw_size])
        plt.plot(self.data_t, self.data, 'k-')
        plt.axis('tight')
        plt.ylabel('Sound Pressure')

        s = (spw_size+5)
        e = s + spec_size
        ax = fig.add_subplot(gs[s:e])
        self.plot_spectrogram()

        if show_rms:
            ax = fig.add_subplot(gs[(e+5):95])
            plt.plot(self.spectrogram_t, self.spectrogram_rms, 'g-')
            plt.xlabel('Time (s)')
            plt.ylabel('RMS')
            plt.axis('tight')

    def plot_spectrogram(self, ax=None):
        if ax is None:
            ax = plt.gca()
        nxticks = 8
        nyticks = 4
        ex = (0.0, self.spectrogram_t.max(), self.spectrogram_f.min(), self.spectrogram_f.max())
        iax = ax.imshow(self.spectrogram, aspect='auto', interpolation='nearest', origin='lower', extent=ex)
        ax.set_ylabel('Frequency (Hz)')
        """
        xtick_spacing = len(self.spectrogram_t) / nxticks
        ytick_spacing = len(self.spectrogram_f) / nyticks
        xtick_indices = np.arange(0, len(self.spectrogram_t), xtick_spacing, dtype='int')
        ytick_indices = np.arange(0, len(self.spectrogram_f), ytick_spacing, dtype='int')
        ax.set_xticks(xtick_indices, ['%0.1f' % x for x in self.spectrogram_t[xtick_indices]])
        ax.set_yticks(ytick_indices, ['%d' % x for x in self.spectrogram_f[ytick_indices]])
        """


def play_sound(file_name):
    """ Install sox to get this to work: http://sox.sourceforge.net/ """
    subprocess.call(['play', file_name])


def log_spectrogram(s, sample_rate, spec_sample_rate, freq_spacing, min_freq=0, max_freq=None, noise_level_db=80, nstd=6):

    increment = 1.0 / spec_sample_rate
    window_length = nstd / (2.0*np.pi*freq_spacing)
    t,freq,timefreq,rms = gaussian_stft(s, sample_rate, window_length, increment, nstd=nstd, min_freq=min_freq, max_freq=max_freq)

    #create log spectrogram (power in decibels)
    spec = 20.0*np.log10(np.abs(timefreq)) + noise_level_db
    #rectify spectrogram
    spec[spec < 0.0] = 0.0

    return t,freq,spec,rms


def gaussian_stft(s, sample_rate, window_length, increment, nstd=6, min_freq=0, max_freq=None):

    if max_freq is None:
        max_freq = sample_rate / 2.0

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_length)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen / 2

    nincrement = int(sample_rate*increment)
    nwindows = len(s) / nincrement
    #print 'len(s)=%d, nwinlen=%d, hwinlen=%d, nincrement=%d, nwindows=%d' % (len(s), nwinlen, hnwinlen, nincrement, nwindows)

    #construct the window
    gauss_t = np.arange(-hnwinlen, hnwinlen, 1.0)
    gauss_std = nwinlen / float(nstd)
    gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))

    #pad the signal with zeros
    zs = np.zeros([len(s) + 2*hnwinlen])
    zs[hnwinlen:-hnwinlen] = s

    #get the frequencies corresponding to the FFTs to come
    fft_len = nwinlen+1
    full_freq = fftfreq(nwinlen+1, d=1.0 / sample_rate)
    freq_index = (full_freq >= min_freq) & (full_freq <= max_freq)
    freq = full_freq[freq_index]
    nfreq = freq_index.sum()

    #take the FFT of each segment, padding with zeros when necessary to keep window length the same
    timefreq = np.zeros([nfreq, nwindows], dtype='complex')
    rms = np.zeros([nwindows])
    for k in range(nwindows):
        center = k*nincrement + hnwinlen
        si = center - hnwinlen
        ei = center + hnwinlen
        rms[k] = zs[si:ei].std(ddof=1)
        windowed_slice = zs[si:ei]*gauss_window
        zs_fft = fft(windowed_slice, n=fft_len, overwrite_x=1)
        timefreq[:, k] = zs_fft[freq_index]

    t = np.arange(0, nwindows, 1.0) * increment

    return t,freq,timefreq,rms
