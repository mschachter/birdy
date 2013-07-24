import os
import h5py
import numpy as np

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from tools.signal import gaussian_stft
from tools.sound import spectrogram,WavFile,plot_spectrogram

from oscillators import NormalOscillator


class Motogram(object):

    def __init__(self):
        self.oscillator = NormalOscillator()
        self.sample_rate = 1000.0
        self.waveform_sample_rate = 1e5
        self.alpha = list()
        self.beta = list()
        self.mu = list()
        self.sigma1 = list()
        self.sigma2 = list()

    def simulate(self):

        nsteps = len(self.alpha)
        if nsteps == 0:
            print 'Nothing to simulate!'

        total_duration = nsteps / self.sample_rate
        step_duration = 1.0 / self.sample_rate
        dt = 1.0 / self.waveform_sample_rate

        Nwave = int(np.ceil(total_duration * self.waveform_sample_rate))
        waveform = np.zeros(Nwave)
        print 'nsteps=%d, total_duration=%0.6f, step_duration=%0.6f, dt=%0.6f, Nwave=%d' % (nsteps, total_duration, step_duration, dt, Nwave)

        #extend alpha and beta by one time step for interpolation
        alpha_ext = np.zeros(nsteps+1)
        alpha_ext[:-1] = self.alpha
        alpha_ext[-1] = alpha_ext[-2]
        
        beta_ext = np.zeros(nsteps+1)
        beta_ext[:-1] = self.beta
        beta_ext[-1] = beta_ext[-2]

        #upsample alpha and beta to a finer timescale for simulation
        t_alpha = np.arange(nsteps+1)*(1.0 / self.sample_rate)
        f_alpha = interp1d(t_alpha, alpha_ext)
        f_beta = interp1d(t_alpha, beta_ext)

        #simulate the oscillator to produce the sound pressure waveform
        last_x = 0.0
        last_v = 0.0
        for k in range(Nwave):
            t = k*dt
            a = f_alpha(t)
            b = f_beta(t)
            states = self.oscillator.simulate(last_x, last_v, dt, dt, alpha=a, beta=b)
            #print 'k=%d, last_x=%f, last_v=%f, step_duration=%0.6f, dt=%0.6f, alpha=%0.6f, beta=%0.6f' % (k, last_x, last_v, step_duration, dt, a, b)
            waveform[k] = states[0, 0]

            last_x = states[0, 0]
            last_v = states[0, 1]
            if np.isnan(last_x):
                last_x = 0.0
            if np.isnan(last_v):
                last_v = 0.0

        #transform the waveform into a spectrogram
        nstd = 6.0
        freq_spacing = 125.0
        increment = 1.0 / 1000.0
        window_length = nstd / (2.0*np.pi*freq_spacing)
        spec_t,spec_f,spec,rms = gaussian_stft(waveform, self.waveform_sample_rate, window_length, increment, nstd=nstd, min_freq=0.0, max_freq=10000.0)
        spec = np.abs(spec)

        #do vocal tract filtering on the spectrogram
        spec_filt = np.zeros(spec.shape)
        for k,t in enumerate(spec_t):
            mu_t = self.mu[k]
            sigma1_t = self.sigma1[k]
            sigma2_t = self.sigma2[k]
            print 'k=%d, mu_t=%0.6f, sigma1_t=%0.6f, sigma2_t=%0.6f' % (k, mu_t, sigma1_t, sigma2_t)

            #create gaussian based on sigma2
            mean2 = (mu_t - spec_f)
            mean2[mean2 < 0.0] = 0.0
            gauss2 = mean2**2 / (2*sigma2_t**2)

            #create gaussian based on sigma1
            mean1 = (spec_f - mu_t)
            mean1[mean1 < 0.0] = 0.0
            gauss1 = mean1**2 / (2*sigma1_t**2)

            #create filter as combination of two gaussians
            filt = np.exp(-(gauss1 + gauss2))

            #filter the spectrogram
            sfilt = spec[:, k]*filt

            #compute normalization factors
            spi = spec[:, k].sum()
            ssi = sfilt.sum()

            #normalize the filtered spectrogram
            spec_filt[:, k] = sfilt / (spi*ssi)

        return waveform,spec_t,spec_f,spec_filt

    def plot(self):

        waveform,spec_t,spec_f,spec = self.simulate()

        nsp = 4
        if hasattr(self, 'wav_file'):
            nsp += 1

        plt.figure()
        plt.subplots_adjust(bottom=0.05, right=0.99, top=0.99, left=0.08, wspace=0.0, hspace=0.25)
        plt.subplot(nsp, 1, 1)
        wt = np.arange(len(waveform))*(1.0 / self.waveform_sample_rate)
        plt.plot(wt, waveform, 'k-')
        plt.legend(['x(t)'])
        plt.axis('tight')

        plt.subplot(nsp, 1, 2)
        plt.title('Model Spectrogram')
        plot_spectrogram(spec_t, spec_f, spec, fmin=0.0, fmax=8000.0)
        plt.axis('tight')

        plt.subplot(nsp, 1, 3)
        mt = np.arange(len(self.alpha))*(1.0 / self.sample_rate)
        plt.plot(mt, self.alpha, 'r-')
        #plt.plot(self.beta, 'b-')
        plt.legend(['alpha'])
        plt.axis('tight')

        plt.subplot(nsp, 1, 4)
        plt.plot(mt, self.sigma1, 'g-')
        plt.plot(mt, self.sigma2, 'b-')
        plt.plot(mt, self.mu, 'k-', linewidth=2)
        plt.legend(['sigma1', 'sigma2', 'mu'])
        plt.axis('tight')

        if hasattr(self, 'wav_file'):
            plt.subplot(nsp, 1, 5)
            print 'start_time=%f, end_time=%f' % (self.start_time, self.end_time)
            si = int(self.start_time * self.sample_rate)
            ei = int(self.end_time * self.sample_rate)
            plot_spectrogram(self.wav_file.spectrogram_t[si:ei], self.wav_file.spectrogram_f, self.wav_file.spectrogram[:, si:ei], fmin=0.0, fmax=8000.0)
            plt.title('Actual Spectrogram')

            """
            plt.subplot(nsp, 1, 5)
            plot_spectrogram(spec_t, self.hedi_spectrogram_freq, self.hedi_spectrogram, fmin=0.0, fmax=8000.0)
            plt.axis('tight')
            plt.title('Hedi Spectrogram')
            """


def motograms_from_file(file_name, wav_file_dir):

    hf = h5py.File(file_name, 'r')

    motograms = list()

    for md5,stim_group in hf['motograms'].iteritems():

        wav_file_name = os.path.join(wav_file_dir, '%s.wav' % md5)
        if not os.path.exists(wav_file_name):
            print 'No such file: %s' % wav_file_name
            continue

        intercept = stim_group.attrs['intercept']
        slope = stim_group.attrs['slope']

        wf = WavFile(wav_file_name, log_spectrogram=False)
        wf.analyze(min_freq=300, max_freq=8000, spec_sample_rate=1000.0, freq_spacing=125.0)

        for split_name,split_group in stim_group.iteritems():

            mgram = np.array(split_group['motogram'])
            time = mgram[:, 0]
            alpha = mgram[:, 1]
            beta = -1.0 * (slope*alpha + intercept)  # sign inversion from difference between Hedi's and Mike's models
            mu = mgram[:, 2]
            sigma1 = mgram[:, 3]
            sigma2 = mgram[:, 4]
            fitvals = mgram[:, 5]

            motogram = Motogram()
            motogram.alpha = alpha
            motogram.beta = beta
            motogram.mu = mu
            motogram.sigma1 = sigma1
            motogram.sigma2 = sigma2

            motogram.fitvals = fitvals
            motogram.hedi_spectrogram = np.array(split_group['spectrogram'])
            motogram.hedi_spectrogram_freq = np.array(split_group['frequencies'])
            motogram.start_time = split_group.attrs['start_time'] / 1000.0
            motogram.end_time = split_group.attrs['end_time'] / 1000.0
            motogram.md5 = md5
            motogram.wav_file = wf
            motogram.wav_file_name = wav_file_name

            motograms.append(motogram)

    hf.close()

    return motograms

