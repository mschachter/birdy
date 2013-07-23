import os
import h5py
import numpy as np
from tools.signal import gaussian_stft

from tools.sound import log_spectrogram,WavFile

from oscillators import NormalOscillator


class Motogram(object):

    def __init__(self):
        self.oscillator = NormalOscillator()
        self.sample_rate = 1000.0
        self.waveform_sample_rate = 1e6
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

        #simulate the oscillator to produce the sound pressure waveform
        steps_per_step = int(step_duration / dt)
        last_index = 0
        last_x = 0.0
        last_v = 0.0
        for k in range(nsteps):
            states = self.oscillator.simulate(last_x, last_v, step_duration, dt, alpha=self.alpha[k], beta=self.beta[k])
            print 'k=%d, last_x=%f, last_v=%f, step_duration=%0.6f, dt=%0.6f, alpha=%0.6f, beta=%0.6f' % (k, last_x, last_v, step_duration, dt, self.alpha[k], self.beta[k])
            print 'states.shape=',states.shape
            si = last_index
            ei = last_index + steps_per_step
            waveform[si:ei] = states[:steps_per_step, 0]

            last_x = states[steps_per_step, 0]
            last_v = states[steps_per_step, 1]
            if np.isnan(last_x):
                last_x = 0.0
            if np.isnan(last_v):
                last_v = 0.0
            last_index = ei

        #transform the waveform into a spectrogram
        nstd = 6.0
        increment = 1.0 / 1000.0
        window_length = nstd / (2.0*np.pi*125.0)
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
            mean2 = mu_t - spec_f
            mean2[mean2 < 0.0] = 0.0
            gauss2 = mean2**2 / (2*sigma2_t**2)

            #create gaussian based on sigma1
            mean1 = spec_f - mu_t
            mean1[mean1 < 0.0] = 0.0
            gauss1 = mean1**2 / (2*sigma1_t**2)

            #create filter as combination of two gaussians
            filt = np.exp(-(gauss1 + gauss2))
            print list(filt)
            print ''

            #filter the spectrogram
            spec_filt[:, k] = spec[:, k]*filt

        return spec_filt


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

        wf = WavFile(wav_file_name)
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
            motogram.start_time = split_group.attrs['start_time']
            motogram.end_time = split_group.attrs['end_time']
            motogram.md5 = md5
            motogram.wav_file = wf

            motograms.append(motogram)

    hf.close()

    return motograms

