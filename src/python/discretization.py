import glob
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tools.sound import WavFile,log_spectrogram


class PCADiscretizer(object):

    def __init__(self, spec_sample_rate=1000.0, spec_freq_spacing=125.0, min_freq=500.0, max_freq=8000.0):
        self.files = list()
        self.samples = list()
        self.spec_sample_rate = spec_sample_rate
        self.spec_freq_spacing = spec_freq_spacing
        self.min_freq = min_freq
        self.max_freq = max_freq

    def add_wav_file(self, file_name, rms_thresh=40.0):
        wf = WavFile(file_name)
        t,f,spec,spec_rms = log_spectrogram(wf.data, wf.sample_rate, spec_sample_rate=self.spec_sample_rate,
                                            freq_spacing=self.spec_freq_spacing, min_freq=self.min_freq, max_freq=self.max_freq)
        self.spec_t = t
        self.spec_f = f
        for spectrum in spec[:, spec_rms < rms_thresh].transpose():
            self.samples.append(spectrum)

    def add_directory(self, dir_name):
        for fname in glob.glob(os.path.join(dir_name, '*.wav')):
            self.add_wav_file(fname)

    def discretize(self, variance_to_capture=0.80, num_clusters=50):
        self.samples = np.array(self.samples)

        #use PCA to project the samples into a lower dimensionality
        self.pca = PCA(n_components=variance_to_capture)
        self.pca.fit(self.samples)
        self.transformed_samples = self.pca.transform(self.samples)

        #cluster the transformed samples
        self.clusters = self.cluster_with_kmeans(num_clusters)

    def cluster_with_kmeans(self, num_clusters):
        #use K-means to cluster the components
        kmean = KMeans(num_clusters, init='k-means++', n_init=25, max_iter=10000, precompute_distances=True)
        kmean.fit(self.transformed_samples)

        clusters = kmean.predict(self.transformed_samples)
        cdict = dict()
        for k,c in enumerate(clusters):
            if c not in cdict:
                cdict[c] = list()
            cdict[c].append(k)
        return cdict

    def plot(self, num_clusters_to_plot=10):
        for c in range(num_clusters_to_plot):
            sample_indices = self.clusters[c]
            nsamps = min(15, len(sample_indices))
            dlist = list()
            for k in range(nsamps):
                samp_index = sample_indices[k]
                dlist.append( {'f':self.spec_f, 'spectrum':self.samples[samp_index, :]} )
            multi_plot(dlist, plot_spectrum_single, title='Cluster %d' % c, nrows=5, ncols=3)


def plot_spectrum_single(pdata, ax):
    plt.plot(pdata['f'], pdata['spectrum'], 'k-')
    plt.xticks([])
    plt.yticks([])


def multi_plot(data_list, plot_func, title=None, nrows=4, ncols=5):

    nsp = 0
    fig = None
    plots_per_page = nrows*ncols
    for pdata in data_list:
        if nsp % plots_per_page == 0:
            fig = plt.figure()
            fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20)
            if title is not None:
                plt.suptitle(title)

        nsp += 1
        sp = nsp % plots_per_page
        ax = fig.add_subplot(nrows, ncols, sp)
        plot_func(pdata, ax)