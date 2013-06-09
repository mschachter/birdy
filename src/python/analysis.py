import h5py
import numpy as np

from scipy.optimize import newton
from scipy.fftpack import fft,fftfreq
from scipy.interpolate import Rbf

import matplotlib.pyplot as plt
import operator
import time

from oscillators import NormalOscillator,PhysicalOscillator


def plot_trajectory(traj, dt):

    x = traj[:, 0]
    v = traj[:, 1]
    t = np.arange(traj.shape[0])*dt

    fftx = fft(x)
    f = fftfreq(len(x), d=dt)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, x, 'k-')
    plt.title('x(t)')
    plt.subplot(3, 1, 2)
    plt.plot(t, v, 'b-')
    plt.title('v(t)')
    plt.subplot(3, 1, 3)

    findx = (f > 100.0) & (f < 12000.0)
    plt.plot(f[findx], np.log10(np.abs(fftx[findx])), 'k-')
    plt.title('Power Spectrum')
    plt.axis('tight')


def find_fixedpoints_normal(xmin=-10.0, xmax=10.0, xstep=1e-3, alpha=-0.41769, beta=-0.346251775, plot=False):

    no = NormalOscillator()

    xrng = np.arange(xmin, xmax+xstep, xstep)
    nullx = np.array([no.nullcline_x(xval, alpha, beta) for xval in xrng])

    xp = zip(xrng, nullx, np.abs(nullx))
    xp.sort(key=operator.itemgetter(-1))

    top3 = np.array(xp[:3])

    f = lambda x: no.nullcline_x(x, alpha, beta)
    df = lambda x: no.nullcline_dx(x, alpha, beta)

    zero_tol = 1e-6
    x_tol = 1e-4
    roots = list()
    for k,xi in enumerate(top3[:, 0]):
        try:
            xz = newton(f, xi, fprime=df, maxiter=100)
        except RuntimeError:
            continue
        xz_val = no.nullcline_x(xz, alpha, beta)
        #print 'xz=%0.9f, xz_val=%0.9f' % (xz, xz_val)
        if np.abs(xz_val) < zero_tol:
            is_duplicate = False
            for x,xv in roots:
                if np.abs(x - xz) < x_tol:
                    is_duplicate = True
            if not is_duplicate:
                #print 'Root: x=%0.6f, f(x)=%0.6f' % (xz, xz_val)
                roots.append([xz, xz_val])
    roots = np.array(roots)

    if plot:
        plt.figure()
        plt.plot(xrng, nullx, 'k-')
        plt.plot(top3[:, 0], top3[:, 1], 'rx')
        plt.plot(roots[:, 0], roots[:, 1], 'go')
        plt.title('Fixed Point Surface for dv')
        plt.xlabel('x')
        plt.axis('tight')

    return roots


def find_fixedpoints_grid_normal(output_file=None, xmin=-10.0, xmax=10.0, xstep=1e-3, alphamin=-1.00, alphamax=0.05, alphastep=0.01, betamin=-0.60, betamax=0.60, betastep=0.01):

    stime = time.time()
    alpharng = np.arange(alphamin, alphamax, alphastep)
    betarng = np.arange(betamin, betamax, betastep)

    nrows = len(alpharng)
    ncols = len(betarng)
    print '# of (alpha,beta) pairs: %d' % (nrows*ncols)

    total_mem = ((nrows*ncols*2)+(nrows*ncols*3*2)*8.0) / 1024.0**2
    print 'Total Memory: %0.0f MB' % total_mem

    all_pairs = np.zeros([nrows, ncols, 2])
    all_roots = np.zeros([nrows, ncols, 3, 2]) * np.nan

    for i,alpha in enumerate(alpharng):
        for j,beta in enumerate(betarng):
            roots = find_fixedpoints_normal(xmin=xmin, xmax=xmax, xstep=xstep, alpha=alpha, beta=beta, plot=False)
            all_pairs[i, j, :] = [alpha, beta]
            for k,(xz,xz_val) in enumerate(roots):
                all_roots[i, j, k, :] = [xz, xz_val]
    etime = time.time() - stime
    print 'Elapsed Time: %0.2f s' % etime

    if output_file is not None:
        hf = h5py.File(output_file, 'w')
        hf['all_pairs'] = all_pairs
        hf['all_roots'] = all_roots
        hf.close()


def find_admissible_controls(output_file=None, alphamin=-1.25, alphamax=0.25, alphastep=0.01, betamin=-1.50, betamax=1.50, betastep=0.01):

    no = NormalOscillator()

    stime = time.time()
    alpharng = np.arange(alphamin, alphamax, alphastep)
    betarng = np.arange(betamin, betamax, betastep)

    nrows = len(alpharng)
    ncols = len(betarng)
    print '# of (alpha,beta) pairs: %d' % (nrows*ncols)

    all_pairs = np.zeros([nrows, ncols, 2])
    all_dv_rms = np.zeros([nrows, ncols]) * np.nan
    all_ff = np.zeros([nrows, ncols]) * np.nan

    sim_duration = 0.010
    step_size = 1e-6
    steady_state_point = 0.005
    steady_state_index = int(steady_state_point / step_size)

    for i,alpha in enumerate(alpharng):
        for j,beta in enumerate(betarng):
            all_pairs[i, j, :] = [alpha, beta]
            output = no.simulate(0.0, 0.0, duration=sim_duration, dt=step_size, alpha=alpha, beta=beta)
            dv = np.diff(output[:, 1])
            dv_rms = dv[steady_state_index:].std(ddof=1)
            all_dv_rms[i, j] = dv_rms

            #compute power spectrum
            fftx = fft(output[:, 0])
            ps_f = fftfreq(len(output[:, 0]), d=step_size)
            findx = (ps_f > 100.0) & (ps_f < 8000.0)

            #estimate fundamental frequency from log power spectrum in the simplest way possible
            ps = np.abs(fftx[findx])
            peak_index = ps.argmax()
            all_ff[i, j] = ps_f[findx][peak_index]

    etime = time.time() - stime
    print 'Elapsed Time: %0.2f s' % etime

    if output_file is not None:
        hf = h5py.File(output_file, 'w')
        hf['all_pairs'] = all_pairs
        hf['all_dv_rms'] = all_dv_rms
        hf['all_ff'] = all_ff
        hf.close()


def get_ff_rbf(dv_file, dv_rms_thresh=1e-2, plot=False, bandwidth=0.08):

    hf = h5py.File(dv_file, 'r')
    all_pairs = np.array(hf['all_pairs'])
    all_dv_rms = np.array(hf['all_dv_rms'])
    all_ff = np.array(hf['all_ff'])
    hf.close()

    alpha = all_pairs[:, :, 0]
    beta = all_pairs[:, :, 1]

    all_ff[all_dv_rms < dv_rms_thresh] = 0.0

    ap = np.array(zip(alpha.ravel(), beta.ravel()))
    x = ap[:, 0]
    y = ap[:, 1]
    d = all_ff.ravel()

    ff_rbf = Rbf(x, y, d, function='gaussian', epsilon=bandwidth)

    if plot:
        alpha_rng = np.arange(-1.25, 0.25, 0.01)
        beta_rng = np.arange(-1.50, 1.50, 0.01)
        interp_ff = np.zeros([len(alpha_rng), len(beta_rng)])
        for i,alpha in enumerate(alpha_rng):
            for j,beta in enumerate(beta_rng):
                interp_ff[i, j] = ff_rbf(alpha, beta)
        interp_ff[interp_ff < 0.0] = 0.0

        plt.figure()
        plt.imshow(all_ff, interpolation='nearest', aspect='auto', extent=[alpha_rng.min(), alpha_rng.max(), beta_rng.min(), beta_rng.max()])
        plt.colorbar()
        plt.xlabel('Beta')
        plt.ylabel('Alpha')
        plt.title('Fundamental Frequency')

        plt.figure()
        plt.imshow(interp_ff, interpolation='nearest', aspect='auto', extent=[alpha_rng.min(), alpha_rng.max(), beta_rng.min(), beta_rng.max()])
        plt.colorbar()
        plt.xlabel('Beta')
        plt.ylabel('Alpha')
        plt.title('Interpolated Fundamental Frequency')

    return ff_rbf


def plot_ff(dv_file, dv_rms_thresh=1e-2):

    hf = h5py.File(dv_file, 'r')
    all_pairs = np.array(hf['all_pairs'])
    all_dv_rms = np.array(hf['all_dv_rms'])
    all_ff = np.array(hf['all_ff'])
    hf.close()

    alpha = all_pairs[:, :, 0]
    beta = all_pairs[:, :, 1]

    all_ff[all_dv_rms < dv_rms_thresh] = 0.0

    alpha_y = alpha[:, 0]
    beta_x = beta[0, :]
    ytick_rng = range(0, len(alpha_y), 8)
    xtick_rng = range(0, len(beta_x), 8)

    plt.figure()
    plt.imshow(all_ff, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xticks(xtick_rng, ['%0.2f' % x for x in beta_x[xtick_rng]])
    plt.yticks(ytick_rng, ['%0.2f' % y for y in alpha_y[ytick_rng]])
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.title('Fundamental Frequency')


def plot_fixedpoints(fp_file):
    hf = h5py.File(fp_file, 'r')
    all_pairs = np.array(hf['all_pairs'])
    all_roots = np.array(hf['all_roots'])
    hf.close()

    alpha = all_pairs[:, :, 0]
    beta = all_pairs[:, :, 1]

    fixed_points = all_roots[:, :, :, 0]
    num_fixed_points = (~np.isnan(fixed_points)).sum(-1)

    alpha_y = alpha[:, 0]
    beta_x = beta[0, :]
    ytick_rng = range(0, len(alpha_y), 8)
    xtick_rng = range(0, len(beta_x), 8)

    plt.figure()
    plt.imshow(num_fixed_points, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xticks(xtick_rng, ['%0.2f' % x for x in beta_x[xtick_rng]])
    plt.yticks(ytick_rng, ['%0.2f' % y for y in alpha_y[ytick_rng]])
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.title('# of Fixed Points')

    return alpha,beta,fixed_points,num_fixed_points
