import h5py
import numpy as np

from scipy.optimize import newton

import matplotlib.pyplot as plt
import operator

from oscillators import NormalOscillator,PhysicalOscillator

def plot_trajectory(traj, dt):

    x = traj[:, 0]
    v = traj[:, 1]
    t = np.arange(traj.shape[0])*dt

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'k-')
    plt.title('x(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, v, 'b-')
    plt.title('v(t)')


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

def find_fixedpoints_grid_normal(output_file=None, xmin=-10.0, xmax=10.0, xstep=1e-3, alphamin=-0.45, alphamax=0.45, alphastep=0.001, betamin=-0.40, betamax=0.20, betastep=0.001):


    alpharng = np.arange(alphamin, alphamax, alphastep)
    betarng = np.arange(betamin, betamax, betastep)

    nrows = len(alpharng)
    ncols = len(betarng)
    print '# of (alpha,beta) pairs: %d' % (nrows*ncols)

    all_pairs = np.zeros([nrows, ncols, 2])
    all_roots = np.zeros([nrows, ncols, 3, 2]) * np.nan

    for i,alpha in enumerate(alpharng):
        for j,beta in enumerate(betarng):
            roots = find_fixedpoints_normal(xmin=xmin, xmax=xmax, xstep=xstep, alpha=alpha, beta=beta, plot=False)
            all_pairs[i, j, :] = [alpha, beta]
            for k,(xz,xz_val) in enumerate(roots):
                all_roots[i, j, k, :] = [xz, xz_val]

    if output_file is not None:
        hf = h5py.File(output_file, 'w')
        hf['all_pairs'] = all_pairs
        hf['all_roots'] = all_roots
        hf.close()













