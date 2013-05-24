import numpy as np

import matplotlib.pyplot as plt


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
