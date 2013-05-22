import numpy as np

from scipy.integrate import ode as sciode

import matplotlib.pyplot as plt


class SittOscillator(object):
    """
        Model of zebra finch vocal cords from:
        "Physiologically driven avian vocal synthesizer"
        Jacobo D. Sitt, Ezequiel M. Arneodo, Franz Goller, Gabriel B. Mindlin
        PHYSICAL REVIEW E 81, 031927 (2010)
    """

    def __init__(self, initial_state=[0.0, 0.0], alpha=-0.41769, beta=-0.346251775, gamma=23500.0, dt=1.0 / 44100.0):

        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.state = initial_state
        self.t = 0
        self.dt = dt

        self.ode = sciode(self.rhs, self.jacobian)
        #self.ode.set_integrator('dop853', first_step=1e-10, nsteps=1000)
        self.ode.set_integrator('vode', method='bdf', order=15, nsteps=3000)
        self.ode.set_initial_value(self.state, self.t*self.dt)

    def rhs(self, t, state):

        x,v = state
        #print 't=%0.6f, x=%0.6f, v=%0.3f' % (t, x, v)
        x2 = x**2
        x3 = x**3
        g = self.gamma
        g2 = self.gamma**2
        a = self.alpha
        b = self.beta

        dx = v
        dv = g2*a + g2*b*x + g2*x2 - g*x*v - g*x3 - g*x2*v
        #dv = g2*a - g2*b*x + g2*x2 - g*x*v - g*x3 - g*x2*v  # Hedi's model

        return np.array([dx, dv])

    def jacobian(self, t, state):

        x,v = state
        x2 = x**2
        g = self.gamma
        g2 = g**2
        a = self.alpha
        b = self.beta

        j11 = 0.0
        j12 = 1.0
        j21 = g2*b + 2*g2*x - g*v - 3*g*x2 - 2*g*x*v
        #j21 = -g2*b + 2*g2*x - g*v - 3*g*x2 - 2*g*x*v  # Hedi's model
        j22 = -g*x - g*x2

        return np.array([[j11, j12], [j21, j22]])

    def step(self, ):
        self.t += 1
        self.ode.integrate(self.t*self.dt)
        if not self.ode.successful():
            raise Exception('SciPy ODE solver failed at t=%0.6f' % (self.t*self.dt))
        self.state = self.ode.y

    def simulate(self, duration=0.001):
        nsteps = int(duration / self.dt)
        trajectory = np.zeros([nsteps+1, 2])
        trajectory[0, :] = self.state
        for k in range(nsteps):
            self.step()
            trajectory[k+1, :] = self.state
        return trajectory

    def plot_trajectory(self, traj):

        x = traj[:, 0]
        v = traj[:, 1]
        t = np.arange(traj.shape[0])*self.dt

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, x, 'k-')
        plt.title('x(t)')
        plt.subplot(2, 1, 2)
        plt.plot(t, v, 'b-')
        plt.title('v(t)')


class SittVocalTract(object):

    def __init__(self, L=0.019, r=-0.9, c=340.0, sample_rate=44100.0):
        self.L = L
        self.r = r
        self.c = c
        self.sample_rate = sample_rate
        self.tpast = int(((2.0*self.L) / self.c) * self.sample_rate)

    def simulate(self, x):
        pi = np.zeros([len(x)])
        po = np.zeros([len(x)])
        for t in range(len(x)):
            pi_past = 0.0
            if t >= self.tpast:
                pi_past = pi[t-self.tpast]
            pi[t] = x[t] - self.r*pi_past
            po[t] = (1.0 - self.r)*pi_past
        return po
