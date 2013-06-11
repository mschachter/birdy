import h5py
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import time
from analysis import get_ff_rbf

from optimization import *
from oscillators import NormalOscillator


class CostToGo(object):

    def __init__(self, system, ff_rbf, ff_desired, dt=0.001, umin=-1.0, umax=1.0, ustep=0.25, bandwidth=0.30, ff_normalizer=10.0):

        self.system = system
        self.next_c2g = None
        self.dt = dt
        self.ff_rbf = ff_rbf
        self.ff_desired = ff_desired
        self.ff_normalizer = ff_normalizer

        u1 = np.arange(umin, umax+ustep, ustep)
        u2 = np.arange(umin, umax+ustep, ustep)
        U1,U2 = np.meshgrid(u1, u2)
        centers = np.array(zip(U1.ravel(), U2.ravel()))

        self.control = RBControlFunction(centers, bandwidths=bandwidth)

        #grid to evaluate RBF cost function
        self.alpha = np.arange(-1.25, -0.05, 0.05)
        self.beta = np.arange(-1.10, 1.10, 0.05)
        A, B = np.meshgrid(self.alpha, self.beta)
        self.rbf_eval_points = np.array(zip(A.ravel(), B.ravel()))
        self.rbf_num_eval_points = self.rbf_eval_points.shape[0]

    def plot(self):

        cost_xy = np.zeros([len(self.alpha), len(self.beta)])
        u1_xy = np.zeros([len(self.alpha), len(self.beta)])
        u2_xy = np.zeros([len(self.alpha), len(self.beta)])
        for i,alpha in enumerate(self.alpha):
            for j,beta in enumerate(self.beta):
                x = np.array([alpha, beta])
                u = self.optimal_control(x)
                u1_xy[i, j] = u[0]
                u2_xy[i, j] = u[1]
                cost_xy[i, j] = self.optimal_c2g(x)

        plt.subplot(2, 2, 1)
        plt.imshow(cost_xy, interpolation='nearest', aspect='auto', extent=[self.alpha.min(), self.alpha.max(), self.beta.min(), self.beta.max()])
        plt.title('Optimal Cost Surface')
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.plot(u1_xy.ravel(), u2_xy.ravel(), 'go')
        plt.axis('tight')
        plt.xlabel('Optimal Control 1')
        plt.ylabel('Optimal Control 2')

        plt.subplot(2, 2, 3)
        plt.imshow(u1_xy, interpolation='nearest', aspect='auto', extent=[self.alpha.min(), self.alpha.max(), self.beta.min(), self.beta.max()])
        plt.title('Optimal Control 1')
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(u2_xy, interpolation='nearest', aspect='auto', extent=[self.alpha.min(), self.alpha.max(), self.beta.min(), self.beta.max()])
        plt.title('Optimal Control 2')
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.colorbar()

    def c2g(self, x, u):

        #compute state at next time point
        xnext = self.system.rhs(x, u)*self.dt + x
        #compute cost of missing desired frequency at next time point
        fcost = ((self.ff_rbf(xnext[0], xnext[1]) - self.ff_desired) / self.ff_normalizer)**2
        #compute overall cost
        l = np.dot(x, x) + np.dot(u, u) + fcost
        #print '\t[c2g]: x=[%f, %f], u=[%f, %f], fcost=%f, l=%f' % (x[0], x[1], u[0], u[1], fcost, l)
        #add all the tail costs
        if self.next_c2g is not None:
            l += self.next_c2g.optimal_c2g(xnext)
        return l

    def optimal_control(self, x):
        return self.control.evaluate(x)

    def optimal_c2g(self, x):
        u = self.optimal_control(x)
        return self.c2g(x, u)

    def optimal_next(self, x):
        u = self.optimal_control(x)
        v = self.optimal_c2g(x)
        xnext = self.system.rhs(x, u)*self.dt + x
        return xnext,u,v

    def optimal_c2g_objective(self, C):
        stime = time.time()
        cost_sum = 0.0
        Cm = C.reshape(2, len(self.control.centers))
        for x in self.rbf_eval_points:
            u = self.control.evaluate(x, Cm)
            cost_sum += self.c2g(x, u)
        etime = time.time() - stime
        return cost_sum / self.rbf_num_eval_points

    def compute_optimal_control_rbf(self, local_step_size=1e-1, global_step_size=10.0, max_iter=10):
        #generate an initial guess by optimizing at each RBF center
        print 'Doing local pre-optimization to generate initial guess...'
        for k,ci in enumerate(self.control.centers):
            ui = self.compute_optimal_control_single(ci, step_size=local_step_size)
            ui_nan = np.isnan(ui)
            if ui_nan.sum() > 0:
                print 'nan found for optimal control at center (%f, %f)' % (ci[0], ci[1])
                ui[ui_nan] = 0.0
            self.control.C[:, k] = ui

        #do global optimization across RBF centers
        print 'Doing global optimization...'
        C0 = self.control.C.ravel()
        m = RBFCostToGoModel(self, C0)
        tgd = ThresholdGradientDescent(m, step_size=global_step_size, threshold=0.0, slope_threshold=1e-6)
        while not tgd.converged and tgd.iter < max_iter:
            tgd.iterate()
            print '[%d]: avg. cost=%0.6f' % (tgd.iter, tgd.errors[-1])
        Cstar = tgd.best_params
        self.control.C = Cstar.reshape(2, len(self.control.centers))

    def compute_optimal_control_single(self, x, step_size=1e-1, max_iter=1500):
        """
            Compute the optimal control at a single point.
        """
        m = SingleCostToGoModel(self, x, np.array([0.0, 0.0]))
        tgd = ThresholdGradientDescent(m, step_size=step_size, threshold=0.0, slope_threshold=-1e-6)
        while not tgd.converged and tgd.iter < max_iter:
            tgd.iterate()
            #print '[%d]: cost=%0.6f, u[0]=%f, u[1]=%f, slope=%f' % (tgd.iter, tgd.errors[-1], tgd.params[0], tgd.params[1], tgd.slope)
        ustar = tgd.best_params
        return ustar

    def set_next(self, c2g_obj):
        self.next_c2g = c2g_obj


class RBControlFunction(object):
    """ Nonparametric representation of feedback control for a single time point. Modeled as
        a combination of radial basis functions.
    """

    def __init__(self, centers, bandwidths, coefficients=None):
        self.M = len(centers)
        self.bandwidths = bandwidths
        self.centers = centers

        if coefficients is None:
            self.C = np.zeros([2, self.M])
        else:
            self.C = coefficients

        assert self.C.shape[0] == 2
        assert self.C.shape[1] == self.M

    def evaluate(self, x, C=None):

        Cused = self.C
        if C is not None:
            Cused = C

        r = ( (self.centers - x)**2 ).sum(axis=1)
        w = np.exp(-r / self.bandwidths**2)
        cc = Cused * w
        u = cc.sum(axis=1)

        return u


class SingleCostToGoModel(object):

    def __init__(self, c2g_obj, x, u0):
        self.c2g = c2g_obj
        self.params = u0
        self.x = x

    def error(self, u):
        return self.c2g.c2g(self.x, u)

    def grad(self, u):
        return finite_diff_grad(self.error, u, eps=1e-8)


class RBFCostToGoModel(object):

    def __init__(self, c2g_obj, C0):
        self.c2g = c2g_obj
        self.params = C0

    def error(self, C):
        return self.c2g.optimal_c2g_objective(C)

    def grad(self, C):
        return finite_diff_grad(self.c2g.optimal_c2g_objective, C)


class LinearSystem(object):

    def __init__(self, A, center=np.array([0.0, 0.0])):
        self.A = A
        self.center = center

    def rhs(self, x, u=0.0):
        xnew = np.dot(self.A, x-self.center) + u
        return xnew

    def rhs_scipy(self, t, y):
        return self.rhs(y)

    def simulate(self, x0, duration, dt=0.001):

        r = ode(self.rhs_scipy)
        r.set_initial_value(x0, 0.0)
        traj = list()
        while r.successful() and r.t < duration:
            r.integrate(r.t + dt)
            traj.append(r.y)
        return np.array(traj)

    def phase_plot(self, alphamin=-1.25, alphamax=0.05, betamin=-1.25, betamax=1.25, nb_points=20):
        fig = plt.figure()
        x = np.linspace(alphamin, alphamax, nb_points)
        y = np.linspace(betamin, betamax, nb_points)

        X, Y = np.meshgrid(x, y)
        DX = np.zeros([len(x), len(y)])
        DY = np.zeros([len(x), len(y)])
        for k in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[k, j], Y[k, j]])
                dx,dy = self.rhs(x)
                DX[k, j] = dx
                DY[k, j] = dy

        M = (np.hypot(DX, DY))
        M[ M == 0] = 1.
        DX /= M
        DY /= M

        plt.title('Meta-control Phase Plot')
        Q = plt.quiver(X, Y, DX, DY, M, pivot='mid', cmap=plt.cm.jet)
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.plot(self.center[0], self.center[1], 'ro')
        plt.legend()
        plt.grid()


class ControlSystem(object):

    def __init__(self, ff_desired, dv_file='/home/cheese63/birdy/data/admissible_controls_small.h5', control_file=None):

        self.system = LinearSystem(np.array([[-1000.0, 0], [0.0, -900.0]]), center=np.array([-0.30, 0.30]))
        self.ff_rbf = get_ff_rbf(dv_file, plot=False, bandwidth=0.08)
        self.c2g_chain = list()
        self.ff_desired = ff_desired
        nsteps = len(ff_desired)
        for k in range(nsteps):
            desired_freq = ff_desired[nsteps-k-1]
            c2g = CostToGo(self.system, self.ff_rbf, desired_freq)
            if k > 0:
                c2g.set_next(self.c2g_chain[-1])
            self.c2g_chain.append(c2g)

        if control_file is None:
            for k,c2g in enumerate(self.c2g_chain):
                stime = time.time()
                print 'Computing optimal cost to go for time %d' % (nsteps - k - 1)
                c2g.compute_optimal_control_rbf()
                etime = time.time() - stime
                print 'Elapsed time: %0.2f min' % (etime / 60.0)
        else:
            hf = h5py.File(control_file, 'r')
            nsteps = len(hf.keys())
            for k in range(nsteps):
                index = nsteps - k - 1
                key = '%d' % index
                grp = hf[key]
                c2g = self.c2g_chain[index]
                c2g.control.C = np.array(grp['C'])
                c2g.control.centers = np.array(grp['centers'])
                c2g.control.bandwidths = np.array(grp['bandwidths'])
            hf.close()

    def save(self, output_file):
        nsteps = len(self.c2g_chain)
        hf = h5py.File(output_file, 'w')
        for k,c2g in enumerate(self.c2g_chain):
            grp = hf.create_group('%d' % (nsteps - k - 1))
            grp['C'] = c2g.control.C
            grp['bandwidths'] = c2g.control.bandwidths
            grp['centers'] = c2g.control.centers
            grp['ff_desired'] = c2g.ff_desired
        hf.close()

    def run(self, x0):

        states = list()
        states.append(x0)
        controls = list()
        costs = list()
        for k,c2g in enumerate(self.c2g_chain[::-1]):
            x = states[-1]
            xnext,u,v = c2g.optimal_next(x)
            costs.append(v)
            controls.append(u)
            states.append(xnext)

        return np.array(states),np.array(controls),np.array(costs)

    def control_oscillator(self, states, dt=1e-3, oscillator_dt=1e-6):

        no = NormalOscillator()

        output = list()

        oscillator_x = 0.0
        oscillator_v = 0.0

        for alpha,beta in states:
            params = {'alpha':alpha, 'beta':beta}
            ostates = no.run_simulation(params, dt, oscillator_dt, initial_x=oscillator_x, initial_v=oscillator_v)
            oscillator_x = ostates[-1, 0]
            oscillator_v = ostates[-1, 1]
            output.extend(ostates[:, 0])

        return output