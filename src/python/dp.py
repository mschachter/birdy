import copy
import operator

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt


class CostToGo(object):

    def __init__(self, system):
        self.system = system
        self.next_c2g = None

    def c2g(self, x, u):
        l = np.dot(x, x) + np.dot(u, u)
        if self.next_c2g is not None:
            xnext = self.system.rhs(x, u)
            l += self.next_c2g.optimal_c2g(xnext)
        return l

    def optimal_c2g(self, x):
        #interpolate lookup table to find the lowest cost for x
        pass

    def compute_optimal_control(self, x):
        #find the optimal control at this time point by minimizing the cost-to-go
        m = CostToGoModel(self, x)
        tgd = ThresholdGradientDescent(m, step_size=1e-6, threshold=0.0)
        niter = 100
        for k in range(niter):
            tgd.iterate()
        optimal_u = tgd.best_params
        return optimal_u

    def set_next(self, c2g_obj):
        self.next_c2g = c2g_obj


class CostToGoModel(object):

    def __init__(self, c2g_obj, x):
        self.c2g = c2g_obj
        self.x = x

    def error(self, u):
        return self.c2g.c2g(self.x, u)

    def grad(self, u):
        return finite_diff_grad(self.error, u)


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

    def phase_plot(self, alphamin=-1.25, alphamax=0.05, betamin=-1.25, betamax=1.25):
        fig = plt.figure()
        """
        values = np.linspace(0.3, 0.9, 5)                          # position of X0 between X_f0 and X_f1
        vcolors = plt.cm.autumn_r(np.linspace(0.3, 1., len(values)))  # colors for each trajectory
        # plot trajectories
        for v, col in zip(values, vcolors):
            X0 = v * X_f1                               # starting point
            X = integrate.odeint( dX_dt, X0, t)         # we don't need infodict here
            plt.plot( X[:,0], X[:,1], lw=3.5*v, color=col, label='X0=(%.f, %.f)' % ( X0[0], X0[1]) )
        """

        nb_points = 30

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


class ThresholdGradientDescent(object):

    def __init__(self, model, step_size=1e-3, threshold=1.0, earlystop_model=None, gradient_norm=True, group_indices=None, slope_threshold=-1e-3):
        self.threshold = threshold
        self.model = model
        self.step_size = step_size
        self.earlystop_model = earlystop_model
        self.errors = list()
        self.es_errors = list()
        self.params = model.params
        self.gradient_norm = gradient_norm
        self.converged = False
        self.num_iters_for_slope = 5
        self.slope = -np.Inf
        self.slope_threshold = slope_threshold
        self.iter = 0
        self.group_indices = group_indices
        self.groups = None
        self.best_params = None
        self.best_err = np.inf

        if self.group_indices is not None:
            self.groups = np.unique(self.group_indices)
            if len(threshold) > 1 and len(threshold) != len(self.groups):
                raise Exception('Number of thresholds specified must equal the number of unique groups!')
            self.group_indices_map = dict()
            for g in self.groups:
                self.group_indices_map[g] = np.where(self.group_indices == g)

    def iterate(self):

        #compute gradient, update parameters
        g = self.model.grad(self.params)

        #threshold out elements of the gradient
        if not np.isscalar(self.threshold):
            for k,(group,gindex) in enumerate(self.group_indices_map.iteritems()):
                gsub = g[gindex]
                if self.gradient_norm:
                    gsub /= np.linalg.norm(gsub)
                gabs = np.abs(gsub)
                gthresh = gabs.max()*self.threshold[k]
                gsub[gabs < gthresh] = 0.0
                g[gindex] = gsub

        else:
            if self.gradient_norm:
                g /= np.linalg.norm(g)
            gabs = np.abs(g)
            gthresh = gabs.max()*self.threshold
            g[gabs < gthresh] = 0.0

        self.params = self.params - self.step_size*g

        #compute error, check for convergence
        e = self.model.error(self.params)
        self.errors.append(e)

        if self.earlystop_model is not None:
            es_err = self.earlystop_model.error(self.params)
            self.es_errors.append(es_err)

            if es_err < self.best_err:
                self.best_err = es_err
                self.best_params = copy.copy(self.params)

            if len(self.es_errors) >= self.num_iters_for_slope:
                slope,intercept = np.polyfit(range(self.num_iters_for_slope), self.es_errors[-self.num_iters_for_slope:], 1)
                slope /= np.abs(np.array(self.es_errors[-self.num_iters_for_slope:]).mean())
                self.slope = slope
                if self.slope > self.slope_threshold:
                    self.converged = True
        else:
            if len(self.errors) >= self.num_iters_for_slope:
                slope,intercept = np.polyfit(range(self.num_iters_for_slope), self.errors[-self.num_iters_for_slope:], 1)
                slope /= np.abs(np.array(self.errors[-self.num_iters_for_slope:]).mean())
                self.slope = slope
                if self.slope > self.slope_threshold:
                    self.converged = True

                if e < self.best_err:
                    self.best_err = e
                self.best_params = copy.copy(self.params)

        self.iter += 1


def generate_stable_linear_systems(plot=True, amin=-1.0, amax=1.0, dmin=-1.0, dmax=1.0):
    """
        Generates linear systems of the form:
        [[1.0 a]
         [a   d]]
        that has real-valued negative eigenvalues.

        Returns a list of matrices, ordered by the norm of [eigenvalue1, eigenvalue2] (fastest to slowest decay)
    """
    a = np.arange(amin, amax, 0.05)
    d = np.arange(dmin, dmax, 0.05)

    A,D = np.meshgrid(a, d)

    #compute eigenvalues
    e1 = D-1.0 + np.sqrt(D**2 + 4*A**2 + 2*D + 1.0)
    e2 = D-1.0 - np.sqrt(D**2 + 4*A**2 + 2*D + 1.0)

    #find points where both eigenvalues are negative
    nindx = (e1 < 0.0) & (e2 < 0.0)

    #compute magnitude of eigenvalues
    evals = np.array(zip(e1[nindx], e2[nindx]))

    mtudes = [np.linalg.norm(x) for x in evals]
    pts = zip(A[nindx], D[nindx])

    #sort values of a and d by the magnitude of eigenvalues
    all_vals = [(pts[k], mtudes[k]) for k in range(len(mtudes))]
    all_vals.sort(key=operator.itemgetter(-1), reverse=True)

    #construct matrices
    systems = list()
    for (a,d),mtude in all_vals:
        M = np.array([[-1.0, a], [a, d]])
        systems.append(M)

    if plot:
        pe1 = copy.copy(e1)
        pe1[~nindx] = np.nan
        pe2 = copy.copy(e2)
        pe2[~nindx] = np.nan

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(np.real(pe1), interpolation='nearest', aspect='auto', extent=[amin, amax, dmin, dmax])
        plt.colorbar()
        plt.title('$\lambda_1$')
        plt.subplot(2, 1, 2)
        plt.imshow(np.real(pe2), interpolation='nearest', aspect='auto', extent=[amin, amax, dmin, dmax])
        plt.colorbar()
        plt.title('$\lambda_2$')

    return systems


def finite_diff_grad(errorfunc, params, eps=1e-8):

    base_err = errorfunc(params)
    fdgrad = np.zeros(len(params))

    for k in range(len(params)):
        dparams = copy.deepcopy(params)
        dparams[k] += eps
        fdgrad[k] = (errorfunc(dparams) - base_err) / eps

    return fdgrad