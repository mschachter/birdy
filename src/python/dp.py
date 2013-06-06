import copy
import numpy as np


class CostToGo(object):

    def __init__(self, system):
        self.system = system
        self.next_c2g = None

    def c2g(self, x, u):
        l = np.dot(x, x) + np.dot(u, u)
        if self.next_c2g is not None:
            xnext = self.system.next(x, u)
            l += self.next_c2g.optimal_c2g(xnext)
        return l

    def optimal_c2g(self, x):
        #interpolate lookup table to find the lowest cost for x
        pass

    def compute_optimal_cost2go(self, x):
        #find the optimal control at this time point by minimizing the cost-to-go
        pass

    def set_next(self, c2g_obj):
        self.next_c2g = c2g_obj


class CostToGoModel(object):

    def __init__(self, c2g_obj, x):
        self.c2g = c2g_obj
        self.x = x

    def error(self, u):
        pass

    def grad(self, u):
        pass

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


def finite_diff_grad(errorfunc, params, eps=1e-8):

    base_err = errorfunc(params)
    fdgrad = np.zeros(len(params))

    for k in range(len(params)):
        dparams = copy.deepcopy(params)
        dparams[k] += eps
        fdgrad[k] = (errorfunc(dparams) - base_err) / eps

    return fdgrad