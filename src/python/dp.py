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


