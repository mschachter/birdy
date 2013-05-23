"""
    A cython interface to the C++ oscillator code.
"""

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cdef double** npy2c_double2d(np.ndarray[double, ndim=2] a):
    cdef double** a_c = <double**> malloc(a.shape[0] * sizeof(double*))
    for k in range(a.shape[0]):
        a_c[k] = &a[k, 0]
    return a_c

cdef extern from "numpy/arrayobject.h":
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags

cdef extern from "../cpp/physical_oscillator.h":
    ctypedef struct PhysicalParams:
        pass
    PhysicalParams* physical_oscillator_init(double k1, double psub, double f0)
    void physical_oscillator_run(double** output, double* initial_state, double duration, double dt, PhysicalParams* pp)

cdef extern from "../cpp/normal_oscillator.h":
    ctypedef struct NormalParams:
        pass
    NormalParams* normal_oscillator_init(double alpha, double beta)
    void normal_oscillator_run(double** output, double* initial_state, double duration, double dt, NormalParams* pp)

cdef class PhysicalOscillator:

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    cpdef simulate(self, double initial_x, double initial_v, double duration, double dt,
                         double k1 = 0.016, double psub = 1900.0, double f0 = 0.0399):

        #initialize model and control parameters
        cdef PhysicalParams* pp = physical_oscillator_init(k1, psub, f0)

        #create initial state vector
        cdef np.ndarray istate = np.zeros([2], dtype=np.double)
        istate[0] = initial_x
        istate[1] = initial_v

        #create output array that contains the 2D state
        cdef int nsteps = int(duration / dt) + 1
        cdef np.ndarray output = np.zeros([nsteps, 2], dtype=np.double)

        #run the simulation and return the output state
        physical_oscillator_run(npy2c_double2d(output), <double*>istate.data, duration, dt, pp)
        return output

cdef class NormalOscillator:

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    cpdef simulate(self, double initial_x, double initial_v, double duration, double dt,
                         double alpha=-0.41769, double beta=-0.346251775):

        #initialize model and control parameters
        cdef NormalParams* pp = normal_oscillator_init(alpha, beta)

        #create initial state vector
        cdef np.ndarray istate = np.zeros([2], dtype=np.double)
        istate[0] = initial_x
        istate[1] = initial_v

        #create output array that contains the 2D state
        cdef int nsteps = int(duration / dt) + 1
        cdef np.ndarray output = np.zeros([nsteps, 2], dtype=np.double)

        #run the simulation and return the output state
        normal_oscillator_run(npy2c_double2d(output), <double*>istate.data, duration, dt, pp)
        return output