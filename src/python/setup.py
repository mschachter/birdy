from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sourcefiles = ['oscillators.pyx', '../cpp/common.cpp', '../cpp/physical_oscillator.cpp', '../cpp/normal_oscillator.cpp']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('oscillators', sourcefiles, libraries=['gsl', 'gslcblas', 'm'], language='c++')]
)
