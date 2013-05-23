from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from Cython.Build import cythonize

"""
ext_modules = [
    Extension('oscillators',
              ['oscillators.pyx'],
              libraries=['m', 'cblas', 'gsl'])
]

setup(
    name = "Oscillators",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
"""

"""
setup(
    name = "oscillators",
    ext_modules = cythonize(
        'oscillators.pyx',
        sources=['../cpp/physical_oscillator.cpp'],
        language='c++',
        libraries=['oscillator'],
        library_dirs=['../cpp']
    )
)
"""

sourcefiles = ['oscillators.pyx', '../cpp/physical_oscillator.cpp']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('oscillators', sourcefiles, libraries=['cblas', 'gsl', 'm'])]
)