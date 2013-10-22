# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:52:24 2013

@author: dgevans
"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext = Extension('primitives.CES_parameters_c', ['primitives/CES_parameters_c.pyx'],
include_dirs=[numpy.get_include()]
)

setup(
    name = 'CES_parameters_c',
    ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
)

ext = Extension('distributions.lognormal', ['distributions/lognormal.pyx'],
include_dirs=['/opt/local/include/',numpy.get_include()],
library_dirs=["/opt/local/lib"],
libraries=["gsl","gslcblas"]
)

setup(
    name = 'lognormal',
    ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
)

ext = Extension('distributions.lognormal_pareto', ['distributions/lognormal_pareto.pyx'],
include_dirs=['/opt/local/include/',numpy.get_include()],
library_dirs=["/opt/local/lib"],
libraries=["gsl","gslcblas"]
)

setup(
    name = 'lognormal',
    ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
)

