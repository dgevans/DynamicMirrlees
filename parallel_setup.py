# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:23:17 2013

@author: dgevans
"""

from numpy import *
from numpy import polynomial
import primitives.CES_parameters_c as Para
from bellman import bellman
from scipy.optimize import brentq
import distributions.lognormal_pareto as lognormal
from utilities import makeGrid_generic
from utilities import interpolator
from scipy.optimize import root
from scipy.integrate import ode
from bellman import time0_BellmanMap
from copy import copy


Para.f = lognormal.f
Para.F = lognormal.F
Para.f2 = lognormal.f2
Para.F2 = lognormal.F2
Para.integration_nodes = lognormal.integration_nodes
Para.integration_nodes0 = lognormal.integration_nodes0
Para.f0 = lognormal.f0
Para.F0 = lognormal.F0
Para.AlphaF0 = lognormal.AlphaF0
Para.alpha = lognormal.alpha

#setup bellman map
T = bellman(Para,-100.)
Vnew = T(None,None,None)#no continuationvalue functions
