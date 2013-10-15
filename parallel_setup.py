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
from distributions import lognormal
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
T = bellman(Para,-1./(1.-Para.sigma))
Vnew = T(None,None,None)#no continuationvalue functions

#build bgrids
theta_min = amin(lognormal.integration_nodes0()[0])
theta_max = amax(lognormal.integration_nodes0()[0])
lambda_1_diff_grid = hstack((linspace(0,1,8)[:7],linspace(1,5,8)))#linspace(0.,.02,15)#hstack((linspace(0,10,8)[:7],linspace(10,30,8)))
lambda_2_grid = hstack((linspace(-5.,-0.1,8)[:7],linspace(-0.1,-0.01,8)))#hstack((linspace(-30,-1.,9)[:8],linspace(-1.,-.1,7)))
theta_grid = exp(linspace(log(theta_min),log(theta_max),15))
X2 = vstack(makeGrid_generic([lambda_2_grid,theta_grid]))
X = vstack(makeGrid_generic([lambda_1_diff_grid,lambda_2_grid,theta_grid]))