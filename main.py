# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:07:15 2013

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

Para.f = lognormal.f
Para.f2 = lognormal.f2
Para.integration_nodes = lognormal.integration_nodes
Para.f0 = lognormal.f0
Para.alpha = lognormal.alpha

T = bellman(Para,-1e-7)
Vnew = T(None,None,None)

theta_min = amin(lognormal.integration_nodes0()[0])
theta_max = amax(lognormal.integration_nodes0()[0])
ugrid = hstack((linspace(-.1,10.,11)[:10],linspace(10.,200.,10)))
lambda_2_grid = hstack((linspace(-80,-1.,16)[:15],linspace(-1.,-.1,5)))
theta_grid = linspace(theta_min,theta_max,10)

X = vstack(makeGrid_generic([ugrid,lambda_2_grid,theta_grid]))

interpolate = interpolator(['spline','spline','spline'],[20,20,10],[2,2,1])
Vf= interpolate(X,policies[:,0])
wf= interpolate(X,policies[:,1])
w2f= interpolate(X,policies[:,2])
Fs = (Vf,wf,w2f)

T = time0_BellmanMap(Para,-6.049365142868869e-13)
T(Vf,wf,w2f)
try:
    T.integrateODE(-1.,lognormal.integration_nodes0()[0],6.)
except Exception as inst:
    F,y,theta = inst.args