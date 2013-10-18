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

#find minimum lambda_1
lambda_1_min = zeros(len(X2))
for i,x in enumerate(X2):
    xhat = hstack((x[0]*x[1],x[1]))
    lambda_1_min[i] = T.find_lambda_1_min(xhat)
    print i
interpolate2d = interpolator(['spline','spline'],[15,15],[1,1])
lambda_1_minf = interpolate2d(X2,lambda_1_min)

Xhat = copy(X)
for x in Xhat:
    x[0] += lambda_1_minf(x[1:])
#solve time 1 problem at each grid point
policies = zeros((len(Xhat),3))
for i,x in enumerate(Xhat):
    policies[i,:] = Vnew(x)
    print i

#fit  time 1 value function
interpolate = interpolator(['spline','spline','spline'],[15,15,15],[1,1,1],lambda_1_minf)
Vf= interpolate(X,policies[:,0])
wf= interpolate(X,policies[:,1])
w2f= interpolate(X,policies[:,2])

#bellman map for time 0
T0= time0_BellmanMap(Para,-1)
V0 =  T0(Vf,wf,w2f)
brentq(V0,-2.1,-2)#find root for V0










#==============================================================================
# X = vstack(makeGrid_generic([ugrid,lambda_2_grid,theta_grid]))
# 
# interpolate = interpolator(['spline','spline','spline'],[20,20,10],[2,2,1])
# Vf= interpolate(X,policies[:,0])
# wf= interpolate(X,policies[:,1])
# w2f= interpolate(X,policies[:,2])
# Fs = (Vf,wf,w2f)
# 
# T = time0_BellmanMap(Para,-6.049365142868869e-13)
# T(Vf,wf,w2f)
# try:
#     T.integrateODE(-1.,lognormal.integration_nodes0()[0],6.)
# except Exception as inst:
#   F,y,theta = inst.args
#==============================================================================