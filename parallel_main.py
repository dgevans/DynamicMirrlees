# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:24:18 2013

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
from IPython.parallel import Client
c = Client()
v = c[:]
lv = c.load_balanced_view()


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

#build bgrids
theta_min = amin(lognormal.integration_nodes0()[0])
theta_max = amax(lognormal.integration_nodes0()[0])
lambda_1_diff_grid = linspace(0.,20.,10)#hstack((linspace(0,3,5)[:4],linspace(3,30,6)))#linspace(0.,.02,15)#hstack((linspace(0,10,8)[:7],linspace(10,30,8)))
lambda_2_grid = linspace(-10.,-0.01,10)#hstack((linspace(-10,-1.,4)[:3],linspace(-1.,-0.01,7)))#hstack((linspace(-30,-1.,9)[:8],linspace(-1.,-.1,7)))
theta_grid = exp(linspace(log(theta_min),log(theta_max),40))
X2 = vstack(makeGrid_generic([lambda_2_grid,theta_grid]))
X = vstack(makeGrid_generic([lambda_1_diff_grid,lambda_2_grid,theta_grid]))

def lambda_1_minf(x):
    xhat = array([x[0]*x[1],x[1]])
    return T.find_lambda_1_min(xhat)

tasks = []
for x in X2:
    tasks.append(lv.apply(lambda_1_minf,x))
    
lambda_1_min = zeros(len(X2))
for i,task in enumerate(tasks):
    lambda_1_min[i] = task.get()
    print i
    
interpolate2d = interpolator(['spline','spline'],[15,15],[1,1])
lambda_1_minf = interpolate2d(X2,lambda_1_min)

Xhat = copy(X)
for x in Xhat:
    x[0] += lambda_1_minf(x[1:])
    
tasks = []
for x in Xhat:
    tasks.append(lv.apply(lambda state:Vnew(state),x))

policies = zeros((len(Xhat),3))
for i,task in enumerate(tasks):
    policies[i,:] = task.get()
    print i

fout = open('sigma1.dat','w')
import cPickle
cPickle.dump((X,Xhat,X2,lambda_1_min,policies),fout)
fout.close()