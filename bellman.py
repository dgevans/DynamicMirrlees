# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:49:42 2013

@author: dgevans
"""
from scipy.integrate import ode
from scipy.optimize import brentq
from numpy import *

class bellman(object):
    '''
    Bellman equation object for the time 1 onwards problem
    '''
    def __init__(self,Para,mubar):
        '''
        Constructs the bellman equation class, with object Para that holds parameter
        information
        '''
        self.Para = Para
        self.mubar = mubar
        
    def __call__(self,Vf,wf,w2f):
        '''
        Returns a new function based on the coninuation 
        '''
        self.Vf = Vf
        self.wf = wf
        self.w2f = w2f
        
        return self.computeValue
        
    def computeValue(self,state):
        '''
        Computes the value and continuation terms associated with a state
        '''
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        #first compute lambda_1
        lambda_1 = self.find_lambda_1(state)
        #next compute history of u,mu
        theta_ = state[2]
        thetavec,w = self.Para.integration_nodes(theta_)
        yvec = self.integrateODE(state,thetavec,lambda_1)
        obj = zeros((len(w),3))
        for i,theta in enumerate(thetavec):
            V,c,l,tau = self.Para.quantities(yvec[i],theta,state,lambda_1,Fs)
            obj[i,0] = V
            obj[i,1] = yvec[i,0]
            obj[i,2] = yvec[i,0]*self.Para.f2(theta,theta_)/self.Para.f(theta,theta_)
        return hstack((w.dot(obj),lambda_1)) #returns V,w,w2,lambda_1
        
    def integrateODE(self,state,thetavec,lambda_1):
        '''
        Integrates ODE over the gridpoints theta
        '''
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        def dy_dtheta(theta,y):
            dy = self.Para.differentialEquation(y,theta,state,lambda_1,Fs)
            return dy
        u0 = state[0]
        r = ode(dy_dtheta)
        r.set_initial_value([u0,self.mubar],thetavec[0])
        y = ones((len(thetavec),2))
        y[0] = [u0,self.mubar]
        for i,theta in enumerate(thetavec[1:]):
            y[i+1] = r.integrate(theta)
            if y[i+1,1] >0:
                break
            if not r.successful():
                break
        return y
    
    def find_lambda_1(self,state):
        '''
        Computes the lambda_1 asssociated with the optimal allocation
        '''
        theta_ = state[2]
        thetavec,_ = self.Para.integration_nodes(theta_)
        def f(lambda_1):
            return self.integrateODE(state,thetavec,lambda_1)[-1,1]-self.mubar
            
        lambda_1 =  bracket_and_solve(f,1.)
        while f(lambda_1)-1e-7 >0:
            lambda_1 *= 1.000001
        return lambda_1
    
    def integrateODE_verbose(self,state,thetavec,lambda_1):
        '''
        Integrates ODE over the gridpoints theta
        '''
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        def dy_dtheta(theta,y):
            dy = self.Para.differentialEquation(y,theta,state,lambda_1,Fs)
            print theta,y,dy
            return dy
        u0 = state[0]
        r = ode(dy_dtheta)
        r.set_initial_value([u0,self.mubar],thetavec[0])
        y = ones((len(thetavec),2))
        y[0] = [u0,self.mubar]
        for i,theta in enumerate(thetavec[1:]):
            y[i+1] = r.integrate(theta)
            if y[i+1,1] >0:
                break
            if not r.successful():
                break
        return y#odeint(dy_dtheta,[u0,mu0],theta)
        
        
class time0_BellmanMap(object):
    '''
    The bellman map for the time zero bellman equation
    '''
    def __init__(self,Para,mubar):
        self.Para = Para
        self.mubar = mubar
        
    def __call__(self,Vf,wf,w2f):
        '''
        Returns a new function based on the coninuation 
        '''
        self.Vf = Vf
        self.wf = wf
        self.w2f = w2f
        
        pass
    
    def integrateODE(self,u0,thetavec,lambda_):
        '''
        Integrates ODE over the gridpoints theta
        '''
        Fs = self.Vf,self.wf,self.w2f
        
        def dy_dtheta(theta,y):
            dy = self.Para.differentialEquation0(y,theta,lambda_,Fs)
            print theta,y,dy
            return dy[0]
        r = ode(dy_dtheta)
        r.set_initial_value([u0,self.mubar],thetavec[0])
        y = ones((len(thetavec),2))
        y[0] = [u0,self.mubar]
        for i,theta in enumerate(thetavec[1:]):
            y[i+1] = r.integrate(theta)
            if y[i+1,1] >0:
                break
            if not r.successful():
                break
        return y
    
def bracket_and_solve(F,x0):
    '''
    Brackets the root of F and solves under assumption that F is decreasing
    '''
    xmin = x0
    xmax = x0
    if(F(x0) > 0.):
            xmax *= 2.
            while(F(xmax) >0.):
                xmin = xmax
                xmax *= 2.
    else:
        xmin *= 0.5
        while(F(xmin) <0.):
            xmax = xmin
            xmin *= 0.5
    return brentq(F,xmin,xmax)