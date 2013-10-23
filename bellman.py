# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:49:42 2013

@author: dgevans
"""
from scipy.integrate import ode
from scipy.optimize import brentq
from scipy.special import erf
from numpy import *

class bellman(object):
    '''
    Bellman equation object for the time 1 onwards problem
    '''
    def __init__(self,Para,u0_min):
        '''
        Constructs the bellman equation class, with object Para that holds parameter
        information
        '''
        self.Para = Para
        self.u0_min = u0_min 
        self.theta_min = 0.01
        
    def __call__(self,Vf,wf,w2f):
        '''
        Returns a new function based on the coninuation 
        '''
        self.Vf = Vf
        self.wf = wf
        self.w2f = w2f
        
        def Vnew(state):
            lambda_1,lambda_2_hat,theta_ = state
            lambda_2 = lambda_2_hat*theta_
            state_alt = array([lambda_1,lambda_2,theta_])
            return self.computeValue(state_alt)
        
        return Vnew
        
    def computeValue(self,state):
        '''
        Computes the value and continuation terms associated with a state
        '''
        #first compute lambda_1
        u0 = self.find_u0(state)
        
        return self.computeExpectations(state,u0)[:3]
        
    def computeExpectations(self,state,u0):
        '''
        Computes the expectations of various terms
        '''
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        theta_ = state[2]
        thetavec,w = self.Para.integration_nodes(theta_)
        yvec = self.integrateODE(state,thetavec,u0)
        if yvec[-1,1] > 0.:
            return nan*ones(4)
        obj = zeros((len(w),4))
        for i,theta in enumerate(thetavec):
            V,c,l,tau,Uc = self.Para.quantities(yvec[i],theta,state,Fs)
            obj[i,0] = V
            obj[i,1] = yvec[i,0]
            obj[i,2] = yvec[i,0]*self.Para.f2(theta,theta_)/self.Para.f(theta,theta_)
            obj[i,3] = 1./Uc
        return w.dot(obj)
        
    def getPolicies(self,state,thetavec,u0=None):
        '''
        Computes the expectations of various terms
        '''
        if u0 == None:
            u0 = self.find_u0(state)
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        yvec = self.integrateODE(state,thetavec,u0)
        if yvec[-1,1] > 0.:
            return nan*ones(4)
        pol = zeros((len(yvec),3))
        for i,theta in enumerate(thetavec):
            V,c,l,tau,Uc = self.Para.quantities(yvec[i],theta,state,Fs)
            pol[i,:] = [c,l,tau]
        return pol
        
    def integrateODE(self,state,thetavec,u0):
        '''
        Integrates ODE over the gridpoints theta
        '''
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        def dy_dtheta(theta,y):
            dy = self.Para.differentialEquation(y,theta,state,Fs)
            return dy
        
        lambda_1,lambda_2,theta_ = state
        mu_tilde = -lambda_2/theta_
        if thetavec[0] > self.theta_min:
            thetavec = hstack((self.theta_min,thetavec))
        y0 = self.getInitial_y(u0,mu_tilde,thetavec[0],state)
        
        r = ode(dy_dtheta).set_integrator('vode',rtol=1e-10,atol = 1e-10,nsteps=1000)
        r.set_initial_value(y0,thetavec[0])
        y = ones((len(thetavec),2))
        y[0] = y0
        for i,theta in enumerate(thetavec[1:]):
            y[i+1] = r.integrate(theta)
            if y[i+1,1] >0:
                break
            if not r.successful():
                print "ode failed"
                break
            
        if thetavec[0] == self.theta_min:
            return y[1:]
        else:
            return y
            
    
            
    def integrateODEverbose(self,state,thetavec,u0):
        '''
        Integrates ODE over the gridpoints theta
        '''
        if not self.Vf ==None:
            Fs = self.Vf,self.wf,self.w2f
        else:
            Fs = None
        def dy_dtheta(theta,y):
            print theta,y
            dy = self.Para.differentialEquation(y,theta,state,Fs)
            print dy
            return dy
        
        lambda_1,lambda_2,theta_ = state
        mu_tilde = -lambda_2/theta_
        if thetavec[0] > self.theta_min:
            thetavec = hstack((self.theta_min,thetavec))
        y0 = self.getInitial_y(u0,mu_tilde,thetavec[0],state)
        
        r = ode(dy_dtheta).set_integrator('vode',rtol=1e-10,atol = 1e-10,nsteps=1000)
        r.set_initial_value(y0,thetavec[0])
        y = ones((len(thetavec),2))
        y[0] = y0
        for i,theta in enumerate(thetavec[1:]):
            y[i+1] = r.integrate(theta)
            if y[i+1,1] >0:
                break
            if not r.successful():
                break
            
        if thetavec[0] == self.theta_min:
            return y[1:]
        else:
            return y
        
    def getInitial_y(self,u0,mu_tilde,theta_0,state):
        '''
        Computes initial y for ode
        '''
        lambda_1,lambda_2,theta_ = state
        F = lambda Uc: self.Para.residuals_end(Uc,u0,mu_tilde,0)
        Uc0 = bracket_and_solve(F,1.)
            
        mu0 = (1/Uc0-lambda_1)*self.Para.F(theta_0,theta_)-lambda_2*self.Para.F2(theta_0,theta_)
        
        return array([u0,mu0])
        
        
    
    def find_lambda_1_min(self,state_2):
        '''
        Computes the lambda_1 asssociated with u_min for lambda_2,theta
        '''
        lambda_2,theta_ = state_2
        thetavec,_ = self.Para.integration_nodes(theta_)
        u0_min = self.u0_min

        def f(lambda_1):
            state = hstack((lambda_1,state_2))
            res = self.computeExpectations(state,u0_min)[3]-lambda_1
            if isnan(res):
                return 1.
            else:
                return res
            
        lambda_1 =  bracket_and_solve(f,1.)
        while f(lambda_1) >0:
           lambda_1 *= 1.000001
       # state = hstack((lambda_1,state_2))
        return lambda_1
        
    def find_u0(self,state):
        '''
        Finds the u0 associated with the optimal allocation
        '''
        lambda_1,_,theta_ = state
        thetavec,_ = self.Para.integration_nodes(theta_)
        def f(u0_diff):
            u0 = self.u0_min+u0_diff
            res = lambda_1 - self.computeExpectations(state,u0)[3]
            if isnan(res):
                return -1.
            return res
        f0 = f(0.)
        if(f0<=0.) and f0 != -1.:
            return self.u0_min
        u0_diff = bracket_and_solve(f,1.)
        while self.integrateODE(state,thetavec,self.u0_min+u0_diff)[-1,1] > 0.:
            u0_diff *= 0.999999
        return self.u0_min+u0_diff
        
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
        
        return self.computeValue
    
    def computeValue(self,u0):
        '''
        Computes the value of the associated mu0
        '''
        Fs = self.Vf,self.wf,self.w2f
        lambda_ = self.find_lambda_(u0)
        
        return self.computeExpectation(u0,lambda_)[0]
        
    def computeExpectation(self,u0,lambda_):
        '''
        Computes expectations given u0 and lambda_
        '''
        Fs = self.Vf,self.wf,self.w2f
        thetavec,w = self.Para.integration_nodes0()
        yvec = self.integrateODE(thetavec,u0,lambda_)
        if yvec[-1,1] > 0.:
            return nan*ones(2)
        obj = zeros((len(w),2))
        for i,theta in enumerate(thetavec):
            V,c,l,tau,Uc = self.Para.quantities0(yvec[i],theta,Fs)
            obj[i,:] = [V,1./Uc]
        return w.dot(obj)
        
    def getPolicies(self,thetavec,u0,lambda_=None):
        '''
        Computes policies at vector thetavec
        '''
        if lambda_ == None:
            lambda_ = self.find_lambda_(u0)
        Fs = self.Vf,self.wf,self.w2f
        yvec = self.integrateODE(thetavec,u0,lambda_)
        if yvec[-1,1] > 0.:
            return nan*ones(2)
        policies = zeros((len(thetavec),3))
        stateprime = zeros((len(thetavec),3))
        Para = self.Para
        for i,theta in enumerate(thetavec):
            V,c,l,tau,Uc = self.Para.quantities0(yvec[i],theta,Fs)
            policies[i,:] = [c,l,tau]
            f_ = self.Para.f0(theta)
            lambda_2 = yvec[i,1]/(f_)*(Para.beta/Para.delta)
            lambda_1 = (1./Uc)*(Para.beta/Para.delta)
            stateprime[i,:] = [lambda_1,lambda_2,theta]
        return policies,stateprime

    def find_lambda_(self,u0):
        '''
        Computes the lambda_ associated with u0
        '''
        def f(lambda_):
            res =  self.computeExpectation(u0,lambda_)[1]-lambda_
            if isnan(res):
                return 1.
            return res
            
        lambda_ =  bracket_and_solve(f,1.)
        while f(lambda_) >0:
            lambda_ *= 1.000001
        return lambda_
        
    def integrateODE(self,thetavec,u0,lambda_):
        '''
        Integrates ODE over the gridpoints theta
        '''
        Fs = self.Vf,self.wf,self.w2f
        
        def dy_dtheta(theta,y):
            dy = self.Para.differentialEquation0(y,theta,lambda_,Fs)
            return dy
        
        y0 = self.getInitial_y(u0,lambda_,thetavec[0])
        r = ode(dy_dtheta).set_integrator('vode',rtol=1e-10,atol = 1e-10,nsteps=1000)
        r.set_initial_value(y0,thetavec[0])
        y = ones((len(thetavec),2))
        y[0] = y0
        for i,theta in enumerate(thetavec[1:]):
            y[i+1] = r.integrate(theta)
            if y[i+1,1] >0:
                break
            if not r.successful():
                break
        return y
        
    def getInitial_y(self,u0,lambda_,theta0):
        '''
        Computes initial y for ode
        '''
        sigma= self.Para.sigma
        if sigma == 1.:
            c0 = exp(u0)
            Uc0 = 1./c0
        else:
            Uc0 = ( (1-sigma)*u0 + 1 )**(sigma/(sigma-1))
        
            
        mu0 = 1./Uc0*self.Para.F0(theta0)-lambda_*self.Para.AlphaF0(theta0)
        
        return array([u0,mu0])
    
def bracket_and_solve(F,x0,scale = 2.):
    '''
    Brackets the root of F and solves under assumption that F is decreasing
    '''
    xmin = x0
    xmax = x0
    if(F(x0) > 0.):
            xmax *= 2.
            while(F(xmax) >0.):
                xmin = xmax
                xmax *= scale
    else:
        xmin *= 0.5
        while(F(xmin) <0.):
            xmax = xmin
            xmin /= scale
    return brentq(F,xmin,xmax)