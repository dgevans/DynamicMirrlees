# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:27:16 2013

@author: dgevans
"""
import numpy as np
cimport numpy
from scipy.optimize import root
from scipy.optimize import brentq

#parameters
sigma = 0.5
epsilon = 0.5
beta = 0.95
delta = 0.95

#functions of the distribution, to be set externaly
f = None #distribution needs to be set
f2 = None
f0 = None
alpha = None
integration_nodes = None


#utility functions
def U(c,l):
    if sigma == 1:
        return np.log(c)-l**(1.+1./epsilon)/(1.+1./epsilon)
    else:
        return (c**(1-sigma)-1)/(1-sigma)-l**(1.+1./epsilon)/(1.+1./epsilon)
        
def Uc(c,l):
    return c**(-sigma)

def Ul(c,l):
    return -l**(1./epsilon)

def Ull(c,l):
    return -l**(1./epsilon-1.)/epsilon
        

cdef numpy.ndarray resid = np.zeros(3)
cdef numpy.ndarray stateprime = np.zeros(3)
#Derivative matrix
D = np.eye(3,dtype=np.int)
#functions used in bellman equaion
cdef numpy.ndarray  __residuals(numpy.ndarray x,double u, double mu,double theta, double f_,Fs):
    '''
    Computes the residuls of the equations to solve to find quantities at any given
    theta.  Inputs are
    x: quantities
    y: current values of u and mu
    theta
    state: previou lagrange multipliers and theta as well as value function
    Value functions: distribution
    '''
    #unpack variables
    (Vf,wf,w2f) = Fs
    Uc,u0,lambda_2 = x

    #compute Uc
    temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
    tau = temp/(1.+temp)
    phi = f_/Uc
    stateprime[:] = [u0,lambda_2,theta]
    #compute residuals of equation
    res = np.zeros(3)
    if sigma==1:
        resid[0] = -np.log(Uc)
    else:
        resid[0] = (Uc**(1.-1./sigma)-1)/(1-sigma)
    resid[0] += -( theta*(1-tau)*Uc )**(1+epsilon)/(1.+1./epsilon) + beta*wf(stateprime) - u
    resid[1] = delta*Vf(stateprime,D[0]) - beta*wf(stateprime,D[0])/Uc-beta*(mu/f_)*w2f(stateprime,D[0])
    resid[2] = delta*Vf(stateprime,D[1]) - beta*wf(stateprime,D[1])/Uc-beta*(mu/f_)*w2f(stateprime,D[1])
    
    return resid
    
cdef double __residuals_end(double Uc,double u,double mu,double theta,double f_):
    '''
    Computes the residuls of the equations to solve to find quantities at any given
    theta.  Inputs are
    x: quantities
    y: current values of u and mu
    theta
    state: previou lagrange multipliers and theta as well as value function
    Value functions: distribution
    '''
    #Derivative matrix
    #compute Uc
    temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
    tau = temp/(1.+temp)
    #compute residuals of equation
    if sigma==1:
        res = -np.log(Uc)
    else:
        res = (Uc**(1.-1./sigma)-1)/(1-sigma)
    res -= ( theta*(1-tau)*Uc )**(1+epsilon)/(1.+1./epsilon) + u
    
    return res
    
def quantities(y,theta,state,lambda_1,Fs):
    '''
    Computes quantities
    '''
    if Fs == None:
        theta_ = state[2]
        u,mu = y
        f_ = f(theta,theta_)
        F = lambda Uc: __residuals_end(Uc,u,mu,theta,f_)
        Uc = bracket_and_solve(F,1.)
        temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
        tau = temp/(1.+temp)
        Vcont = 0
    else:
        #first find tax levels at this state
        lambda_1_,lambda_2_,theta_ = state
        (Vf,wf,w2f) = Fs
        u,mu = y
        f_ = f(theta,theta_)
        
        res = root(lambda x: __residuals(x,u,mu,theta,f_,Fs),[1.,-1.,1.])
        if not res.success:
            raise Exception('Could not find quantities at this theta')
        Uc,u0,lambda_2 = res.x
        stateprime = np.array([u0,lambda_2,theta])
        #compute derivatives
        temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
        tau = temp/(1.+temp) 
        Vcont = Vf(stateprime)
    c = Uc**(-1./sigma)
    l = (theta*(1-tau)*Uc)**epsilon
    return c-theta*l+beta*Vcont,c,l,tau
    
    
cdef double bracket_and_solve(F,double x0,double scale = 2.):
    '''
    Brackets the root of F and solves under assumption that F is decreasing
    '''
    cdef double xmin = x0
    cdef double xmax = x0
    if(F(x0) > 0.):
            xmax *= scale
            while(F(xmax) >0.):
                xmin = xmax
                xmax *= scale
    else:
        xmin /= scale
        while(F(xmin) <0.):
            xmax = xmin
            xmin /= scale
    return brentq(F,xmin,xmax)
    


def differentialEquation(y,theta,state,lambda_1,Fs):
    '''
    Computes the differential equation 
    '''
    if Fs == None:
        #first find tax levels at this state
        u0,lambda_2,theta_ = state
        u,mu = y
        f_ = f(theta,theta_)
        if mu > 0.:
            return np.zeros(2)
        
        F = lambda Uc: __residuals_end(Uc,u,mu,theta,f_)
        
        Uc = bracket_and_solve(F,1.)

        
        #compute derivatives
        temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
        tau = temp/(1.+temp)
        u_dot = (theta*(1-tau)*Uc)**(1+epsilon)
        mu_dot = f_*(1./Uc - lambda_1) - lambda_2*f2(theta,theta_)
        
        return np.hstack([u_dot,mu_dot])
    else:
        #first find tax levels at this state
        lambda_1_,lambda_2_,theta_ = state
        (Vf,wf,w2f) = Fs
        u,mu = y
        f_ = f(theta,theta_)
        
        res = root(lambda x: __residuals(x,u,mu,theta,f_,Fs),[1.,0.,1.])
        if not res.success:
            raise Exception('Could not find quantities at this theta')
        Uc,u0,lambda_2 = res.x
        stateprime = np.array([u0,lambda_2,theta])
        
        #compute derivatives
        temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
        tau = temp/(1.+temp) 
        
        u_dot = (theta*(1-tau)*Uc)**(1+epsilon)+beta*w2f(stateprime)
        mu_dot = f_*(1./Uc - lambda_1_) - lambda_2_*f2(theta,theta_)
        
        return np.hstack([u_dot,mu_dot])
        
x_guess = np.array([1.,  0.39997597, -64.60000853])

def differentialEquation0(y,theta,lambda_,Fs ):
    '''
    The differential equation for the time zero problem
    '''
    f_ = f0(theta)
    (Vf,wf,w2f) = Fs
    u,mu = y
    F= lambda x: __residuals(x,u,mu,theta,f_,Fs) 
    
    #def g(Uc):
    #    f = lambda x:F(np.hstack((Uc,x)))[1:]
    #    try:
    #        res = root(f,x_guess,method='krylov')
    #    except:
    #        raise Exception(f,F,y,theta,Uc)
    #    n_try = 0
    #    while not res.success:
    #        res = root(f,[5.*np.random.rand(),-100.*np.exp(np.random.randn())])
    #        n_try += 1
    #        if n_try > 10:
    #            raise Exception(f,F,y,theta,Uc)
    #    
    #    if res.success:
    #        x_guess[:] = res.x
    #        return F(np.hstack((Uc,res.x)))[0],res.x
        
        
    #Uc = bracket_and_solve(lambda Uc: g(Uc)[0],1.,scale=1.2)
    #    u0,lambda_2 = g(Uc)[1]
    res = root(F,x_guess,method='krylov')
    if not res.success:
        raise Exception(F,y,theta)
    x_guess[:] = res.x
    Uc,u0,lambda_2 = res.x
    

    stateprime[:] = [u0,lambda_2,theta]
    
    #compute derivatives
    temp = -Uc*theta*f_/((1.+1./epsilon)*mu)
    tau = temp/(1.+temp) 
    
    u_dot = (theta*(1-tau)*Uc)**(1+epsilon)+beta*w2f(stateprime)
    mu_dot = f_*(1./Uc -alpha(theta)*lambda_)
    return np.hstack([u_dot,mu_dot]),stateprime