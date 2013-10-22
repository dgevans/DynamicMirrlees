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
sigma = 1.
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
        


cdef numpy.ndarray stateprime = np.zeros(3)
cdef numpy.ndarray stateprime_2 = np.zeros(2)
cdef numpy.ndarray dy = np.zeros(2)
#Derivative matrix
D = np.eye(3,dtype=np.int)
#functions used in bellman equaion
cpdef double  __residuals(double Uc,double lambda_2_hat,double u, double mu_tilde,double theta, wf):
    '''
    Computes the residuls of the equations to solve to find quantities at any given
    theta.  Inputs are
    x: quantities
    y: current values of u and mu,
    theta
    state: previou lagrange multipliers and theta as well as value function
    Value functions: distribution
    '''
    #compute Uc
    temp = -Uc*((1.+1./epsilon)*mu_tilde)
    #tau = temp/(1.+temp)
    Ul = theta*Uc/(1+temp)
    lambda_1 = (beta/delta)/Uc
    stateprime[:] = [lambda_1,lambda_2_hat,theta]
    #compute residuals of equation
    cdef double res
    if sigma==1:
        res = -np.log(Uc)
    else:
        res = (Uc**(1.-1./sigma)-1)/(1-sigma)
    res += -( Ul )**(1+epsilon)/(1.+1./epsilon) + beta*wf(stateprime) - u
    
    return res
    
def residuals_end(double Uc,double u,double mu_tilde,double theta):
    '''
    wrapper
    '''
    return __residuals_end(Uc,u,mu_tilde,theta)
    
cpdef double __residuals_end(double Uc,double u,double mu_tilde,double theta):
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
    temp = -Uc*((1.+1./epsilon)*mu_tilde)
    #tau = temp/(1.+temp)
    Ul = theta*Uc/(1+temp)
    #compute residuals of equation
    if sigma==1:
        res = -np.log(Uc)
    else:
        res = (Uc**(1.-1./sigma)-1)/(1-sigma)
    res += -( Ul )**(1+epsilon)/(1.+1./epsilon) - u
    
    return res
    
def quantities(y,theta,state,Fs):
    '''
    Computes quantities
    '''
    if Fs == None:
        theta_ = state[2]
        u,mu = y
        f_ = f(theta,theta_)
        mu_tilde = mu/(theta*f_)
        F = lambda Uc: __residuals_end(Uc,u,mu_tilde,theta)
        
        Ucmax = 1e10**(sigma)
        Ucmin = 1e10**(-sigma)
        
        if(F(Ucmax) >= 0.):
            Uc = Ucmax
        elif( F(Ucmin) <=0.):
            Uc = Ucmin
        else:
            Uc = bracket_and_solve(F,1.)
            
        temp = -Uc*((1.+1./epsilon)*mu_tilde)
        tau = temp/(1.+temp)
        Vcont = 0
    else:
        #first find tax levels at this state
        theta_ = state[2]
        (Vf,wf,w2f) = Fs
        u,mu = y
        f_ = f(theta,theta_)
        mu_tilde = (mu/(theta*f_))
        lambda_2_hat = mu_tilde*(beta/delta)
        stateprime_2[:] = [lambda_2_hat,theta]

        Ucmax = (beta/delta)/Vf.lambda_1_min(stateprime_2)
        if Ucmax < 0.:
            Ucmax = 1e10**sigma
        
        Ucmin = 1e10**(-sigma)
        
        F = lambda Uc: __residuals(Uc,lambda_2_hat,u,mu_tilde,theta,wf)
        
        
        if(F(Ucmax) >= 0.):
            Uc = Ucmax
        elif( F(Ucmin) <=0.):
            Uc = Ucmin
        else:
            Uc = bracket_and_solve(F,Ucmax)
        
        lambda_1 = (beta/delta)/Uc
        stateprime[:] = [lambda_1,lambda_2_hat,theta]
        
        #compute derivatives
        temp = -Uc*((1.+1./epsilon)*mu_tilde)
        tau = temp/(1.+temp) 
        Vcont = Vf(stateprime)
    c = Uc**(-1./sigma)
    l = (theta*(1-tau)*Uc)**epsilon
    return c-theta*l+beta*Vcont,c,l,tau,Uc

def quantities0(y,theta,Fs):
    '''
    Get's the quantities associated with y = [u,mu] at point theta for the time
    0 solution
    '''
    u,mu = y
    if mu > 0.:
        return np.zeros(2)
    f_ = f0(theta)
    (Vf,wf,w2f) = Fs
    mu_tilde = mu/(theta*f_)
    lambda_2_hat = mu_tilde*(beta/delta)
    
    #solve the problem for this theta
    stateprime_2[:] = [lambda_2_hat,theta]

    Ucmax = (beta/delta)/Vf.lambda_1_min(stateprime_2)
    if Ucmax < 0.:
        Ucmax = 1e10**sigma
    
    Ucmin = 1e10**(-sigma)
    
    F = lambda Uc: __residuals(Uc,lambda_2_hat,u,mu_tilde,theta,wf)
    
    if(F(Ucmax) >= 0.):
        Uc = Ucmax
    elif( F(Ucmin) <=0.):
        Uc = Ucmin
    else:
        Uc = bracket_and_solve(F,Ucmax)
    
    lambda_1 = (beta/delta)/Uc
    stateprime[:] = [lambda_1,lambda_2_hat,theta]
    
    #compute derivatives
    temp = -Uc*((1.+1./epsilon)*mu_tilde)
    tau = temp/(1.+temp)
    Vcont = Vf(stateprime)
    c = Uc**(-1./sigma)
    l = (theta*(1-tau)*Uc)**epsilon
    return c-theta*l+beta*Vcont,c,l,tau,Uc
    
    
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
    


def differentialEquation(y,theta,state,Fs):
    '''
    Computes the differential equation 
    '''
    if Fs == None:
        #first find tax levels at this state
        lambda_1,lambda_2,theta_ = state
        u,mu = y
        f_ = f(theta,theta_)
        if mu > 0.:
            return np.zeros(2)
            
        mu_tilde = mu/(theta*f_)
        F = lambda Uc: __residuals_end(Uc,u,mu_tilde,theta)
        
        Ucmax = 1e10**(sigma)
        Ucmin = 1e10**(-sigma)
        
        if(F(Ucmax) >= 0.):
            Uc = Ucmax
        elif( F(Ucmin) <=0.):
            Uc = Ucmin
        else:
            Uc = bracket_and_solve(F,1.)

        
        #compute derivatives
        temp = -Uc*((1.+1./epsilon)*mu_tilde)
        Ul = theta*Uc/(1+temp)
        dy[0] = (Ul)**(1+epsilon)/theta
        dy[1] = f_*(1./Uc - lambda_1 - lambda_2*f2(theta,theta_)/f_)
        return dy
    else:
        #first find tax levels at this state
        lambda_1_,lambda_2_,theta_ = state
        (Vf,wf,w2f) = Fs
        u,mu = y
        f_ = f(theta,theta_)
        mu_tilde = mu/(theta*f_)
        lambda_2_hat = mu_tilde*(beta/delta)
        
        stateprime_2[:] = [lambda_2_hat,theta]

        Ucmax = (beta/delta)/Vf.lambda_1_min(stateprime_2)
        if Ucmax < 0.:
            Ucmax = 1e10**(sigma)
            
        Ucmin = 1e10**(-sigma)

        F = lambda Uc: __residuals(Uc,lambda_2_hat,u,mu_tilde,theta,wf)
        
        if(F(Ucmax) >= 0.):
            Uc = Ucmax
        elif( F(Ucmin) <=0.):
            Uc = Ucmin
        else:
            Uc = bracket_and_solve(F,Ucmax)
        
        lambda_1 = (beta/delta)/Uc
        stateprime[:] = [lambda_1,lambda_2_hat,theta]
        
        #compute derivatives
        temp = -Uc*((1.+1./epsilon)*mu_tilde)
        Ul = theta*Uc/(1+temp)
        
        dy[0] = (Ul)**(1+epsilon)/theta+beta*w2f(stateprime) #udot
        dy[1] = f_*(1./Uc - lambda_1_ - lambda_2_*f2(theta,theta_)/f_)#mu_dot
        
        return dy
        

def differentialEquation0(y,theta,lambda_,Fs ):
    '''
    The differential equation for the time zero problem
    '''
    u,mu = y
    if mu > 0.:
        return np.zeros(2)
    f_ = f0(theta)
    (Vf,wf,w2f) = Fs
    mu_tilde = mu/(theta*f_)
    lambda_2_hat = mu_tilde*(beta/delta)
    
    #solve the problem for this theta
    stateprime_2[:] = [lambda_2_hat,theta]

    Ucmax = (beta/delta)/Vf.lambda_1_min(stateprime_2)
    if Ucmax < 0.:
        Ucmax = 1e10**(sigma)
    Ucmin = 1e10**(-sigma)

    
    F = lambda Uc: __residuals(Uc,lambda_2_hat,u,mu_tilde,theta,wf)
    
    if(F(Ucmax) >= 0.):
        Uc = Ucmax
    elif( F(Ucmin) <=0.):
        Uc = Ucmin
    else:
        Uc = bracket_and_solve(F,Ucmax)
    
    lambda_1 = (beta/delta)/Uc
    stateprime[:] = [lambda_1,lambda_2_hat,theta]
    
    #compute derivatives
    temp = -Uc*((1.+1./epsilon)*mu_tilde)
    #tau = temp/(1+temp)
    Ul = theta*Uc/(1+temp)
    
    dy[0] = (Ul)**(1+epsilon)/theta + beta*w2f(stateprime)
    dy[1] = f_*(1./Uc -alpha(theta)*lambda_)
    return dy
    