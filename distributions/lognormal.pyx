from math import sqrt
from math import log
from scipy.special import erf
import numpy as np
from scipy.integrate import quad

x0,w0 = np.polynomial.hermite.hermgauss(20)
x,w = np.polynomial.hermite.hermgauss(40)


w /= sum(w)
w0 /= sum(w0)

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_pdf(double x,double sigma)
    
cpdef double sigma_tilde =  sqrt(0.4)
cpdef double rho = 1.

def f(double theta,double theta_):
    '''
    One period ahead distribution of theta
    '''
    return gsl_ran_gaussian_pdf(log(theta)-log(theta_),sigma_tilde)/theta
    
def F(double theta,double theta_):
    '''
    One period ahead cdf
    '''
    z = (log(theta)-log(theta_))/sigma_tilde
    return 0.5*(1 + erf(z/sqrt(2)))
    
def f2(double theta, double theta_):
    '''
    Derivative with respect to theta_ of the one period ahead distribution of theta
    '''
    return f(theta,theta_)*(log(theta)-log(theta_))/(sigma_tilde**2*theta_)
    
def F2(double theta,double theta_):
    '''
    Derivative of f2 from 0 to theta
    '''
    return -gsl_ran_gaussian_pdf(log(theta)-log(theta_),sigma_tilde)/theta_
    
def f0(double theta0):
    '''
    Initial distribution of productivities
    '''
    return gsl_ran_gaussian_pdf(log(theta0),sigma_tilde)/theta0
    
def alpha(double theta0):
    '''
    Pareto weights
    '''
    return 1.
    
def AlphaF0(double theta0):
    '''
    Compute the integral Integrate_0^theta_0 alpha(theta)f(theta)dtehta0
    '''
    alpha_f = lambda theta: alpha(theta)*f0(theta)
    return quad(alpha_f,0.,theta0)[0]
    
def F0(double theta0):
    '''
    Return the integral Integrate_0^theta0 f(theta) dtheta
    '''
    z = log(theta0)/sigma_tilde
    return 0.5*(1 + erf(z/sqrt(2)))
    
def integration_nodes(theta_):
    return np.exp(x*sigma_tilde+log(theta_)),w
    
def integration_nodes0():
    return np.exp(x0*sigma_tilde),w0