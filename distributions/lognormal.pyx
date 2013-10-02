from math import sqrt
from math import log
import numpy as np

x,w = np.polynomial.hermite.hermgauss(40)
w /= sum(w)

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_pdf(double x,double sigma)
    
cdef double sigma_tilde =  sqrt(0.4)
cdef double rho = 1.

def f(double theta,double theta_):
    '''
    One period ahead distribution of theta
    '''
    return gsl_ran_gaussian_pdf(log(theta)-log(theta_),sigma_tilde)/theta
    
def f2(double theta, double theta_):
    '''
    Derivative with respect to theta_ of the one period ahead distribution of theta
    '''
    return f(theta,theta_)*(log(theta)-log(theta_))/(sigma_tilde**2*theta_)
    
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
    
def integration_nodes(theta_):
    return np.exp(x*sigma_tilde+log(theta_)),w
    
def integration_nodes0():
    return np.exp(x*sigma_tilde),w