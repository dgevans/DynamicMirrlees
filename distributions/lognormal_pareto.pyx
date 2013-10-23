from math import sqrt
from math import log
from scipy.special import erf
import numpy as np
from scipy.integrate import quad


x,w = np.polynomial.hermite.hermgauss(30)

theta_bar = 3.




w /= sum(w)

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_pdf(double x,double sigma)
cdef extern from "gsl/gsl_integration.h":
    ctypedef struct gsl_integration_glfixed_table:
        pass
    
    gsl_integration_glfixed_table * gsl_integration_glfixed_table_alloc (int n)
    
    int gsl_integration_glfixed_point (double a, double b, size_t i, double * xi, double * wi, const gsl_integration_glfixed_table * t)
    

def gl_nodes(a,b,n):
    cdef gsl_integration_glfixed_table * table = gsl_integration_glfixed_table_alloc(n)
    cdef double xi,wi
    x = np.zeros(n)
    w = np.zeros(n)
    for i in range(n):
        gsl_integration_glfixed_point (a, b, i, &xi, &wi, table)
        x[i] = xi
        w[i] = wi
    return x,w
    
    
cpdef double sigma_tilde =  sqrt(0.5)
cpdef double rho = 1.


def pareto_params(double theta_bar):
    '''
    Computes pareto weight of tail starting at theta_bar
    '''
    f0bar = (gsl_ran_gaussian_pdf(log(theta_bar),sigma_tilde)/theta_bar)
    f0bar_1 =-1./theta_bar*f0bar-log(theta_bar)/(sigma_tilde**2 * theta_bar)*f0bar
    
    a =  -theta_bar*f0bar_1/f0bar-1
    theta_m = (f0bar*theta_bar**(a+1)/a)**(1./a)
    z = log(theta_bar)/sigma_tilde
    scale = 0.5*(1 + erf(z/sqrt(2))) + (theta_m/theta_bar)**a
    return [a,theta_m,scale]
    
a,theta_m,scale = pareto_params(theta_bar)
    
    
def mutilde_dot(double theta,double mu_tilde, double Uc,double lambda_1, double lambda_2, double theta_):
    '''
    Computes the derivative of mu_tilde
    '''
    return (1./Uc - lambda_1 + (log(theta/theta_)/sigma_tilde**2)*(mu_tilde - lambda_2))/theta
    
def mutilde0_dot(double theta,double mu_tilde, double Uc,double lambda_):
    '''
    Computes the derivative of mu_tilde
    '''
    if theta< theta_bar:
        return (1./Uc - alpha(theta)*lambda_ + (log(theta)/sigma_tilde**2)*mu_tilde )/theta 
    else:
        return (1./Uc - alpha(theta)*lambda_ + a*mu_tilde )/theta 

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
    if theta0 <= theta_bar:
        return (gsl_ran_gaussian_pdf(log(theta0),sigma_tilde)/theta0)/scale
    else:
        return a* (theta_m**a)/(theta0**(a+1.)) / scale
    
def alpha(double theta0):
    '''
    Pareto weights
    '''
    return 1.
    #return 1./theta0/1.265262100649486 #normalized so average pareto weight is 1.
    
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
    if theta0 < theta_bar:
        z = log(theta0)/sigma_tilde
        return 0.5*(1 + erf(z/sqrt(2)))/scale
    else:
        z_bar = log(theta_bar)/sigma_tilde
        return (0.5*(1 + erf(z_bar/sqrt(2))) + (theta_m/theta_bar)**a - (theta_m/theta0)**a )/scale  
    
def integration_nodes(theta_):
    return np.exp(x*sigma_tilde+log(theta_)),w
    
    
x0,w0 = gl_nodes(0.033129137639212609,30.184908852452924,100)
for i,theta in enumerate(x0):
    w0[i] *= f0(theta)
w0 /= sum(w0)

def integration_nodes0():
    return x0,w0