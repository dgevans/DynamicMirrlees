# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:36:27 2013

@author: dgevans
"""
from cpp_interpolator import interpolate
from cpp_interpolator import interpolate_INFO
import numpy as np
from copy import copy

def makeGrid_generic(x):
    '''
    Makes a grid to interpolate on from list of x's
    '''
    N = 1
    n = []
    n.append(N)
    for i in range(0,len(x)):
        N *= len(x[i])
        n.append(N)
    X =[]
    for i in range(0,N):
        temp = i
        temp_X = []
        for j in range(len(x)-1,-1,-1):
            temp_X.append(x[j][temp/n[j]])
            temp %= n[j]
        temp_X.reverse()
        X.append(temp_X)
    return X
    
    
class interpolator(object):
    '''
    Factory object used to fit interpolated data
    '''
    def __init__(self,types,order,k,lambda_1_min=None):
        '''
        Inits the interpolator object.  
            types: must be a list of strings.  
            order: related to the number of basis functions for each dimension
            k: used only for splines.  List of the dimension of spline
        '''
        self.INFO = interpolate_INFO(types,order,k)
        self.lambda_1_min = lambda_1_min
        self.theta_min = 0.1
        
    def __call__(self,X,y):
        '''
        Interpolates and returns a function object
        '''
        if self.lambda_1_min == None:
            return interpolate(X,y,self.INFO)
        f =  interpolate(X,y,self.INFO)
        return interpolated_function(f,self.lambda_1_min)
        
class interpolated_function(object):
    '''
    A wrapper for interpolated function which takes into account lambda_1_min
    '''
    def __init__(self,f,lambda_1_min):
        '''
        Initializes taking a function who's first argument is lambda_1_diff
        and a function lambda_1_min
        '''
        self.f = f
        self.lambda_1_min = lambda_1_min
        
    def __call__(self,X):
        '''
        Evaluate at a matrix X
        '''
        Xhat = copy(np.atleast_2d(X))
        for x in Xhat:
            x[0] -= self.lambda_1_min(x[1:])
        return self.f(Xhat)