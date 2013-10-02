# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:36:27 2013

@author: dgevans
"""
from cpp_interpolator import interpolate
from cpp_interpolator import interpolate_INFO
import numpy as np

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
    def __init__(self,types,order,k):
        '''
        Inits the interpolator object.  
            types: must be a list of strings.  
            order: related to the number of basis functions for each dimension
            k: used only for splines.  List of the dimension of spline
        '''
        self.INFO = interpolate_INFO(types,order,k)
        
    def __call__(self,X,y):
        '''
        Interpolates and returns a function object
        '''
        return interpolate(X,y,self.INFO)