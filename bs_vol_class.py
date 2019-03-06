#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:46:15 2019

@author: caolifeng
"""

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import newton
import numpy as np


class bs_implied_vol:
    
    
    def __init__(self, S, K, r, t, market_price, tol, type_flag, method):
    
        self.S = S
        self.K = K
        self.r=r
        self.t=t
        self.market_price=market_price
        self.tol = tol
        self.type_flag = type_flag    
        self.method=method
    
    @staticmethod
    def N(z):

        return norm.cdf(z)   
    
    
    
    def black_scholes_value(self,vol):
        """
        
        """
        d1 = (1.0/(vol * np.sqrt(self.t))) * (np.log(self.S/self.K) + (self.r + 0.5 * vol**2.0) * self.t)
        d2 = d1 - (vol * np.sqrt(self.t))
       
        if self.type_flag=='call':
           
            return     bs_implied_vol.N(d1) * self.S - bs_implied_vol.N(d2) * self.K * np.exp(-self.r * self.t)
        
        
        if self.type_flag=='put':
        
            return      bs_implied_vol.N(-d2) * self.K * np.exp(-self.r * self.t) - bs_implied_vol.N(-d1) * self.S
        

    
        
    def implied_volatility(self):
        """
        
        """
        
        def object_fun(vol):
            
        
            
            return self.market_price-self.black_scholes_value(vol)
        
        
        if self.method=='brents':
        
            try:
                result = brentq(object_fun, a=-3.0, b=3.0, xtol=self.tol)
        
                return 0.01 if result <= self.tol else result
        
            except ValueError:
                return np.nan
        
        elif self.method=='newton':
            
            try:
                result = newton(object_fun, x0=0.2, tol=self.tol)
        
                return 0.01 if result <= self.tol else result
        
            except ValueError:
                return np.nan
            
            
        