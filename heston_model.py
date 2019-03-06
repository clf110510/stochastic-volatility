#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:38:12 2019

@author: caolifeng
"""

import numpy as np

import scipy.integrate as spi

from scipy.optimize import minimize

from bs_vol_class import *

class heston_vol:
    def __init__(self, data,init_parms):
            self.data=data
            self.k=np.array(data['strike'])
            self.t=data.iloc[0]['maturity']
            self.s0=data.iloc[0]['50ETF_close']
            self.r=data.iloc[0]['R_avg7']
            self.price=np.array(data['close'])
            self.parms=init_parms
    
    def characteristic_function(self,phi, parms ,  K, type_flag):
        
        """
        
        """
        v0,kappa, theta, sigma, rho=parms

        
        if type==1:
            u = 0.5
            b = kappa - rho*sigma
        else: 
            u = -0.5
            b = kappa
        
        a = kappa*theta
        x = np.log(self.s0)
        d = np.sqrt((rho*sigma*phi*1j-b)**2 - sigma**2*(2*u*phi*1j-phi**2))
        g = (b-rho*sigma*phi*1j+d)/(b-rho*sigma*phi*1j-d)
        D = self.r*phi*1j*self.t + (a/sigma**2)*((b-rho*sigma*phi*1j+d)*self.t - 2*np.log((1-g*np.exp(d*self.t))/(1-g)))
        E = ((b-rho*sigma*phi*1j+d)/sigma**2)*(1-np.exp(d*self.t))/(1-g*np.exp(d*self.t))
        
        return np.exp(D + E*v0 + 1j*phi*x)

    def integral_function(self,phi,parms,  K, type_flag):
        
        integral = (np.exp(-1*1j*phi*np.log(K))*self.characteristic_function(phi, parms,  K, type_flag))    
       
        return integral
    
        
    def Heston_P_Value(self,parms, K, type_flag):
        
        """
        
        """
        ifun = lambda phi: self.integral_function(phi, parms,  K, type_flag)
        return 0.5 + (1/np.pi)*spi.quad(ifun, 0, 100)[0]


   
    def Heston_Call_Value(self, parms, K):
        
        """
        
        """
        
        a = self.s0*self.Heston_P_Value(parms, K, 1)
        b = K*np.exp(-self.r*self.t)*self.Heston_P_Value(parms,K, 2)
           
        return a-b

    def error_function(self,parms):
        """
        
        """
        v0,kappa, theta, sigma, rho  = parms

        heston_price=np.zeros(len(self.k),np.float)
        for i in range(0,len(self.k)):
            heston_price[i]=self.Heston_Call_Value( parms, self.k[i])
        error=np.sqrt(np.sum(self.price-heston_price)**2/len(self.price))
        
        return error

    def optimzation(self):
        """
        
        """
        
        opt = minimize(self.error_function, self.parms, method='Nelder-Mead', tol=1e-6)
        self.optimzation_parms=opt.x

        return opt.x
    
    
    def heston_vol(self):
        """
        
        """
        self.optimzation()
        heston_price=np.zeros(len(self.k),np.float)
        for i in range(0,len(self.k)):
             heston_price[i]=self.Heston_Call_Value( self.optimzation_parms, self.k[i])
        
        return heston_price     