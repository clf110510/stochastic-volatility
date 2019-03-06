#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:59:29 2019

@author: caolifeng
"""

import numpy as np
from scipy.optimize import minimize

class sabr_vol:
    
    def __init__(self,data,beta,shift,strike):
        self.data=data
        self.k=np.array(data['strike'])
        self.t=data.iloc[0]['maturity']
        self.beta=beta
        self.v_sln=np.array(data['implied_vol'])*100
        self.shift=shift
        self.strike=strike
        
    def forward_price(self):
        
        """
        s0为50etf当天价格
        折算到每个期权到期的远期价格
        """
        
        f = self.data.iloc[0]['50ETF_close'] * \
            np.exp(self.data.iloc[0]['R_avg7']*self.data.iloc[0]['maturity'])
        return f    
    
    def sabr_log_vol(self,k,alpha,rho,volvar):
        f=self.forward_price()
        f=f+self.shift
        
        # strike为负或forward price为负
        if k <= 0 or f <= 0:
            return 0        
        # f等于k的情况，用精度数代替0
        eps = 1e-07
        logfk = np.log(f/k)
        fk = (f*k)**(1-self.beta)
        a = (1-self.beta)**2*alpha**2/(24*fk)
        b = 0.25*rho*self.beta*volvar*alpha/fk**0.5
        c = (2-3*rho**2)*volvar**2/24
        d = fk**0.5
        v = (1-self.beta)**2*logfk**2/24
        w = (1-self.beta)**4*logfk**4/1920
        z = volvar*fk**0.5*logfk/alpha
        
        def x(rho, z):
            """
            返回函数x基于 sabr lognormal vol expansion
            """
            a = (1-2*rho*z+z**2)**0.5+z-rho
            b = 1-rho
            return np.log(a/b)   
        
        
        if abs(z) > eps:
              vz = alpha * z * (1 + (a+b+c) * self.t) / (d * (1+v+w)*x(rho, z))
              return vz
        # f=k
        else:
            v0 = alpha*(1+(a+b+c)*self.t)/(d*(a+v+w))
            return v0
     
    def calibration(self):
        
        """
        校准sabr模型的参数alpha，rho，volvar
    
        基于bs model算出的volatility smile （strike和volatility两个维度）
        返回一个sabr模型的参数元组
        """
        
        def vol_square_error(x):
            
            vols = [self.sabr_log_vol(k_+self.shift,  x[0],
                                  x[1], x[2])*100 for k_ in self.k]
            return sum((vols-self.v_sln)**2)
        x0 = np.array([0.01, 0.00, 0.1])
        bounds = [(0.0001, None), (-0.9999, 0.9999), (0.0001, None)]
        res = minimize(vol_square_error, x0, method='L-BFGS-B', bounds=bounds)
        self.alpha, self.rho, self.volvar = res.x
        return [self.alpha, self.rho, self.volvar]
    
    def get_sabr_value(self):
        
        self.calibration()
        k=self.strike+self.shift
        f=self.forward_price()+self.shift
        # strike为负或forward price为负
        if self.strike <= 0 or f <= 0:
            return 0        
        # f等于k的情况，用精度数代替0
        eps = 1e-07
        logfk = np.log(f/k)
        fk = (f*k)**(1-self.beta)
        a = (1-self.beta)**2*self.alpha**2/(24*fk)
        b = 0.25*self.rho*self.beta*self.volvar*self.alpha/fk**0.5
        c = (2-3*self.rho**2)*self.volvar**2/24
        d = fk**0.5
        v = (1-self.beta)**2*logfk**2/24
        w = (1-self.beta)**4*logfk**4/1920
        z = self.volvar*fk**0.5*logfk/self.alpha
        rho=self.rho
        def x(rho,z):
            """
            返回函数x基于 sabr lognormal vol expansion
            """
            a = (1-2*rho*z+z**2)**0.5+z-rho
            b = 1-rho
            return np.log(a/b)   
        
        
        if abs(z) > eps:
              vz = self.alpha * z * (1 + (a+b+c) * self.t) / (d * (1+v+w)*x( rho,z))
              return vz
        # f=k
        else:
            v0 = self.alpha*(1+(a+b+c)*self.t)/(d*(a+v+w))
            return v0
    