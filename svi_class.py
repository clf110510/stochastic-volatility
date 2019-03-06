#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:57:40 2018

@author: caolifeng
"""
from scipy.optimize import minimize
import numpy as np


class svi_model:
    def __init__(self, data, init_adc, init_msigma, tol):
        
        self.init_msigma = init_msigma
        self.init_adc = init_adc
        self.tol = tol
        self.data = data

    def forward_price(self):
        
        """
        s0为50etf当天价格
        折算到每个期权到期的远期价格
        """
        
        f = self.data.iloc[0]['50ETF_close'] * \
            np.exp(self.data.iloc[0]['R_avg7']*self.data.iloc[0]['maturity'])
        return f

    def outter_function(self, params):
       
        """
        外层函数
        """
       
        m, sigma = params
        sigma = max(0, sigma)
        adc_0 = self.init_adc
        f = self.forward_price()

        def inner_fun(params):
           
            """
            内层函数 用残差最小 拟合估计参数a d c，slsqp 
            注意对implied vol 进行转换 成 omega=vol**2*t
            """
            
            a, d, c = params
            error_sum = 0.0
            xi = np.log(self.data['strike']/f)
            y = (xi-m)/sigma
            z = np.sqrt(y**2+1)
            error_sum = np.sum(np.array(a + d * y + c * z -
                                        np.array(self.data['implied_vol'])**2*self.data.iloc[0]['maturity']) ** 2)
            return error_sum
        bnds = (
            (1e-10, max(np.array(self.data['implied_vol']))), (-4*sigma, 4*sigma), (0, 4*sigma))
        b = np.array(bnds, float)
        cons = (
            {'type': 'ineq', 'fun': lambda x: x[2]-abs(x[1])},
            {'type': 'ineq', 'fun': lambda x: 4*sigma-x[2]-abs(x[1])}
        )
        inner_res = minimize(inner_fun, adc_0, method='SLSQP', tol=1e-6)
        
        a_star, d_star, c_star = inner_res.x
        self._a_star, self._d_star, self._c_star = inner_res.x

        sum = 0.0
        xi = np.log(self.data['strike']/f)
        y = (xi-m)/sigma
        z = np.sqrt(y**2+1)
        sum = np.sum(np.array(a_star + d_star * y + c_star *
                              z - np.array(self.data['implied_vol'])**2*self.data.iloc[0]['maturity']) ** 2)
        return sum

    def optimization(self):
        
        """
        
        """

        outter_res = minimize(
            self.outter_function, self.init_msigma, method='Nelder-Mead', tol=self.tol)
        m_star, sigma_star = outter_res.x
        self._m_star, self._sigma_star = outter_res.x
        #obj = outter_res.fun
        calibrated_params = [self._a_star, self._d_star,
                             self._c_star, m_star, sigma_star]
        return calibrated_params

    def svi_vol(self):
       
        """
        
        """

        f = self.forward_price()
        self. optimization()
        xi = np.log(self.data['strike']/f)
        y = (xi-self._m_star)/self._sigma_star
        z = np.sqrt(y**2+1)
       
        omega = np.array(self._a_star + self._d_star * y + self._c_star * z)
        sigma = np.sqrt(omega/self.data.iloc[0]['maturity'])

        return sigma


