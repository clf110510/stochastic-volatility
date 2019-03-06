#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:09:35 2019

@author: caolifeng
"""

from scipy.stats import norm
import numpy as np


class option_greeks:

    def __init__(self, S, K, r, t, vol, market_price, tol, type_flag):

        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.market_price = market_price
        self.vol = vol
        self.tol = tol
        self.type_flag = type_flag

    @staticmethod
    def N(z):

        return norm.cdf(z)

    def d1(self):

        d1 = (1.0/(self.vol * np.sqrt(self.t))) * \
            (np.log(self.S/self.K) + (self.r + 0.5 * self.vol**2.0) * self.t)

        return d1

    def d2(self):

        d2 = self.d1() - (self.vol * np.sqrt(self.t))

        return d2

# first order greeks

    def delta(self):
        """
        关于标的物价格一阶导
        """

        if self.type_flag == 'call':

            return option_greeks.N(self.d1())

        elif self.type_flag == 'put':

            return -option_greeks.N(-self.d1())

    def vega(self):
        """
        关于标的物波动率一阶导
        """

        return self.S*option_greeks.N(self.d1())*np.sqrt(self.t)

    def theta(self):
        """
        关于时间的一阶导
        """

        if self.type_flag == 'call':

            return (-self.S*self.vol*option_greeks.N(self.d1())/(2*np.sqrt(self.t)) -
                    self.r*self.K*np.exp(-self.r*self.t)*option_greeks.N(self.d2()))

        elif self.type_flag == 'put':

            return (-self.S*self.vol*option_greeks.N(self.d1())/(2*np.sqrt(self.t)) +
                    self.r*self.K*np.exp(-self.r*self.t)*option_greeks.N(-self.d2()))

    def rho(self):
        """
        关于利率一阶导
        """

        if self.type_flag == 'call':

            return self.K*self.t*np.exp(-self.r*self.t)*option_greeks.N(self.d2())

        elif self.type_flag == 'put':

            return -self.K*self.t*np.exp(-self.r*self.t)*option_greeks.N(-self.d2())

# second order greeks

    def gamma(self):
        """
        关于标的物价格的二阶导
        """

        return option_greeks.N(self.d1())/(self.S*self.vol*np.sqrt(self.t))

    def vanna(self):
        """
        关于波动率和标的物价格的二阶偏导数
        """

        return -option_greeks.N(self.d1())*self.d2()*self.vol

    def volga(self):
        """
        关于波动率的二阶导
        """

        return self.vega()*self.d1()*self.d2()/self.vol

    def charm(self):
        """
        关于标的物价格和时间的二阶偏导数
        """

        if self.type_flag == 'call':

            return (-self.N(self.d1())*(2*self.r*self.t-self.d2()*self.vol*np.sqrt(self.t)) /
                    2*self.t*self.vol*np.sqrt(self.t))
        if self.type_flag=='put':
            
            return (-self.N(self.d1())*(2*self.r*self.t-self.d2()*self.vol*np.sqrt(self.t)) /
                    2*self.t*self.vol*np.sqrt(self.t))
     
    def veta(self):
        """
        关于波动率和时间的二阶偏导数
        """
        
        return (-self.S*self.N(self.d1())*np.sqrt(self.t)*
                (self.r*self.d1()/(self.vol*np.sqrt(self.t))-(1+self.d1()*self.d2())/2*self.t))
     