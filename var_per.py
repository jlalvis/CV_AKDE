#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:15:22 2018

@author: jorge
"""

import numpy as np
from scipy.stats import circmean

# angdata in radians

def var_per(angdata):
    
    # compute circular mean
    n = len(angdata)
    angmean = circmean(angdata)
    
    # Compute variance for angle
    summ = 0.0
    for i in range(n):
        if angdata[i] < angmean:
            d1 = np.abs(angmean - angdata[i])
            d2 = np.abs(2*np.pi - angmean - angdata[i])
            if d1 < d2:
                summ = summ+d1**2
            else:
                summ = summ+d2**2
        else:
            d3 = np.abs(angdata[i] - angmean)
            d4 = np.abs(2*np.pi - angdata[i] + angmean)
            if d3 < d4:
                summ = summ+d3**2
            else:
                summ = summ+d4**2
            
    varor = summ/(n-1)
    return varor