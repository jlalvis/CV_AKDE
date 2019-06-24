#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:01:43 2018

@author: jorge
"""

# Generate Euler characteristic curve with a threshold value, t, taking data
# from a 3-dimensional array, X, where 3rd dimension is number of simulations.

from skimage.measure import regionprops, label
import numpy as np

def euler_curve(X, nsteps):
    v0 = np.amin(X)
    v1 = np.amax(X)
    t = np.linspace(v0, v1, nsteps)
    eulchar = np.zeros([np.size(X, 2), len(t)])
    for i in range(len(t)):
        for j in range(np.size(X, 2)):
            velbin = X[:,:,j] < t[i]
            velbin = 1*velbin
            props = regionprops(velbin)
            if props:
                eulchar[j,i] = euler_number(props)
            else:
                eulchar[j,i] = 0.0
    return eulchar

def euler_number(props):
    euler_array = props[0].filled_image != props[0].image
    _, numholes = label(euler_array, neighbors=8, return_num=True,
                   background=0)
    _, numobj = label(1*props[0].image, neighbors=8, return_num=True,
                      background=0)
    return numobj - numholes